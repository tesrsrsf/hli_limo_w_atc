import textwrap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
import re
import pickle
from typing import List, Dict, Any, Tuple, Optional, Callable
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import app_extract_single_ppl

# =========================
# GLOBAL CONFIG
# =========================
APP_TITLE = "LiMO-ATC Code Detector Demo"
APP_SUBTITLE = "Line-level AI code detector"

DEMO_MODE_DEFAULT = False

# In the future, if you want to switch to a different model/checkpoint, you can configure it here
MODEL_CHECKPOINT_PATH = "./ckpt/codenet(python)_gemini_hybrid_line_best_f1.pt"

# Default example code
DEFAULT_CODE_EXAMPLE = """\
from collections import defaultdict, deque

n, m = map(int, input().split())
xy = []
node = defaultdict(int)
branch = defaultdict(list)
dp = [0]*(n+1) 

for _ in range(m):
    x, y = map(int, input().split())
    xy.append([x, y])
    node[y] += 1 #入り込むノードの数
    branch[x].append(y) # 枝分かれ先


res = []
queue = deque([i for i in range(1, n+1) if node[i]==0])

while queue:
    v = queue.popleft()
    for i in branch[v]:
        node[i] -= 1
        dp[i] = max(dp[i], dp[v]+1)
        if node[i] == 0:
            queue.append(i)


for i in range(m):
    dp[xy[i][1]] = max(dp[xy[i][1]], dp[xy[i][0]]+1)
print(max(dp))
"""


MAX_CODE_LINES = 400  # Maximum number of lines to display/process in the UI


# Highlight colors
AI_BG_COLOR = "#ffc4c4"
HUMAN_BG_COLOR = "#b5eaff"
AI_TEXT_COLOR = "#8E3535"
HUMAN_TEXT_COLOR = "#125C65"


# ============ AT feature global config ============

# Model names can be overridden by environment variables
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "codellama/CodeLlama-7b-Instruct-hf")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "microsoft/codebert-base")

# Generation / encoding related
N_TASKS_PER_SEGMENT = int(os.environ.get("N_TASKS_PER_SEGMENT", "1"))   # Usually 1 is enough
LLM_MAX_NEW_TOKENS  = int(os.environ.get("LLM_MAX_NEW_TOKENS", "64"))
LLM_TEMPERATURE     = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
EMBED_MAX_LEN       = int(os.environ.get("EMBED_MAX_LEN", "256"))

# PCA path (segment-level PCA)
SEG_PCA_PATH = os.environ.get("SEG_PCA_PATH", "./pca/segment_python_atfeature_pca128.pkl")
AT_DIM       = int(os.environ.get("AT_DIM", "128"))

# Segmentation related
MIN_SEG_LINES = int(os.environ.get("MIN_SEG_LINES", "1"))  # Minimum segment lines
LANGUAGE_HINT = os.environ.get("LANGUAGE_HINT", "python")

# Random seed
SEED = int(os.environ.get("SEED", "1234"))
np.random.seed(SEED)
torch.manual_seed(SEED)

# GPU settings
GPU_ID_LLAMA = int(os.environ.get("GPU_ID_LLAMA", "2"))
GPU_ID_BERT = int(os.environ.get("GPU_ID_BERT", "3"))
GPU_MEM_LLAMA = os.environ.get("GPU_MEM_LLAMA", "12GiB")
GPU_MEM_BERT = os.environ.get("GPU_MEM_BERT", "12GiB")


# ============ Segmenter: same segmentation logic as new_2ds ============

def _is_blank(s: str) -> bool:
    return len(s.strip()) == 0


def _brace_delta(s: str) -> int:
    return s.count("{") - s.count("}")


def segment_code(lines: List[str], lang_hint: Optional[str] = None, min_seg_lines: int = 1) -> List[Tuple[int, int]]:
    """
    Returns a list of (start_idx, end_idx) tuples, where indices are **0-based and inclusive of end**.

    Rules:
        - Skip leading consecutive blank lines;
        - Split a segment when encountering a blank line and the brace depth returns to 0;
        - Also end at EOF;
        - Finally, if min_seg_lines > 1, merge segments that are too short.
    """
    n = len(lines)
    segs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        # Skip leading blank lines
        while i < n and _is_blank(lines[i]):
            i += 1
        if i >= n:
            break
        start = i
        depth = 0
        j = i
        while j < n:
            s = lines[j]
            depth += _brace_delta(s)
            if _is_blank(s) and depth == 0 and j > start:
                # Remove trailing blank lines
                k = j - 1
                while k >= start and _is_blank(lines[k]):
                    k -= 1
                if k >= start:
                    segs.append((start, k))
                j += 1
                break
            j += 1
        else:
            # To EOF
            k = n - 1
            while k >= start and _is_blank(lines[k]):
                k -= 1
            if k >= start:
                segs.append((start, k))
            break
        i = j

    # Merge segments that are too short
    if min_seg_lines > 1 and segs:
        merged: List[Tuple[int, int]] = []
        buf_start, buf_end = segs[0]
        for (s, e) in segs[1:]:
            if (buf_end - buf_start + 1) < min_seg_lines:
                # too short, merge
                buf_end = e
            else:
                merged.append((buf_start, buf_end))
                buf_start, buf_end = s, e
        merged.append((buf_start, buf_end))
        segs = merged

    return segs


# ============ TaskGenerator: LLM generates segment-level task descriptions ============

class TaskGenerator:
    """Use LLM to turn a code segment into one sentence describing its design intent."""

    def __init__(self):
        max_memory = {GPU_ID_LLAMA: GPU_MEM_LLAMA}

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.gen = torch.Generator(device=str(self.model.device)).manual_seed(SEED)

    @staticmethod
    def prompt_segment(
        segment_text: str,
        language_hint: Optional[str] = None,
        context_text: str = "",
    ) -> str:
        lang = language_hint or ""
        prog_ctx = (
            f"\n<program_context>\n{context_text}\n</program_context>\n"
            if context_text
            else "\n"
        )

        return (
            "You are a careful code reviewer analyzing a full program.\n"
            "Your goal is to **infer the design intent** behind each code segment, not to restate its actions.\n"
            "Read the overall program to understand its purpose, then hypothesize **why the TARGET SEGMENT exists**, what issue it solves, or what design goal it fulfills.\n\n"
            "Output: ONE concise English sentence (<=60 words) describing the segment\'s intent or motivation within the program.\n"
            "Prefer action verbs (e.g., validate, parse, dispatch, fetch, cache, format). If context is insufficient, make a\n"
            "best-effort guess based on available information.\n"
            "Avoid implementation detail or pseudocode; focus on purpose and reasoning.\n\n"
            f"<language>{lang}</language>\n"
            f"{prog_ctx}"
            f"<target_segment>\n{segment_text}\n</target_segment>\n"
        )

    def wrap_chat(self, user_text: str) -> str:
        """把 user_text 包进 chat 模板（兼容 llama / instruct 系模型）。"""
        if hasattr(self.tokenizer, "apply_chat_template") and callable(
            self.tokenizer.apply_chat_template
        ):
            msgs = [{"role": "user", "content": user_text}]
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        return f"[INST] {user_text.strip()} [/INST]"

    @torch.inference_mode()
    def gen_tasks(
        self,
        segment_text: str,
        language_hint: Optional[str] = None,
        context_text: str = "",
    ) -> List[str]:
        prompt = self.prompt_segment(segment_text, language_hint, context_text)
        full = self.wrap_chat(prompt)
        in_ids = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **in_ids,
            do_sample=True,
            temperature=LLM_TEMPERATURE,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            num_return_sequences=N_TASKS_PER_SEGMENT,
            pad_token_id=self.tokenizer.pad_token_id,
            top_p=0.95,
            generator=self.gen,
        )
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        results: List[str] = []
        for s in texts:
            # Remove instruction wrappers
            if "### Response:" in s:
                s = s.split("### Response:", 1)[-1]
            elif "[/INST]" in s:
                s = s.split("[/INST]", 1)[-1]

            s = s.strip()
            s = re.sub(r"\s+", " ", s)
            if s:
                results.append(s)

        # Deduplicate + take the first N (usually 1)
        seen, uniq = set(), []
        for t in results:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        if not uniq:
            uniq = ["Summarize the high-level purpose of this code segment."]
        return uniq[: max(1, N_TASKS_PER_SEGMENT)]


# ============ TaskEmbedder: CodeBERT 768-d mean pooling ============
class TaskEmbedder:
    def __init__(self):
        max_memory = {GPU_ID_BERT: GPU_MEM_BERT}
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, use_fast=True)
        self.model = AutoModel.from_pretrained(
            EMBED_MODEL_NAME,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=dtype,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Input multiple task sentences, output a numpy array of shape [len(texts), 768].
        Use last_hidden_state + attention_mask for mean pooling.
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=EMBED_MAX_LEN,
            return_tensors="pt",
        ).to(self.model.device)
        out = self.model(**enc)
        last = out.last_hidden_state                 # [B, L, H]
        mask = enc["attention_mask"].unsqueeze(-1)   # [B, L, 1]
        summed = (last * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        mean_pool = summed / denom                   # [B, H]
        return mean_pool.float().cpu().numpy()       # float32


# ============ Segment-level PCA: from 768 → 128 ============
def load_segment_pca(pca_path: Optional[str] = None) -> Optional[IncrementalPCA]:
    path = pca_path or SEG_PCA_PATH
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def project_at_vecs(vecs_768: np.ndarray, pca: Optional[IncrementalPCA]) -> np.ndarray:
    vecs_768 = np.asarray(vecs_768, dtype=np.float32)
    if pca is not None:
        return pca.transform(vecs_768).astype(np.float32)

    # fallback: directly take the first AT_DIM dimensions, zero-pad if not enough
    n, d = vecs_768.shape
    out = np.zeros((n, AT_DIM), dtype=np.float32)
    take = min(d, AT_DIM)
    out[:, :take] = vecs_768[:, :take]
    return out


# ============ segment-level AT features ============

def compute_segment_atfeatures_for_code(code: str, language_hint: str = LANGUAGE_HINT, progress_cb: Optional[callable] = None) -> List[Dict[str, Any]]:
    lines = code.splitlines()
    if not lines:
        return []

    segs = segment_code(lines, lang_hint=language_hint, min_seg_lines=MIN_SEG_LINES)
    full_program_text = "\n".join(lines)

    taskgen = TaskGenerator()
    embedder = TaskEmbedder()
    pca = load_segment_pca()

    seg_records: List[Dict[str, Any]] = []

    total = len(segs) if segs else 1
    for seg_id, (s, e) in enumerate(segs):
        seg_text = "\n".join(lines[s : e + 1])
        ctx_text = full_program_text   # Same as new_2ds: use the entire program as context

        try:
            tasks = taskgen.gen_tasks(seg_text, language_hint, ctx_text)
        except Exception as ex:
            # Default fallback
            tasks = ["Summarize the high-level purpose of this code segment."]

        vecs = embedder.encode(tasks)      # [n_tasks, 768]
        vec_raw = vecs.mean(axis=0)        # [768]
        vec128 = project_at_vecs(vec_raw[None, :], pca)[0]  # [AT_DIM]

        rec = {
            "seg_id": seg_id,
            "seg_start": s,
            "seg_end": e,
            "n_lines": e - s + 1,
            "tasks": tasks,
            "vec_raw": vec_raw,
            "vec_atc128": vec128,
        }
        seg_records.append(rec)
        progress_cb(seg_id + 1, total)

    return seg_records


# ============ Broadcast segment vectors to lines: line-level atfeatures ============

def broadcast_segment_atfeatures_to_lines(code: str, segments: List[Dict[str, Any]], dim: int = AT_DIM) -> np.ndarray:
    lines = code.splitlines()
    L = len(lines) if lines else 1

    A = np.zeros((L, dim), dtype=np.float32)
    C = np.zeros((L, 1), dtype=np.int32)

    for rec in segments:
        s = int(rec["seg_start"])
        e = int(rec["seg_end"])
        vec = np.asarray(rec["vec_atc128"], dtype=np.float32)

        s = max(0, s)
        e = min(L - 1, e)
        if e < s:
            continue

        A[s : e + 1] += vec[None, :]
        C[s : e + 1] += 1

    # C: [L, 1]
    # We only care about whether each line is covered by at least one segment
    mask = (C[:, 0] > 0)        # [L] bool
    if np.any(mask):
        counts = C[mask, 0].astype(np.float32)
        counts = counts.reshape(-1, 1)
        A[mask] = A[mask] / counts

    # For lines not covered by any segment (e.g., completely empty lines), keep zero vectors
    return A


def compute_atfeatures_for_code_segment_based(code: str, language_hint: str = LANGUAGE_HINT, progress_cb: Optional[callable] = None) -> np.ndarray:
    segments = compute_segment_atfeatures_for_code(code, language_hint=language_hint, progress_cb=progress_cb)
    if not segments:
        # in case there is no segment, return all-zero features
        n_lines = max(len(code.splitlines()), 1)
        return np.zeros((n_lines, AT_DIM), dtype=np.float32)
    return broadcast_segment_atfeatures_to_lines(code, segments, dim=AT_DIM)



# =========================
# Backend
# =========================

# ==== ccfeature & atfeature backend helpers ====
import os
import re
import tempfile
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# -------------------------
# 1) Load pylint.txt → ERROR_CODES
# -------------------------

PYLINT_TXT_PATH = Path(__file__).parent / "pylint.txt"

# Global cache to avoid reading the file multiple times
_ERROR_CODES: Optional[List[str]] = None


def load_error_codes(pylint_path: Optional[Path] = None) -> List[str]:
    """
    Parse all message codes (e.g., C0114, E0602 ...) from pylint.txt

    Returns:
        error_codes: List[str], fixed order, used for subsequent vector dimensions
    """

    global _ERROR_CODES

    if _ERROR_CODES is not None:
        return _ERROR_CODES

    if pylint_path is None:
        pylint_path = PYLINT_TXT_PATH

    if not pylint_path.exists():
        # If not found, return an empty list, subsequent ccfeature will be all zeros
        _ERROR_CODES = []
        return _ERROR_CODES

    text = pylint_path.read_text(encoding="utf-8", errors="ignore")
    # Same regex as training script: match IDs like "(C0103)"
    codes = re.findall(r"\((\w\d{4})\)", text)
    # Remove duplicates while preserving order
    seen = set()
    ordered_codes = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            ordered_codes.append(c)

    _ERROR_CODES = ordered_codes
    return _ERROR_CODES


# -------------------------
# 2) Run pylint to get output
# -------------------------

def run_pylint_on_code(code: str, pylint_cmd: str = "pylint") -> str:
    """
    Write code to a temporary file, run pylint on it, and return the stdout text.
    If pylint is not installed or the call fails, return an empty string.
    """
    # Create a temporary .py file
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        tmp_path = f.name
        f.write(code)

    try:
        # -r n: don't report summary, only message list
        proc = subprocess.run(
            [pylint_cmd, tmp_path, "-r", "n"],
            capture_output=True,
            text=True,
        )
        stdout = proc.stdout
    except FileNotFoundError:
        stdout = ""
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return stdout or ""


# -------------------------
# 3) Convert pylint output → ccfeature
# -------------------------

def analyze_pylint_output_line(eval_result: str, total_lines: int, error_codes: Optional[List[str]] = None) -> List[List[int]]:
    """
    Parse pylint output into a line-level error code count matrix.

    Args:
        eval_result: pylint stdout text
        total_lines: total number of lines in the code (used to determine matrix rows)
        error_codes: global list of error codes; if None, automatically loaded from pylint.txt

    Returns:
        ccfeature: List[List[int]], shape [total_lines, len(error_codes)]
    """
    if error_codes is None:
        error_codes = load_error_codes()

    if not error_codes:
        # No error codes: return an all-zero matrix
        return [[0 for _ in range(0)] for _ in range(total_lines)]

    # Lines like "3:0: C0116: Missing function or method docstring",
    # we extract (line number, error code)
    pattern = re.compile(r"(\d+):\d+:\s(\w\d{4}):\s")
    matches = pattern.findall(eval_result)

    # First count: per line → Counter(error codes)
    per_line = defaultdict(Counter)
    for line_str, code in matches:
        try:
            line_no = int(line_str)
        except ValueError:
            continue
        # pylint line numbers start at 1, we also treat code lines as 1-based
        if 1 <= line_no <= total_lines:
            per_line[line_no][code] += 1

    # Then map the Counter to a fixed-dimension vector
    ccfeature: List[List[int]] = []
    for line_idx in range(1, total_lines + 1):
        counter = per_line.get(line_idx, {})
        vec = [counter.get(code, 0) for code in error_codes]
        ccfeature.append(vec)

    return ccfeature


def compute_ccfeatures_for_code(code: str, pylint_cmd: str = "pylint") -> Tuple[np.ndarray, List[str]]:
    lines = code.splitlines()
    n_lines = len(lines) if lines else 1

    error_codes = load_error_codes()
    eval_text = run_pylint_on_code(code, pylint_cmd=pylint_cmd)

    cc_list = analyze_pylint_output_line(
        eval_result=eval_text,
        total_lines=n_lines,
        error_codes=error_codes,
    )

    if not cc_list:
        # Extreme case: no lines / no error codes, construct an empty matrix of shape [n_lines, 0]
        cc_mat = np.zeros((n_lines, 0), dtype=np.float32)
    else:
        cc_mat = np.asarray(cc_list, dtype=np.float32)

    return cc_mat, error_codes


# -------------------------
# 4) atfeature
# -------------------------

def compute_atfeatures_for_code(code: str, dim: int = 128, progress_cb: Optional[callable] = None) -> np.ndarray:
    return compute_atfeatures_for_code_segment_based(code, language_hint="python", progress_cb=progress_cb)


# -------------------------
# 5) features
# -------------------------

def compute_ll_features_for_code(code: str):
    ppl = app_extract_single_ppl.retrieve_ppl_features(code)

    model_order = ["incoder", "polycoder", "codellama", "starcoder2"]

    ll_tokens_list = []
    begin_idx_list = []

    for name in model_order:
        info = ppl[name]
        ll_tokens = info.get("ll_tokens", [])
        begin_idx = info.get("begin_word_idx", 0)

        # Fallback: in case any entry is None
        if ll_tokens is None:
            ll_tokens = []

        ll_tokens_list.append(ll_tokens)
        begin_idx_list.append(begin_idx)

    # If there is nothing, return an empty array directly
    if not ll_tokens_list or len(ll_tokens_list[0]) == 0:
        return np.zeros((0, len(model_order)), dtype=np.float32)

    # 2. Align according to the logic in DataManager.initialize_dataset
    begin_idx_arr = np.array(begin_idx_list, dtype=int)
    max_begin_idx = int(begin_idx_arr.max(initial=0))

    trimmed = []
    lengths = []
    for seq, b in zip(ll_tokens_list, begin_idx_arr):
        # Reserve for future cases where begin_idx is not 0
        offset = max(max_begin_idx - b, 0)
        sub = seq[offset:]
        trimmed.append(sub)
        lengths.append(len(sub))

    min_len = min(lengths) if lengths else 0
    if min_len <= 0:
        return np.zeros((0, len(model_order)), dtype=np.float32)

    # 3. Trim to the same length, then transpose [4, T] → [T, 4]
    aligned = [seq[:min_len] for seq in trimmed]
    arr = np.asarray(aligned, dtype=np.float32)  # [4, T]
    feats = arr.T                                 # [T, 4]

    return feats


@st.cache_resource
def load_detector_model(checkpoint_path: str):
    # Load a trained MultiModalConcatLineFocalBMESBinaryClassifier.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        st.error(f"找不到 checkpoint: {checkpoint_path}")
        return None

    with st.spinner("Loading checkpoint shards..."):
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)

    st.success("Finished loading checkpoint shards.")


    if isinstance(saved, nn.Module):
        model = saved
    else:
        # This should not happen, but just in case
        from model_4 import MultiModalConcatLineFocalBMESBinaryClassifier

        def build_id2label():
            labels = ["human", "AI"]
            prefix = ["B-", "M-", "E-", "S-"]
            id2label = {}
            idx = 0
            for name in labels:
                for pre in prefix:
                    id2label[idx] = pre + name
                    idx += 1
            return id2label

        id2label = build_id2label()
        seq_len = 1024
        model = MultiModalConcatLineFocalBMESBinaryClassifier(
            id2labels=id2label,
            seq_len=seq_len,
            alpha=0.8,
        )
        model.load_state_dict(saved)

    model.to(device)
    model.eval()
    return model


def run_detection_real_backend(code: str, threshold: float, model) -> pd.DataFrame:
    # If the model failed to load, fall back to demo logic
    if model is None:
        return run_detection_demo(code, threshold)

    lines = code.splitlines()
    n_lines = len(lines) if lines else 1

    device = next(model.parameters()).device
    seq_len = getattr(model, "seq_len", 1024)

    # --- Ordinary ll_token features: [T, 4] ---
    with st.spinner("Computing ll_features..."):

        ll_mat = compute_ll_features_for_code(code)          # [T, 4]
        if ll_mat.ndim != 2 or ll_mat.shape[1] != 4:
            # If something went wrong, treat as none
            T_ll = 0
            feat_channels = 4
            ll_mat = np.zeros((0, feat_channels), dtype=np.float32)
        else:
            T_ll = ll_mat.shape[0]
            feat_channels = ll_mat.shape[1]
    
    st.success("Finished computing ll_features.")

    # --- 1) ccfeature: [n_lines, C_cc] ---
    cc_mat, _ = compute_ccfeatures_for_code(code)
    if cc_mat.ndim != 2:
        # Extreme fallback
        cc_mat = np.zeros((n_lines, 0), dtype=np.float32)
    cc_dim = cc_mat.shape[1]

    # --- 2) atfeatures: [n_lines, 128] ---
    progress_bar = st.progress(0, text="Generating AT features (segments-level)")
    progress_text = st.empty()

    def atf_progress(done: int, total: int):
        frac = done / max(total, 1)
        progress_bar.progress(
            frac,
            text=f"Generating AT features: segment {done}/{total}",
        )
        # Optional
        progress_text.write(f"ATF segments: {done}/{total}")

    # Broadcast segment-level ATF to each line
    at_mat = compute_atfeatures_for_code(code, progress_cb=atf_progress)

    # After generation, remove the components
    progress_bar.empty()
    progress_text.empty()

    if at_mat.ndim != 2:
        at_mat = np.zeros((n_lines, AT_DIM), dtype=np.float32)
    at_dim = at_mat.shape[1]

    # --- 3) Align to seq_len (model's internal CRF requires length and mask to be consistent) ---
    if T_ll > 0:
        valid_L = min(n_lines, T_ll, seq_len)
    else:
        valid_L = min(n_lines, seq_len)

    x_full = torch.zeros((1, seq_len, feat_channels), dtype=torch.float32, device=device)
    if T_ll > 0:
        x_full[0, :valid_L, :] = torch.from_numpy(ll_mat[:valid_L, :]).to(device)


    # ccfeature pad to [seq_len, cc_dim]
    if cc_dim == 0:
        cc_full = np.zeros((seq_len, 0), dtype=np.float32)
    else:
        cc_full = np.zeros((seq_len, cc_dim), dtype=np.float32)
        cc_full[:valid_L, :] = cc_mat[:valid_L, :]
    cc_tensor = torch.from_numpy(cc_full).unsqueeze(0).to(device)  # [1, seq_len, C_cc]

    # atfeatures pad to [seq_len, at_dim]
    at_full = np.zeros((seq_len, at_dim), dtype=np.float32)
    at_full[:valid_L, :] = at_mat[:valid_L, :]
    at_tensor = torch.from_numpy(at_full).unsqueeze(0).to(device)  # [1, seq_len, 128]

    # --- 4) labels: only used for mask ---
    # Training convention: labels=-1 is padding; 0~7 are real BMES labels
    # During inference, we just use 0 to mark real lines (only for mask=True)
    labels = torch.full((1, seq_len), -1, dtype=torch.long, device=device)
    labels[0, :valid_L] = 0  # 0 = B-human, the specific value doesn't matter, just > -1


    # --- 5) Run model ---
    model.eval()
    with torch.no_grad():
        out = model(
            x_full,  # ll_token_feature stub: [1, seq_len, 4]
            labels=labels,  # for mask
            ccfeature=cc_tensor,
            atfeatures=at_tensor,
        )
        logits = out["logits"]  # [1, seq_len, 8]

    # Only take the first valid_L lines (the rest are padding)
    logits = logits[0, :valid_L, :]  # [valid_L, 8]

    # --- 6) Calculate "AI probability" from 8-class BMES labels ---
    # Training convention: 0-3 = human(BMES), 4-7 = AI(BMES)
    probs = torch.softmax(logits, dim=-1)  # [valid_L, 8]
    ai_probs = probs[:, 4:].sum(dim=-1).cpu().numpy()  # [valid_L]

    is_ai = ai_probs >= threshold

    # --- 7) Put back into DataFrame for frontend ---
    rows = []
    for i in range(valid_L):
        content = lines[i] if i < len(lines) else ""
        rows.append(
            {
                "line_no": i + 1,
                "content": content,
                "ai_prob": float(ai_probs[i]),
                "is_ai": bool(is_ai[i]),
            }
        )

    df = pd.DataFrame(rows)
    return df


# for UI demo purposes only
def run_detection_demo(code: str, threshold: float) -> pd.DataFrame:
    lines = code.splitlines()
    data = []
    n = max(len(lines), 1)

    rng = np.random.default_rng(42)

    for i, raw in enumerate(lines, start=1):
        stripped = raw.strip()

        # Fake logic for demo:
        base = i / n * 0.6
        if len(stripped) > 60:
            base += 0.15
        if any(kw in stripped for kw in ["for ", "while ", "def ", "return ", "class "]):
            base += 0.15

        noise = rng.normal(loc=0.0, scale=0.05)
        ai_prob = float(np.clip(base + noise, 0.01, 0.99))
        is_ai = ai_prob >= threshold

        data.append(
            {
                "line_no": i,
                "content": raw,
                "ai_prob": ai_prob,
                "is_ai": is_ai,
            }
        )

    return pd.DataFrame(data)


def run_detection(code: str, threshold: float, model: Optional[object]) -> pd.DataFrame:
    if st.session_state.get("demo_mode", DEMO_MODE_DEFAULT):
        return run_detection_demo(code, threshold)
    return run_detection_real_backend(code, threshold, model)


# =========================
# Frontend rendering helper functions
# =========================

def build_highlighted_code_html(df: pd.DataFrame) -> str:
    """
    Render each row in the DataFrame as a colored <pre> block.
    """
    html_lines = [
        '<div style="font-family:Menlo,Consolas,monospace; font-size:13px;">'
    ]

    for _, row in df.iterrows():
        line_no = row["line_no"]
        text = row["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        is_ai = bool(row["is_ai"])
        ai_prob = float(row["ai_prob"])

        bg = AI_BG_COLOR if is_ai else HUMAN_BG_COLOR
        fg = AI_TEXT_COLOR if is_ai else HUMAN_TEXT_COLOR
        label = "M" if is_ai else "H"

        html_lines.append(
            f'<div style="background-color:{bg}; padding:2px 6px; margin:1px 0;">'
                f'<span style="color:#888; width:3em; display:inline-block; text-align:right;">'
                    f'{line_no:>3}'
                '</span>'
                f'<span style="color:#aaa; margin:0 4px;">|</span>'
                f'<span style="color:{fg}; font-weight:bold; margin-right:6px;">'
                    f'[{label} {ai_prob:.2f}]'
                f'</span>'
                f'<span style="white-space:pre;color:{fg}">{text}</span>'
            f'</div>'
        )

    html_lines.append("</div>")
    return "\n".join(html_lines)


# =========================
# Streamlit UI main function
# =========================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Title area
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    # Initialize session_state
    if "demo_mode" not in st.session_state:
        st.session_state["demo_mode"] = DEMO_MODE_DEFAULT

    # --- Sidebar: Settings ---
    with st.sidebar:
        st.header("Settings")

        threshold = st.slider(
            "AI probability threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Lines with a probability greater than or equal to this threshold will be classified as AI-generated.",
        )

        max_lines_display = st.number_input( 
            "Max lines to display", 
            min_value=50, 
            max_value=1000, 
            value=MAX_CODE_LINES, 
            step=50, 
            help="Only affects frontend display, does not affect backend inference.", 
        )

        st.markdown("---")
        st.markdown("**Model checkpoint**")
        st.markdown(f"`{MODEL_CHECKPOINT_PATH}`")

    # --- Main area: input code ---
    with st.expander("How to use", expanded=False):
        st.markdown(
            textwrap.dedent(
                """
                1. Paste your code into the text area. (Python code is recommended.)
                2. Click **Start Detection**.
                3. The results below will show:
                   - A table with AI probability and classification for each line;
                   - A highlighted code view with background colors.
                """
            )
        )

    col_input, col_buttons = st.columns([4, 1])

    with col_input:
        code_text = st.text_area(
            "Paste your code here:",
            value='',
            height=320,
            placeholder="Paste your code here...",
        )

    # Let text_area and session_state sync
    if "code_text" not in st.session_state:
        st.session_state["code_text"] = code_text
    else:
        # If there is a value in session_state, use it to overwrite the current text_area content
        # Note: This does not update the text_area UI in real-time, only affects the text used for subsequent detection
        if st.session_state["code_text"] != code_text:
            st.session_state["code_text"] = code_text

    # --- Detection button ---
    run_clicked = st.button("Start Detection", type="primary")
    st.markdown("---")

    df_result = None

    if run_clicked:
        code = st.session_state.get("code_text", "") or ""
        if not code.strip():
            st.warning("Please input some code to analyze.")
        else:
            # Load model (returns None in demo mode)
            model = load_detector_model(MODEL_CHECKPOINT_PATH)

            # Call unified detection interface
            df_result = run_detection(code, threshold=threshold, model=model)

            # Save for page refresh persistence
            st.session_state["last_result"] = df_result

    # If the button was not just clicked but there is a previous result, display it
    if df_result is None:
        df_result = st.session_state.get("last_result", None)

    # --- Results display ---
    if df_result is not None and not df_result.empty:
        st.subheader("Line-wise predictions")

        # Only show the first max_lines_display lines 
        df_show = df_result.copy()
        if len(df_show) > max_lines_display:
            df_show = df_show.iloc[:max_lines_display, :]

        # Table
        df_table = df_show[["line_no", "ai_prob", "is_ai", "content"]].copy()
        df_table["ai_prob"] = df_table["ai_prob"].map(lambda x: f"{x:.2f}")
        df_table["is_ai"] = df_table["is_ai"].map(lambda x: "AI" if x else "Human")

        st.dataframe(
            df_table,
            use_container_width=True,
            hide_index=True,
        )

        # Highlighted code
        st.subheader("Highlighted code")
        html = build_highlighted_code_html(df_show)
        st.markdown(html, unsafe_allow_html=True)

    else:
        st.info("Detection results will be displayed here. Please input code above and click the button.")


if __name__ == "__main__":
    main()
