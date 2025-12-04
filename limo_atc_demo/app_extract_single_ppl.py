import msgpack
import httpx


def retrieve_ppl_features(code: str):
    API_ENDPOINTS = {
        "incoder":   "http://0.0.0.0:6006/inference",
        "polycoder": "http://0.0.0.0:6007/inference",
        "codellama": "http://0.0.0.0:6008/inference",
        "starcoder2": "http://0.0.0.0:6009/inference",
    }

    mask_start, mask_end = 0, 0

    features = {}
    base_len = None

    with httpx.Client(timeout=None) as client:
        for name, url in API_ENDPOINTS.items():
            payload = {
                "text": code,
                "do_generate": False,
                "mask_start": mask_start,
                "mask_end": mask_end,
            }
            
            try:
                resp = client.post(url, data=msgpack.packb(payload))
                resp.raise_for_status()
            except httpx.RequestError as e:
                # 比如：Connection refused: [Errno 111]
                print(f"[PPL WARN] backend {name} ({url}) unreachable: {e}")
                continue

            loss, begin_word_idx, ll_tokens = msgpack.unpackb(resp.content)

            ll_tokens = [float(x) for x in ll_tokens]
            if base_len is None:
                base_len = len(ll_tokens)

            # result from api
            features[name] = {
                "loss": float(loss),
                "begin_word_idx": int(begin_word_idx),
                "ll_tokens": ll_tokens,
            }

            
        # 如果一个都没连上，说明后端全挂了，直接抛异常让上层报错
        if not features:
            raise RuntimeError("None of the PPL backends are reachable. "
                            "Check backend_api.sh / model ports.")

        # 防御性：如果第一个成功模型给不出长度，就当 0 处理
        if base_len is None:
            base_len = 0

        # 给没连上的模型补 0，保证 4 个 key 都存在
            for name in API_ENDPOINTS.keys():
                if name not in features:
                    print(f"[PPL INFO] Filling zeros for missing model '{name}'")
                    features[name] = {
                        "loss": 0.0,
                        "begin_word_idx": 0,
                        "ll_tokens": [0.0] * base_len,
                    }
        
        
    return features


def compute_program_ppl_record(code: str):
    raw = retrieve_ppl_features(code)

    losses = []
    begin_idx_list = []
    ll_tokens_list = []

    for name in ["incoder", "polycoder", "codellama", "starcoder2"]:
        model_feature = raw[name]

        losses.append(model_feature["loss"])
        begin_idx_list.append(model_feature["begin_word_idx"])
        ll_tokens_list.extend(model_feature["ll_tokens"])

    res = {
        "losses": losses,
        "begin_idx_list": begin_idx_list,
        "ll_tokens_list": ll_tokens_list,
    }

    return res