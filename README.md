# ***LiMO with Approximated Task (AT)***

---

# ***Project Structure***
> `limo_atc` is the project directory for LiMO with `atfeatures`
> 
> `limo_origin` is the project directory for original LiMO
> 
> `res` stores the output of `train_ui.ipynb` during training:
> * `limo_origin.txt`: Output of the original LiMO
> * `limo_atf128.txt`: Output of **_LiMO-ATC_** with 128-dimensions `atfeatures`
> * `limo_atf256.txt`: Output of **_LiMO-ATC_** with 256-dimensions `atfeatures`
> * `limo_atfdoc.txt`: Output of document-level `atfeatures`
>
> `do_test_res` stores the output of `train_ui.ipynb` in test dataset, including:
> * `limo_origin.txt`: Output of the original LiMO
> * `limo_atf128.txt`: Output of **_LiMO-ATC_** with 128-dimensions `atfeatures`
> * `limo_atf256.txt`: Output of **_LiMO-ATC_** with 256-dimensions `atfeatures`
> * `limo_atfdoc.txt`: Output of document-level `atfeatures`
> 
> `new_2ds.ipynb` is used to generate approximated task and getting the embedded values of **_CodeBERT_**

--- 

# ***Results***
## Best Scores Across Epochs
Peak values observed over all parsed epochs

|        | best acc @thr=0.5 | best macroF1 @thr=0.5 | best binF1 @thr=0.5 |   best ROC_AUC   |    best AUPRC    | best BinF1 @thr=bestPosF1 | best MacroF1 @thr=bestPosF1 |
|:------:|:-----------------:|:---------------------:|:-------------------:|:----------------:|:----------------:|:-------------------------:|:---------------------------:|
| origin |    79.6 (ep19)    |      74.3 (ep14)      |     63.2 (ep9)      |   0.869 (ep18)   |   0.772 (ep19)   |        69.6 (ep17)        |         76.7 (ep13)         |
| atf128 |  **81.5 (ep14)**  |    **77.4 (ep14)**    |     67.8 (ep14)     | **0.882 (ep19)** | **0.784 (ep17)** |      **71.8 (ep19)**      |       **78.7 (ep19)**       |
| atf256 |    81.1 (ep13)    |    **77.4 (ep13)**    |   **68.1 (ep13)**   |   0.877 (ep19)   |   0.774 (ep19)   |        70.5 (ep17)        |         76.6 (ep20)         |

---

## Last-Epoch (Epoch 20)
For reference, the final-epoch metrics in these runs.

|        | epoch | acc @thr=0.5 | macroF1 @thr=0.5 | binF1 @thr=0.5 |  ROC_AUC  |   AUPRC   | BinF1 thr=bestPosF1 | MacroF1 thr=bestPosF1 |
|:------:|:-----:|:------------:|:----------------:|:--------------:|:---------:|:---------:|:-------------------:|:---------------------:|
| origin |  20   |    77.471    |      71.051      |     57.418     |   0.848   |   0.765   |        69.1         |         74.9          |
| atf128 |  20   |  **77.863**  |    **71.230**    |     57.415     | **0.861** | **0.776** |      **70.2**       |         75.4          |
| atf256 |  20   |    77.039    |      70.032      |     55.540     |   0.859   |   0.774   |      **70.2**       |       **76.1**        |

---

## ***Test Results***

### Given labels (as-is)

| Model             | Acc          | Macro F1     | F1(AI, pos)  | ROC-AUC     | AUPRC       |
|-------------------|--------------|--------------|--------------|-------------|-------------|
| limo_origin       | 77.471       | 71.051       | 57.418       | 0.848       | 0.765       |
| ***limo_atf128*** | ***79.637*** | ***74.093*** | ***62.109*** | ***0.884*** | ***0.811*** |
| limo_atf256       | 77.039       | 70.032       | 55.540       | 0.859       | 0.774       |
| limo_atfdoc       | 77.486       | 71.148       | 57.624       | 0.853       | 0.767       |

---

## ***Test Results*** - Threshold sweeps on scores

### Threshold = 0.5

| Model             | Acc        | Macro F1   | F1(AI, pos) |
|-------------------|------------|------------|-------------|
| limo_origin       | 76.6       | 69.8       | 55.6        |
| ***limo_atf128*** | ***77.9*** | ***71.6*** | ***58.3***  |
| limo_atf256       | 76.2       | 69.2       | 54.4        |
| limo_atfdoc       | 76.5       | 69.7       | 55.4        |

---

### Threshold = YoudenJ (per-model)

| Model             | Acc        | Macro F1   | F1(AI, pos) |
|-------------------|------------|------------|-------------|
| limo_origin       | 76.3       | 74.9       | 69.1        |
| ***limo_atf128*** | ***79.8*** | ***78.6*** | ***73.5***  |
| limo_atf256       | 77.5       | 76.1       | 70.2        |
| limo_atfdoc       | 76.7       | 75.3       | 69.5        |

---

### Threshold = bestPosF1 (per-model)

| Model             | Acc        | Macro F1   | F1(AI, pos) |
|-------------------|------------|------------|-------------|
| limo_origin       | 76.3       | 74.9       | 69.1        |
| ***limo_atf128*** | ***80.2*** | ***78.9*** | ***73.5***  |
| limo_atf256       | 77.5       | 76.1       | 70.2        |
| limo_atfdoc       | 76.7       | 75.3       | 69.5        |


**Overall performance:** `atf128` > `atf256` > `atfdoc` ≥ `origin`.  
