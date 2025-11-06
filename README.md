# ***LiMO with Approximated Task (AT)***

---

# ***Project Structure***
> `limo_atc` is the project directory for LiMO with `atfeatures`
> 
> `limo_origin` is the project directory for original LiMO
> 
> `res` stores the output of `train_ui.ipynb` in each situation, including:
> * `limo_origin.txt`: Output of the original LiMO
> * `limo_atf128.txt`: Output of **_LiMO-ATC_** with 128-dimensions `atfeatures`
> * `limo_atf256.txt`: Output of **_LiMO-ATC_** with 256-dimensions `atfeatures`
> * `new_2ds.ipynb` is used to generate approximated task and getting the embedded values of **_CodeBERT_**

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

***Overall separability (AUC/AUPRC):** `atf128` ≥ `atf256` > `origin`.  

**Bottom line:** both `atf128` and `atf256` outperform `origin`; within the new models, `atf128` shows a slight edge on AUC/AUPRC and F1, while `atf256` edges the raw binF1@0.5 peak by a hair.