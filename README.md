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
> * `limo_cossim.txt`: Output of **_LiMO-ATC_** with weighted 128-dimensions `atfeatures` (weighted by cosine similarities)
> * `limo_atfdoc.txt`: Output of document-level `atfeatures`
>
> `do_test_res` stores the output of `train_ui.ipynb` in test dataset, including:
> * `limo_origin.txt`: Output of the original LiMO
> * `limo_atf128.txt`: Output of **_LiMO-ATC_** with 128-dimensions `atfeatures`
> * `limo_atf256.txt`: Output of **_LiMO-ATC_** with 256-dimensions `atfeatures`
> * `limo_cossim.txt`: Output of **_LiMO-ATC_** with weighted 128-dimensions `atfeatures` (weighted by cosine similarities)
> * `limo_atfdoc.txt`: Output of document-level `atfeatures`
> 
> `new_2ds.ipynb` is used to generate approximated task and getting the embedded values of **_CodeBERT_**

--- 

## ***Test Results***

> * **Updated _2025/11/17_**
>   * Experiments were done with new parameters
>     * `Seed`:   **42**
>     * `lr`:     **5e-5**
>     * `Epoch`:  **60**
>     * `Alpha`:  **0.8**
>   * After updating parameters, some interesting results are noticed
>     * _Italic_ means the number is higher than `limo_origin`
>     * **_Bolded-Italic_** means the number is the highest among the results

### Given labels (as-is)

| Model             | Acc          | Macro F1     | Bin F1       | ROC-AUC     | AUPRC       |
|-------------------|--------------|--------------|--------------|-------------|-------------|
| limo_origin       | 81.616       | 79.064       | 71.756       | 0.869       | 0.781       |
| ***limo_atf128*** | ***82.777*** | ***80.191*** | ***73.033*** | ***0.884*** | ***0.811*** |
| limo_atf256       | _82.047_     | 79.033       | 71.084       | _0.881_     | _0.805_     |
| limo_atfdoc       | 81.294       | 78.461       | 70.649       | _0.876_     | _0.803_     |
| limo_cos_sim      | _82.134_     | _79.124_     | 71.197       | _0.882_     | _0.809_     |

---

## ***Test Results*** - Threshold sweeps on scores

### Threshold = 0.5

| Model             | Acc        | Macro F1   | Bin F1     |
|-------------------|------------|------------|------------|
| limo_origin       | 80.1       | 77.2       | 69.2       |
| ***limo_atf128*** | ***81.0*** | ***78.3*** | ***70.5*** |
| limo_atf256       | _80.8_     | _77.8_     | _69.7_     |
| limo_atfdoc       | 80.1       | _77.3_     | 69.2       |
| limo_cos_sim      | _80.8_     | _77.9_     | _69.8_     |

---

### Threshold = YoudenJ (per-model)

| Model             | Acc        | Macro F1   | Bin F1     |
|-------------------|------------|------------|------------|
| limo_origin       | 78.4       | 77.2       | 71.9       |
| ***limo_atf128*** | _79.0_     | _77.9_     | _73.0_     |
| limo_atf256       | ***79.6*** | ***78.4*** | **_73.2_** |
| limo_atfdoc       | _79.1_     | _77.8_     | _72.4_     |
| limo_cos_sim      | _79.0_     | _77.8_     | _72.7_     |

---

### Threshold = bestPosF1 (per-model)

| Model             | Acc        | Macro F1   | Bin F1     |
|-------------------|------------|------------|------------|
| limo_origin       | 78.4       | 77.2       | 71.9       |
| ***limo_atf128*** | ***79.9*** | ***78.5*** | _73.0_     |
| limo_atf256       | _79.6_     | _78.4_     | **_73.2_** |
| limo_atfdoc       | _79.2_     | _77.9_     | _72.4_     |
| limo_cos_sim      | _79.7_     | _78.3_     | 72.8       |

---
## Analysis
> * Overall, `limo_atf128` is having the best performance
>   * `limo_atf256` and `limo_atfdoc` showed a better performance compare to old version parameters. 
> * _**AUPRC**_ and _**ROC-AUC**_ improved most significantly, all atf-involved models has a better result in these two metrics.  
> * Conclusion below is received after calculating the cosine similarity between local atf (line-level) and global atf (doc-level)
>   * Image: the relationship between cosine similarity and lines
>     ![Alt](/assets/cosine_sim_dis.png)
>     * ("High-info lines only" means: lines like 'else', '[', '(', '{'... are ignored)
>   * Average cosine similarities:
>     * ```
>       Filtered Human  mean: 0.18534581653262655 std: 0.3078455960799607
>       Filtered Machine mean: 0.14431030143421408 std: 0.2951902140976671
>       ```
>   * These results showed that there is a difference in cosine similarity between local-atf and global-atf among human and machine generated codes
>     * The difference in average cos_sim
>     * Human written lines are likely to be closer to the global intent as shown in the image
>       * (The blue part shifts to the right, and the orange is closer to the left-hand-side)
>   * Even though this result suggests Approximated Task Feature (a.t.f) as an effective feature in MGC detection, but the difference between Human-ATF and Machine-ATF are not decisive enough to detect MGC independently
>     * However, as a supportive feature, it does play a role of improving the performance of LiMO
>       * a.t.f can be considered as a weak signal
>     * The prompting strategy can be revised to fit this feature
>       * Especially for global intent (doc-level)
