# GRACE

## Usage:

Running on RoG-webqsp:
`GPU_ID="0,1,2" bash agc_reasoning2.sh`

Results of GRACE:

| Metric Category       | Metric    | Value |
| --------------------- | --------- | ----- |
| **Overall**     | Accuracy  | 70.37 |
|                       | Hit       | 86.67 |
| **Answer**      | F1        | 44.31 |
|                       | Precision | 43.67 |
|                       | Recall    | 67.82 |
| **Path**        | F1        | 40.76 |
|                       | Precision | 40.93 |
|                       | Recall    | 60.55 |
| **Path Answer** | F1        | 47.28 |
|                       | Precision | 46.76 |
|                       | Recall    | 70.43 |

Results of GRACE (simplified all prompts):

| Metric Category       | Metric    | Value |
| --------------------- | --------- | ----- |
| **Overall**     | Accuracy  | 71.60 |
|                       | Hit       | 88.21 |
| **Answer**      | F1        | 44.82 |
|                       | Precision | 43.99 |
|                       | Recall    | 69.07 |
| **Path**        | F1        | 41.15 |
|                       | Precision | 41.08 |
|                       | Recall    | 61.39 |
| **Path Answer** | F1        | 47.90 |
|                       | Precision | 47.11 |
|                       | Recall    | 71.66 |

Results of GRACE (more simplier)
Accuracy: 71.59722772869637 Hit: 88.20638820638821 F1: 44.8403184933701 Precision: 44.0245358995359 Recall: 69.06765536382869 Path F1: 41.147272112742364 Path Precision: 41.079959829959826 Path Recall: 61.39081793963209 Path Answer F1: 47.90260624893888 Path Answer Precision: 47.105758355758354 Path Answer Recall: 71.66340808333825


Results of GCR:

| Metric Category       | Metric    | Value |
| --------------------- | --------- | ----- |
| **Overall**     | Accuracy  | 72.80 |
|                       | Hit       | 83.65 |
| **Answer**      | F1        | 45.17 |
|                       | Precision | 42.30 |
|                       | Recall    | 72.43 |
| **Path**        | F1        | 51.15 |
|                       | Precision | 52.59 |
|                       | Recall    | 69.16 |
| **Path Answer** | F1        | 55.39 |
|                       | Precision | 55.91 |
|                       | Recall    | 72.51 |
