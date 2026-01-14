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

| Metric Category       | Metric    | Value |
| --------------------- | --------- | ----- |
| **Overall**     | Accuracy  | 71.60 |
|                       | Hit       | 88.21 |
| **Answer**      | F1        | 44.84 |
|                       | Precision | 44.02 |
|                       | Recall    | 69.07 |
| **Path**        | F1        | 41.15 |
|                       | Precision | 41.08 |
|                       | Recall    | 61.39 |
| **Path Answer** | F1        | 47.90 |
|                       | Precision | 47.11 |
|                       | Recall    | 71.66 |

Results of GRACE (set $k_r=k_e=5$):

| Metric Category       | Metric    | Value |
| --------------------- | --------- | ----- |
| **Overall**     | Accuracy  | 71.99 |
|                       | Hit       | 89.07 |
| **Answer**      | F1        | 49.40 |
|                       | Precision | 50.99 |
|                       | Recall    | 69.22 |
| **Path**        | F1        | 45.80 |
|                       | Precision | 48.14 |
|                       | Recall    | 62.21 |
| **Path Answer** | F1        | 52.61 |
|                       | Precision | 54.16 |
|                       | Recall    | 72.05 |

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
