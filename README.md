# A Reproducibility Study of Bundle Editing and Bundle Recommendation

This repository contains the code and datasets for the paper "**A Reproducibility Study of Bundle Editing and Bundle Recommendation**" at SIGIR 2026.

## Dataset
Six datasets (Youshu, NetEase, iFashion, clothing, electronic, food) used in the study are available in the folder  `./dataset`. The datasets are preprocessed and ready to use.

For each dataset, the following important files are considered:

-   `user_item.txt`: user-item interactions are represented in the form of 'user_id item_id'
-   `bundle_item.txt`: bundle-item affiliations are represented in the form of 'bundle_id item_id'
-   `user_bundle_train.txt`: user-bundle interactions for the train set, are represented in the form of 'user_id bundle_id'
-   `user_bundle_tune.txt`: user-bundle interactions for the validation set, are represented in the form of 'user_id bundle_id'
-   `user_bundle_test.txt`: user-bundle interactions for the test set, are represented in the form of 'user_id bundle_id'

It is noteworthy that all user-item interactions and bundle-item affiliations are used as the input for the BR models. 
## Bundle Editing

### 1. Bundle-level Editing

All experiments for **RQ1** are located in `.\bundle_edit\bundle-level`. Run the following commands to reproduce:

**Non-LLM**
```bash
cd non_llm
python run_BBPR.py --dataset clothing
```
**LLM**
```bash
cd llm
python run_zeroshot.py --dataset clothing
```

### 2. Item-level Editing

All experiments for **RQ2** are located in `.\bundle_edit\item-level`.
**Step 1 — Construct the candidate**
```bash
python sample_candidate.py -d clothing
```
**Step 2 — Run experiments**

 - **Static Editing:**

Non-LLM:
```bash
cd non_llm/bundle_add/non_llm
python run_all_add.py -d clothing -m bprmf
```
LLM:
```yaml
api_key: YOUR_API_KEY_HERE
```
```bash
cd llm
python cot.py -m add -d clothing
```
 - **Dynamic Editing:**
```bash
cd llm
python zeroshot.py -d clothing -c ../props/zeroshot_dy.yaml
```
## Bundle Recommendation
All experiments for **RQ3** are located in `.\bundle_rec`.

**Step 1 — Dataset splitting** (cold / warm / mixed start scenarios)

```bash
python cold_start_split.py
python all_bundle_split.py
python warm_start_split.py
```
**Step 2 — Model training**

```bash
cd CrossCBR
python train.py -d Youshu
```

## Unified Pipeline: Item-Level Editing + Bundle Recommendation
All experiments for **RQ4** are located in `.\bundle_edit_rec`.

**Step 1 — Preprocess with BPRMF**

```bash
python bprmf_mask.py -d Youshu
```
**Step 2 — Run recommendation model on processed data**（以 CrossCBR 为例）
```bash
cd CrossCBR
python train.py -d Youshu -rho 0     # -p: 
```
