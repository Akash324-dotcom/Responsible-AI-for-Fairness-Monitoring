# Responsible AI: Mortgage Denial Model and Fairness Audit

An audit of a mortgage-denial prediction model trained on **850,278 California home-purchase applications** (public HMDA data, 2022-2024). It asks three questions: how good is the model, who does it get wrong, and what does it cost to fix that? It was a model-risk exercise, not a leaderboard one, so the result is a deployment recommendation backed by numbers, with the limits stated.

Course project for CSCI 4146 / 6409 Process of Data Science (Dalhousie University). It is written up as a model-risk engagement: a model card, a fairness-definition decision log, a mitigation trade-off log, and a production-monitoring plan are in the notebook.

## Main findings

| Question | Result |
|---|---|
| Model quality | A gradient-boosted model (`HistGradientBoostingClassifier`) reached **0.807 ROC-AUC** (PR-AUC 0.588) against **0.714** (PR-AUC 0.444) for a logistic-regression baseline on the validation split. |
| Operating point | A threshold of about 0.10 flags the top ~20% of applications: **precision 0.359, recall 0.638** (118,194 validation rows). |
| Leakage | `interest_rate`, `rate_spread` and `total_loan_costs` are missing for **100%** of denied applications, so they reveal the outcome. They were removed before training. |
| Who it gets wrong | False-positive rate (a creditworthy applicant flagged) is **21.5% for Black applicants against 15.2% for White applicants**. Group sizes are shown next to every metric. |
| Is it a proxy problem? | A model that sees no race column can still predict White vs non-White with **0.69 AUC**. The strongest proxy is `tract_minority_share` (permutation importance 0.102, nearly double the next feature). |
| Fixing it | Removing that proxy left accuracy unchanged (0.8066 to 0.8064). Up-weighting non-denied Black and American Indian / Alaska Native applicants during training (weight 10) brought their false-positive rates to **15.5%** and **18.0%**, at a cost of 0.005 AUC. |
| What it costs | Recall for Black applicants falls from 67.2% to 60.3% under the weight-10 setting. A group-specific threshold closes the gap too, but it uses race at decision time, so it was **rejected for deployment**. |

All numbers come from the committed notebook outputs and from `project 4/artifacts/`.

## What is in the repo

```
project 4/
  project_workbook.ipynb   the full audit: M0 risk assessment, M1 models and thresholds,
                           M2 fairness audit, M3 mitigation, M4 monitoring, M5 model card
  tutorial_warmup.ipynb    short warm-up on the UCI credit-card default dataset
  data/                    the HMDA slice, the warm-up dataset, and a data README
  scripts/make_hmda_slice.py   rebuilds the HMDA slice from the CFPB/FFIEC API
  artifacts/               every table and figure the notebook exports (CSV, PNG)
```

## How to run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook "project 4/project_workbook.ipynb"
```

The notebook reads the committed slice at `project 4/data/hmda_project4_slice.csv.gz`, so nothing needs downloading. To rebuild the slice from the public HMDA API:

```bash
cd "project 4"
python scripts/make_hmda_slice.py --state CA --years 2022,2023,2024 --max-rows 900000 --out-dir data
```

Training and splits use a fixed random seed (`4146`), so results reproduce.

## Method notes

- **Target.** `target_denied` is a historical lender decision, not a measure of creditworthiness. The audit treats the labels as biased evidence, not as ground truth.
- **Small groups.** Metrics are flagged `stable_n100`: groups with fewer than 100 applicants are shown but not interpreted.
- **Fairness definitions disagree.** Black applicants are flagged at 1.43 times the White rate (demographic parity looks bad), yet a flag is more often correct for them (predictive parity looks fine). The decision log weights the false-positive-rate gap most heavily, because it names a concrete harm: creditworthy Black and American Indian applicants face extra scrutiny more often.
- **Individual fairness.** Among five nearest-neighbour applicants, the average maximum score gap is 0.115, and 24% of applicants have a gap above 0.10.

## Limitations

This is a single state, three years of data, and one model family. The HMDA data has no credit score, which is the main thing a real lender would use, so absolute accuracy would differ in practice. The fairness figures describe this dataset, not mortgage lending generally.

## Data source

Public HMDA loan-level data from the CFPB / FFIEC HMDA Data Browser API (California, 2022-2024, home purchase, first lien, principal residence). The warm-up dataset is the UCI "Default of Credit Card Clients" data, downloaded through OpenML.
