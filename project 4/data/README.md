# Project 4 HMDA Slice

This folder contains a prepared slice of public HMDA loan-level data from the official CFPB/FFIEC HMDA Data Browser API.

It also contains the warmup tutorial dataset:

- `default_credit_card_clients.csv`: UCI Default of Credit Card Clients data downloaded through OpenML and saved locally for `tutorial_warmup.ipynb`.

- State: `CA`
- Years: `2022,2023,2024`
- API filters: home purchase and action taken in originated/approved/denied.
- Local post-filters: first lien and principal residence when those fields are present.
- Rows after filtering/capping: `850,278`
- Target: `target_denied`, where 1 means historical denial and 0 means originated or approved but not accepted.

Important: this target is a historical institutional action, not objective creditworthiness. Students must discuss label limitations, historical bias, missing information, and selection effects.

Regenerate with:

```bash
python scripts/make_hmda_slice.py --state CA --years 2022,2023,2024 --max-rows 900000 --out-dir data
```
