#!/usr/bin/env python3
"""Create the Project 4 HMDA slice from the official CFPB/FFIEC API.

Default slice:
- California
- 2022, 2023, 2024
- Home purchase applications
- First lien
- Principal residence
- Action taken in {originated, approved but not accepted, denied}

The output remains large enough for a real project while avoiding a
multi-gigabyte nationwide download.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import subprocess
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd


API_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"

ACTION_MAP = {
    "1": "Originated",
    "2": "Approved but not accepted",
    "3": "Denied",
}


def build_url(year: int, state: str) -> str:
    params = {
        "states": state,
        "years": str(year),
        "actions_taken": "1,2,3",
        "loan_purposes": "1",
    }
    return f"{API_URL}?{urlencode(params, safe=',')}"


def read_hmda_csv(url: str) -> pd.DataFrame:
    result = subprocess.run(
        ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "300", url],
        check=True,
        capture_output=True,
    )
    payload = result.stdout
    try:
        payload = gzip.decompress(payload)
    except gzip.BadGzipFile:
        pass
    return pd.read_csv(io.BytesIO(payload), low_memory=False)


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {c.lower().replace(" ", "_"): c for c in columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def normalize_slice(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    mapping_candidates = {
        "activity_year": ["activity_year", "activity year"],
        "state_code": ["state_code", "state"],
        "action_taken": ["action_taken", "action taken"],
        "loan_type": ["loan_type", "loan type"],
        "loan_purpose": ["loan_purpose", "loan purpose"],
        "lien_status": ["lien_status", "lien status"],
        "occupancy_type": ["occupancy_type", "occupancy type"],
        "loan_amount": ["loan_amount", "loan amount"],
        "income": ["income"],
        "debt_to_income_ratio": ["debt_to_income_ratio", "debt-to-income ratio", "debt to income ratio"],
        "loan_to_value_ratio": ["loan_to_value_ratio", "loan-to-value ratio", "loan to value ratio"],
        "property_value": ["property_value", "property value"],
        "interest_rate": ["interest_rate", "interest rate"],
        "rate_spread": ["rate_spread", "rate spread"],
        "total_loan_costs": ["total_loan_costs", "total loan costs"],
        "applicant_age": ["applicant_age", "applicant age"],
        "derived_race": ["derived_race", "derived race"],
        "derived_ethnicity": ["derived_ethnicity", "derived ethnicity"],
        "derived_sex": ["derived_sex", "derived sex"],
        "ffiec_msa_md_median_family_income": [
            "ffiec_msa_md_median_family_income",
            "ffiec msa/md median family income",
        ],
        "tract_minority_population_percent": [
            "tract_minority_population_percent",
            "tract minority population percent",
        ],
        "tract_to_msa_income_percentage": [
            "tract_to_msa_income_percentage",
            "tract to msa income percentage",
        ],
        "tract_population": ["tract_population", "tract population"],
        "county_code": ["county_code", "county code"],
        "census_tract": ["census_tract", "census tract"],
        "lei": ["lei"],
    }
    selected = {}
    for standard, candidates in mapping_candidates.items():
        found = first_existing(cols, candidates)
        if found is not None:
            selected[standard] = found

    out = pd.DataFrame()
    for standard, original in selected.items():
        out[standard] = df[original]

    if "action_taken" not in out.columns:
        raise ValueError(f"Could not find action_taken column. Columns: {cols[:30]}")

    action_text = out["action_taken"].astype(str).str.strip()
    out["action_taken"] = action_text.map(ACTION_MAP).fillna(action_text)
    out["outcome_group"] = np.select(
        [
            out["action_taken"].eq("Originated") | out["action_taken"].eq("Approved but not accepted"),
            out["action_taken"].eq("Denied"),
        ],
        ["approved_or_originated", "denied"],
        default="ambiguous_or_excluded",
    )
    out["target_denied"] = out["outcome_group"].eq("denied").astype(int)

    rename = {
        "derived_race": "race_group",
        "derived_ethnicity": "ethnicity_group",
        "derived_sex": "sex_group",
        "applicant_age": "age_group",
        "tract_minority_population_percent": "tract_minority_share",
        "tract_to_msa_income_percentage": "tract_income_ratio",
    }
    out = out.rename(columns=rename)

    numeric_cols = [
        "activity_year",
        "loan_amount",
        "income",
        "loan_to_value_ratio",
        "property_value",
        "interest_rate",
        "rate_spread",
        "total_loan_costs",
        "ffiec_msa_md_median_family_income",
        "tract_minority_share",
        "tract_income_ratio",
        "tract_population",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "tract_minority_share" in out.columns:
        out["tract_minority_share"] = out["tract_minority_share"] / 100.0
    if "tract_income_ratio" in out.columns:
        out["tract_income_ratio"] = out["tract_income_ratio"] / 100.0

    return out


def balanced_cap(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    strata = ["activity_year", "target_denied"]
    strata = [c for c in strata if c in df.columns]
    if not strata:
        return df.sample(max_rows, random_state=seed).sort_index()
    frac = max_rows / len(df)
    return (
        df.groupby(strata, group_keys=False)
        .sample(frac=frac, random_state=seed)
        .sample(min(max_rows, max_rows), random_state=seed)
        .sort_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="CA")
    parser.add_argument("--years", default="2022,2023,2024")
    parser.add_argument("--max-rows", type=int, default=900_000)
    parser.add_argument("--seed", type=int, default=4146)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    frames = []
    urls = []
    for year in [int(y.strip()) for y in args.years.split(",") if y.strip()]:
        url = build_url(year, args.state)
        print(f"Downloading {year} {args.state}: {url}", flush=True)
        raw = read_hmda_csv(url)
        print(f"  raw rows: {len(raw):,}; columns: {len(raw.columns)}", flush=True)
        frames.append(normalize_slice(raw))
        urls.append(url)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["outcome_group"].isin(["approved_or_originated", "denied"])].copy()
    for col, accepted in {
        "lien_status": {"1", "First lien", 1},
        "occupancy_type": {"1", "Principal residence", 1},
    }.items():
        if col in combined.columns:
            combined = combined[combined[col].astype(str).isin({str(x) for x in accepted})].copy()
    combined = balanced_cap(combined, args.max_rows, args.seed).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "hmda_project4_slice.csv.gz"
    parquet_path = args.out_dir / "hmda_project4_slice.parquet"
    meta_path = args.out_dir / "hmda_project4_slice_metadata.json"
    readme_path = args.out_dir / "README.md"

    combined.to_csv(csv_path, index=False)
    try:
        combined.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:
        parquet_written = False
        print(f"Parquet write skipped: {type(exc).__name__}: {exc}", flush=True)

    metadata = {
        "source": "Official CFPB/FFIEC HMDA Data Browser API",
        "api_base_url": API_URL,
        "download_urls": urls,
        "state": args.state,
        "years": args.years,
        "filters": {
            "actions_taken": "1,2,3",
            "loan_purposes": "1",
            "local_post_filter_lien_status": "1 when column is present",
            "local_post_filter_occupancy_type": "1 when column is present",
        },
        "rows": int(len(combined)),
        "columns": list(combined.columns),
        "target": {
            "name": "target_denied",
            "positive_class": "Denied",
            "negative_class": "Originated or approved but not accepted",
        },
        "parquet_written": parquet_written,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme_path.write_text(
        "# Project 4 HMDA Slice\n\n"
        "This folder contains a prepared slice of public HMDA loan-level data from the official CFPB/FFIEC HMDA Data Browser API.\n\n"
        f"- State: `{args.state}`\n"
        f"- Years: `{args.years}`\n"
        "- API filters: home purchase and action taken in originated/approved/denied.\n"
        "- Local post-filters: first lien and principal residence when those fields are present.\n"
        f"- Rows after filtering/capping: `{len(combined):,}`\n"
        "- Target: `target_denied`, where 1 means historical denial and 0 means originated or approved but not accepted.\n\n"
        "Important: this target is a historical institutional action, not objective creditworthiness. Students must discuss label limitations, historical bias, missing information, and selection effects.\n\n"
        "Regenerate with:\n\n"
        "```bash\n"
        "python scripts/make_hmda_slice.py --state CA --years 2022,2023,2024 --max-rows 900000 --out-dir data\n"
        "```\n",
        encoding="utf-8",
    )

    print(f"Wrote {csv_path} ({len(combined):,} rows)")
    if parquet_written:
        print(f"Wrote {parquet_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
