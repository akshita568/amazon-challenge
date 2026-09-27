# Business Entity Resolution ML Challenge

## Overview
This project addresses the Business Entity Resolution task: matching business records across three distinct sources (`source1`, `source2`, and `source3`).

- **Reference Source**: `Source 1` is the deduplicated reference entity store.
- **Goal**: For each `Source 1` entity, identify corresponding matching records in `Source 2` and/or `Source 3`.
- **Ground Truth Format**:
  - `source1_entity_id`: Identifier of the reference record in Source 1.
  - `matched_entity_ids`: Comma-separated list of matched entity IDs from Source 2 and/or Source 3 (empty string if no match exists).

## Constraints & Requirements
- **No External Data / APIs**: External lookups, web scraping, or third-party query APIs of any kind are strictly forbidden (disqualifying if detected).
- **Model Constraints**: Final model must be MIT or Apache-2.0 licensed with at most 8B parameters.
- **Reproducibility**: Entire pipeline must execute and reproduce end-to-end self-contained from `code/business_entity_resolution/`.

## Project Structure
```
amazon-challenge/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py         # Centralized paths and project configurations
│       ├── output/               # Output predictions, models, and submission files
│       ├── README.md             # Project documentation
│       └── requirements.txt      # Project dependencies
└── dataset/
    ├── train/
    │   ├── train_ground_truth.tsv
    │   ├── train_source1.tsv
    │   ├── train_source2.tsv
    │   └── train_source3.tsv
    └── test/
        ├── test_source1.tsv
        ├── test_source2.tsv
        └── test_source3.tsv
```

## Setup & Installation
Install the project dependencies using pip:
```bash
pip install -r requirements.txt
```

Core dependencies:
- `pandas`
- `numpy`
- `scikit-learn`
- `rapidfuzz`
- `lightgbm`

## Data Verification
To verify data accessibility and inspect shapes and initial records:
```bash
python -m src.check_data
```
