# Episode 01 — Benford's Law

The first-digit frequency test running inside every Big Four audit that most data scientists have never implemented.

## What it does

Tests whether the leading-digit distribution of a numerical dataset matches the logarithmic curve predicted by Benford's Law. Deviations indicate the data may contain artificial patterns (fraud, gaming of approval thresholds, data entry errors, system constraints).

## Where it is used

- **Finance**: IRS tax return screening, Big Four audit engagements, general ledger and accounts payable validation
- **Entertainment**: Streaming royalty auditing, box office reporting verification
- **Enterprise**: Procurement fraud detection, expense reimbursement analysis

## Quick start

```bash
pip install -r requirements.txt
python example.py
```

This runs three test cases:
1. 14,000 synthetic vendor invoices (clean — passes Benford's test)
2. 8,200 expense claims with $49 threshold gaming (flags digit 4 overrepresentation)
3. 5,000 uniform random integers (fails — negative control)

Charts are saved to `output/`.

## Usage in your own pipeline

```python
import pandas as pd
from benfords import benfords_test, validate_preconditions, print_report

df = pd.read_csv("your_transactions.csv")

# Check preconditions first
pre = validate_preconditions(df, "amount")
if not pre["is_valid"]:
    print(f"Dataset does not meet Benford's preconditions: {pre}")

# Run the test
results, chi2, p_value, is_anomalous = benfords_test(df, "amount")
print_report(results, chi2, p_value, is_anomalous, pre)
```

## Tests

```bash
pytest tests/ -v
```

## Preconditions

Benford's Law produces valid results only when:
- The dataset contains at least 500 observations
- Values span at least 2 orders of magnitude (e.g., $10 to $1,000)
- Values are naturally occurring measurements, not assigned identifiers (not ZIP codes, not sequential IDs)

The `validate_preconditions()` function checks these automatically.

## Read the full article

[The Unnamed Pipeline, Episode 1: Benford's Law](https://YOURSUBSTACK.substack.com) on Substack.
