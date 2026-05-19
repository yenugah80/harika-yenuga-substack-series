# The Unnamed Pipeline

> Data science techniques that production systems run daily and textbooks skip entirely.

The Unnamed Pipeline is a Substack series about statistical methods that quietly
power real business decisions. Each episode takes one practical technique,
implements it in Python, and shows where it belongs in an operating pipeline.

```mermaid
flowchart LR
    A[Business question] --> B[Statistical method]
    B --> C[Python implementation]
    C --> D[Tests and visuals]
    D --> E[Decision-ready signal]
```

## Episodes

| # | Episode | Real Question | Industry | Code |
|---|---------|---------------|----------|------|
| 01 | [Benford's Law](./episode-01-benfords-law/) | Are these numbers too perfect? | Finance, Entertainment, Enterprise | Python |
| 02 | [Survival Analysis](./episode-02-survival-analysis/) | When will customers churn? | SaaS, Telecom, Insurance, HR | Python |

## Current Map

### Episode 01: Benford's Law

Detect suspicious numeric distributions with first-digit frequency analysis.
Useful for invoices, ledgers, operational metrics, and any workflow where
human-shaped numbers can leak into supposedly natural data.

```bash
cd the-unnamed-pipeline/episode-01-benfords-law
pip install -r requirements.txt
python example.py
python -m pytest tests/ -v
```

### Episode 02: Survival Analysis in Customer Churn

Model time-to-cancellation instead of flattening churn into yes/no. Includes
Kaplan-Meier curves, log-rank tests, Cox Proportional Hazards modeling,
individual survival prediction, and urgent customer flagging.

```bash
cd the-unnamed-pipeline/episode-02-survival-analysis
pip install -r requirements.txt
python example.py
python -m pytest tests/ -v
```

## Episode Structure

```text
episode-NN-topic/
|-- README.md          # Episode summary and quick start
|-- *.py               # Core implementation module
|-- visualize.py       # Charts and plots
|-- example.py         # Runnable demo with test cases
|-- requirements.txt   # Dependencies
`-- tests/             # Unit tests
```
