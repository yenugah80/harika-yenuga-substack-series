# The Unnamed Pipeline

> Practical data science techniques hiding inside real production systems.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/tests-pytest-2ecc71?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/license-MIT-f0f6fc?style=for-the-badge)](./LICENSE)

The Unnamed Pipeline is a companion codebase for the
[Substack series](https://substack.com/@harikayenuga): a collection of field-tested
statistical methods explained through real business problems.

Each episode starts with a messy industry question, translates it into a
statistical lens, then ships the working Python: implementation, examples,
visualizations, tests, and the failure modes worth remembering.

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

## What Makes It Different

This is not a notebook graveyard. Each episode is built like a small production
module:

| Layer | What You Get |
|-------|--------------|
| Core logic | Reusable Python functions with clear inputs and outputs |
| Demo | Synthetic but realistic data and runnable analysis scripts |
| Visuals | Dark-themed charts designed for explanation and review |
| Tests | Pytest coverage for expected behavior and edge cases |
| Readme | Quick start, concepts, limitations, and usage examples |

## Current Map

### Episode 01: Benford's Law

Detect suspicious numeric distributions with first-digit frequency analysis.
Useful for invoices, ledgers, operational metrics, and any workflow where
human-shaped numbers can leak into supposedly natural data.

```bash
cd episode-01-benfords-law
pip install -r requirements.txt
python example.py
python -m pytest tests/ -v
```

### Episode 02: Survival Analysis in Customer Churn

Model time-to-cancellation instead of flattening churn into yes/no. Includes
Kaplan-Meier curves, log-rank tests, Cox Proportional Hazards modeling,
individual survival prediction, and urgent customer flagging.

```bash
cd episode-02-survival-analysis
pip install -r requirements.txt
python example.py
python -m pytest tests/ -v
```

## Repository Structure

```text
episode-NN-topic/
|-- README.md          # Episode summary and quick start
|-- *.py               # Core implementation module
|-- visualize.py       # Charts and plots
|-- example.py         # Runnable demo with test cases
|-- requirements.txt   # Dependencies
`-- tests/             # Unit tests
```

## Design Principles

- **Industry first:** every technique is anchored to a real operating question.
- **Code over vibes:** examples must run, tests must pass, outputs must exist.
- **Failure-aware:** limitations and misleading cases are part of the lesson.
- **Small enough to read:** each episode is a compact, inspectable module.

## Author

Hary (Harika Y), AI Platform Engineer.

Building production AI systems and writing about the parts nobody documents.

## License

MIT
