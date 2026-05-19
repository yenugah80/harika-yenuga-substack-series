# The Unnamed Pipeline

Data science techniques that production systems run daily and textbooks skip entirely.

Each episode covers one technique with working code, a specific industry application
(finance, entertainment, enterprise operations), and at least one failure case
encountered in production.

Companion code for the [Substack series](https://substack.com/@yenugah80).

## Episodes

| # | Technique | Industry | Code |
|---|-----------|----------|------|
| 01 | [Benford's Law](./episode-01-benfords-law/) | Finance, Entertainment, Enterprise | Python |
| 02 | [Survival Analysis](./episode-02-survival-analysis/) | SaaS, Telecom, Insurance, HR | Python |

## Structure

Each episode directory contains:

```
episode-NN-topic/
├── README.md          # Episode summary and quick start
├── *.py               # Core implementation module
├── visualize.py       # Charts and plots
├── example.py         # Runnable demo with test cases
├── requirements.txt   # Dependencies
└── tests/             # Unit tests
```

## Author

Hary (Harika Y) — AI Platform Engineer.
Building production AI systems. Writing about the parts nobody documents.

## License

MIT