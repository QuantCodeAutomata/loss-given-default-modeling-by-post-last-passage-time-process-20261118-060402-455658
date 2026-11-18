# Loss-Given-Default Modeling by Post-Last Passage Time Process

Replication of Egami, M. (2026), "Loss-Given-Default Modeling by Post-Last
Passage Time Process" (working paper).

This repository implements the paper's occupation-time and last-passage-time
theory for a transient linear diffusion (Section 2), the structural credit
model built on the normalized asset-to-debt ratio process (Section 3), and
the calibration to six NYSE-listed firms (Section 4), together with the
numerical experiments described below.

Status: implementation in progress. This README is updated as experiments
are added; see `results/RESULTS.md` for the final comparison against the
paper's reported numbers.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # only needed for experiments 3 and 6 (real market data)
```

`DATALOADER_API_KEY` and `MASSIVE_API_KEY` are only required for experiments
3 and 6, which estimate the structural model from real equity, debt,
risk-free-rate and inflation-rate data. Experiments 1, 2, 4, 5, 7 and 8 use
only the paper's own closed-form formulas and the parameters it reports (its
Table 2 and Figures 1-3) and need no API keys.

## Tests

```bash
python -m pytest -m "not market_data"
```

## Experiments

```bash
python -m exp.exp1   # Laplace transform of the occupation time before the last passage time (Prop. 2.1)
```

Further experiments (`exp2` through `exp8`) are being added; each will be
runnable the same way, `python -m exp.<name>`, and documented here once
implemented.

## Layout

- `src/` — the model: diffusion/occupation-time formulas (`diffusion.py`),
  the post-last-passage h-transform (`post_last_passage.py`), Duan's MLE
  (`mle.py`), the (alpha*, chi) calibration (`calibration.py`), the LGD
  density (`lgd.py`), the CDS term structure (`term_structure.py`), and all
  parameters (`config.py`).
- `exp/` — one script per experiment, calling only `src/`.
- `data/` — `data_loader.py` loads real market data for experiments 3 and 6
  through the QuantX Dataloader and Massive REST APIs; no data is generated
  synthetically for persistence (downloaded data is never committed).
- `tests/` — unit tests of `src/` on small/synthetic inputs.
- `results/` — tables, plots and `RESULTS.md` produced by the experiments.
