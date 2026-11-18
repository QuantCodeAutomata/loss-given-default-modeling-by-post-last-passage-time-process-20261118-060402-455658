# Repository guide

This repository replicates Egami, M. (2026), "Loss-Given-Default Modeling by
Post-Last Passage Time Process": the occupation-time/last-passage-time theory
for a transient linear diffusion (Section 2), the structural credit model on
the normalized asset-to-debt ratio process (Section 3), and the calibration
to six NYSE-listed firms (Table 2, Figures 1-3).

## Layout

- `src/` — flat library modules: `diffusion.py` (Laplace transform of the
  occupation time before the last passage time, Prop. 2.1), `post_last_passage.py`
  (h-transform drift/entrance-boundary, Prop. 2.2-2.3), `mle.py` (Duan's MLE,
  App. A.7), `calibration.py` (alpha*/chi calibration, Prop. 3.1, Eq. 3.13),
  `lgd.py` (LGD density, Corollary 3.2), `term_structure.py` (CDS term
  structure, Figure 3), and `config.py` (every experiment's parameters as
  module-level variables, each with a source comment).
- `exp/` — one script per experiment (`exp1.py` ... `exp8.py`), each calling
  only `src/` and `data/data_loader.py`.
- `data/data_loader.py` — loads real equity, shares-outstanding and
  fundamentals data for Experiments 3 and 6 through the QuantX Dataloader and
  Massive REST APIs; downloaded data is never committed.
- `tests/` — unit tests of `src/` on synthetic/known-answer inputs, plus one
  `market_data`-marked test of the loader on a short real slice.
- `results/` — tables, metrics (JSON/CSV) and plots written by the
  experiments, and `RESULTS.md` with the comparison against the paper.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill DATALOADER_API_KEY and MASSIVE_API_KEY for exp3/exp6

python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # also runs the market_data-marked loader test

python -m exp.exp1   # ... through
python -m exp.exp8

./reproduce.sh        # runs the tests and every experiment in sequence
```

Each experiment script overwrites its own files in `results/` on every run;
after a rerun, check the numbers and verdicts in `results/RESULTS.md` against
the newly written files and update them by hand — they are not regenerated
automatically.

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  `UPPER_CASE` variable with a comment giving its meaning, unit and source
  (a paper section/equation, Table 2, or "not specified in the paper"). No
  parameters are hardcoded inside `src/` or `exp/` modules.
- Market data is loaded only through `data/data_loader.py`; no module under
  `src/` or `exp/` calls an external API directly or writes a second loader.
- Tests on real market data are marked `@pytest.mark.market_data` and are
  skipped automatically when `DATALOADER_API_KEY`/`MASSIVE_API_KEY` are not
  set in the environment; all other tests run on synthetic fixtures or
  closed-form known answers and require no network access.
- API keys are read from an uncommitted `.env` file (see `.env.example`) via
  `python-dotenv`; they are never written to code, configuration, logs or
  results.
- Deviations from the paper's methodology (data source, calibration search
  bounds, firms that did not reproduce Table 2) are recorded only in
  `results/RESULTS.md`, under "Deviations from the paper" and
  "Not reproduced"; they are not described in code comments or config
  defaults beyond a one-line parameter source note.
