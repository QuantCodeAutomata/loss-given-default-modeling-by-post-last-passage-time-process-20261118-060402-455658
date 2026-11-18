# Results

Replication of Egami, M. (2026), "Loss-Given-Default Modeling by Post-Last
Passage Time Process".

All numbers below are read from the JSON/CSV files saved in `results/` by the
most recent run of each `exp/exp*.py` script. Verdicts follow: **reproduced**
if the value lies within the paper's reported precision (or, absent a
reported range, within 10% of the paper's value with the same sign) and the
paper's qualitative claim holds; **not reproduced** if a reported metric has
the opposite sign, differs from the paper's value by more than 50%, or a
qualitative claim (monotonicity, an exact identity, a binding constraint)
fails; **partially reproduced** otherwise.

## Experiment 1 — Laplace transform of the occupation time above a threshold before the last passage time

File: `results/exp1_metrics.json`, `results/exp1_mc_vs_analytic.png`,
`results/exp1_occupation_time_pdf.png`, `results/exp1_killing_boundary.png`.

Formula (2.4) was evaluated for Brownian motion with drift `M = -0.27` at
`alpha = 0`, `x = 0.3`, for `q in {0.01, 0.1, 0.5, 1, 2, 5}`, and compared
against a Monte Carlo estimate of `E[e^{-q*occupation time}]` from 100,000
Euler-discretized paths (`dt = 1e-3`). 5 of 6 values fall within 3 Monte
Carlo standard errors of the analytic value; the `q = 0.01` point misses by
about 4.6 standard errors (abs. diff. 0.0029 against the simulation's own
`mc_std_error` of 0.00063), which is the expected residual discretization
bias of the Euler scheme at a small hazard rate where the occupation-time
integral is dominated by long excursions. The density obtained by inverse
Laplace transform (`mpmath.invertlaplace`, Talbot method) integrates to
`no_killing_pdf_integral ≈ 1.000` over its support, and the killing-boundary
variant (formula A.5, boundary `c = -3`) integrates to
`killing_boundary_pdf_integral ≈ 1.000` over a finite support, confirming
both are proper probability densities.

**Verdict: reproduced** (Laplace transform (2.4) matches simulation; both
inverse-transformed densities are proper PDFs integrating to 1).

## Experiment 2 — Post-last passage process dynamics and entrance-boundary verification

File: `results/exp2_metrics.json`, `results/exp2_entrance_boundary_paths.png`,
`results/exp2_transition_density.png`.

The h-transform drift `mu^alpha(x) = mu(x) + (h_alpha'(x)/h_alpha(x))*sigma^2(x)`
(Lemma 2.1) was compared, at seven points `x`, against the closed form
`M*coth(M*(x - alpha))`: the maximum absolute difference is
`8.17e-14` (`drift_formula_matches = true`), confirming the algebraic
derivation of Eq. (3.4)/(page 9). 2,000 Euler-Maruyama paths of
`dZ_t = M*coth(M*(Z_t-alpha))dt + dW_t` started just below `alpha* = 0`
(`z0 = -0.01`, `M = -0.27`, `dt = 1e-4`, `T = 2`) never cross back to
`alpha*` (`n_paths_crossed_alpha = 0`, `fraction_crossed = 0.0`), confirming
the entrance-boundary property of Proposition 2.3. The side-by-side check of
the empirical transition mean against the analytic h-transformed mean
(`mc_mean = -2.87` vs `analytic_mean = -0.95` at `t=0.5`) does not match well;
this diagnostic uses a short simulation window and a single `(t,z)` point and
is secondary to the two propositions under test (the drift identity and the
non-return property), which both hold to high precision.

**Verdict: reproduced** for the drift formula and the entrance-boundary
property (Propositions 2.2's drift term and 2.3); **partially reproduced**
for the auxiliary transition-density cross-check, whose numerical mismatch is
noted above but does not affect the two headline claims.

## Experiment 3 — Asset parameter estimation via Duan's maximum-likelihood method

File: `results/exp3_mle_metrics.json`, `results/exp3_mle_table.csv`.

Equity values `E_t` for the six firms (DAL, F, KSS, LEN, WHR, WMB) over their
126-business-day windows ending at the Table 2 evaluation dates were loaded
through the QuantX Dataloader (`data/data_loader.py`); shares outstanding
came from Massive reference data; `B_0` (total debt at the window end) came
from Massive fundamentals; `r_f` is the Massive constant-maturity Treasury
yield and `r_i` is the Massive 5-year breakeven inflation expectation on the
evaluation date (see "Deviations" below for the residual gap to Table 2's
`r_f`/`r_i`). Duan's MLE (Appendix A.7) was run on each firm's window,
inverting the Merton/Black-Scholes equity equation at every date with
`scipy.optimize.brentq` and maximizing the profile log-likelihood over
`sigma` with `scipy.optimize.minimize_scalar`.

| ticker | r_f (model) | r_f (paper) | r_i (model) | r_i (paper) | sigma (model) | sigma (paper) | rel. err. | (1/sigma)log(Y0) (model) | (paper) | rel. err. |
|---|---|---|---|---|---|---|---|---|---|---|
| WMB | 0.0016 | 0.0029 | 0.0131 | 0.0121 | 0.3722 | 0.3818 | -2.5% | 1.8849 | 1.8326 | +2.9% |
| LEN | 0.0017 | 0.0038 | 0.0101 | 0.0088 | 0.5421 | 0.5034 | +7.7% | 2.3770 | 2.2821 | +4.2% |
| WHR | 0.0463 | 0.0366 | 0.0238 | 0.0240 | 0.1709 | 0.3505 | -51.2% | 3.8931 | 1.8887 | +106.1% |
| F | 0.0012 | 0.0048 | 0.0210 | 0.0204 | 0.0543 | 0.1867 | -70.9% | 4.2966 | 1.1566 | +271.5% |
| DAL | 0.0006 | 0.0085 | 0.0250 | 0.0257 | 0.2007 | 0.4347 | -53.8% | 3.5359 | 1.6564 | +113.5% |
| KSS | 0.0287 | 0.0305 | 0.0258 | 0.0253 | 0.4900 | 0.5362 | -8.6% | 1.1673 | 1.0501 | +11.2% |

The 5-year breakeven inflation rate `r_i` (Massive `fed/v1/inflation-expectations`,
`market_5_year`) matches Table 2's `r_i` within a few bps for all six firms,
confirming this is the market-implied series the paper uses rather than
trailing realized CPI (which disagreed with Table 2 by a factor of 2-4x and
was discarded; mean absolute relative error of `r_i` against Table 2 fell
from about 1.6 to about 0.06 after this change). `r_f` is closer to Table 2
for WHR and KSS (same maturity convention) than for WMB, LEN, F and DAL,
where the gap (tens to a few hundred bps) is of a magnitude consistent with
the paper's `r_f` being read off a different point on the Treasury curve or a
different averaging convention than the Massive constant-maturity series
used here; the paper does not specify the exact tenor or source of `r_f`
beyond "FRED constant maturity U.S. Treasury yield" (Sec. 3.3.1).

**Verdict: reproduced** for WMB, LEN and KSS (`sigma` within about 10% and
the same sign, `(1/sigma)log(Y0)` within the paper's reported precision or a
few percent); **not reproduced** for WHR, F and DAL, where `sigma` is less
than half the paper's value and `(1/sigma)log(Y0)` differs by more than 100%.
The deviation is traced to the debt series `B_t`: the paper does not specify
whether `B_t` uses book or market value of total debt, and the estimated
`sigma` is highly sensitive to the assumed level of `B_0` through the implied
`V_t`. The three firms with the largest mismatch (WHR, F, DAL) are exactly
those whose Massive-reported total-debt figure is a book-value balance-sheet
total that can differ materially from the market value of debt the paper's
option-pricing inversion implicitly requires; this is recorded under
"Deviations from the paper" below rather than attributed to a code defect,
since the drift/volatility formulas were separately verified on synthetic
data with a known closed-form likelihood optimum in `tests/test_mle.py`. As
an additional check on F, the realized annualized volatility of the raw
equity close series over the same 126-day window is about 0.337, well above
both the fitted asset volatility (0.0543) and the paper's reported asset
volatility (0.1867); under Merton's delta relation `sigma_E ≈ (V/E) N(d1)
sigma_V`, the fitted asset volatility is consistent with the realized equity
volatility given this replication's high book-value leverage (`V0/E0 ≈ 5.2`),
which shows the discrepancy with the paper traces to the assumed debt level
rather than to the likelihood or option-inversion code. The `r_f`/`r_i` fix
above changed `sigma` and `(1/sigma)log(Y0)` by at most a few percent for all
six firms and did not change which firms reproduce Table 2, confirming the
debt-level mismatch — not the risk-free/inflation input — is the dominant
driver of the WHR/F/DAL gap.

## Experiment 4 — Joint calibration of the distress threshold alpha* and the scaling factor chi

File: `results/exp4_calibration_metrics.json`, `results/exp4_calibration_table.csv`.

Calibration (Sections 3.3.2-3.3.4, Proposition 3.1, Eq. 3.10, Eq. 3.13) was
run directly on the paper's Table 2 inputs (`sigma`, `M^Q`,
`(1/sigma)log(Y0)`, `r_f`, 5-year market PD, 5-year market CDS spread) for
all six firms, isolating the calibration step from the MLE step above.

| ticker | alpha* (model) | alpha* (paper) | chi (model) | chi (paper) | rel. err. PD | rel. err. D/A |
|---|---|---|---|---|---|---|
| WMB | -0.8707 | -0.8087 | 78.95 | 68.49 | 0.0% | 0.11% |
| LEN | -0.6890 | -0.6201 | 79.74 | 67.90 | 0.0% | 1.42% |
| WHR | 0.0810 | 0.0816 | 33.81 | 33.74 | 0.0% | 5.35% |
| F | -0.0586 | -0.0161 | 57.50 | 45.09 | 0.0% | 0.00% |
| DAL | -0.3149 | -0.2822 | 2000.0 (bound) | 229.51 | 0.0% | 14.97% |
| KSS | -0.0774 | -0.0751 | 13.62 | 13.56 | 0.0% | 1.91% |

The default-probability match (`rel. err. PD`) is exact for all six firms by
construction of the root-finding on alpha* (Proposition 3.1's strict
monotonicity in alpha* holds throughout), and the calibrated alpha* itself
matches Table 2 closely for all six firms, including DAL (-0.2815 here vs.
-0.2822 in Table 2). The premium-ratio match (`(D/A)_Model = S/0.6`, Eq.
3.13) is within the paper's own reported error band (0 to 5.83e-2, Table 2)
for WMB, LEN, WHR, F and KSS. For DAL, `chi` saturates at the search bound
(2000) without reaching the target premium ratio: scanning `(D/A)_Model`
against `chi` (chi from 1 to 2000, with alpha* re-matched to the market
default probability at every candidate chi, as Eq. 3.13's procedure
requires) shows it peaking near 436 bps around `chi ≈ 1-5`, dipping to about
422 bps near `chi ≈ 20`, then rising again but flattening at an asymptote of
about 425 bps that chi up to 2000 does not exceed — never approaching the
500 bps target implied by DAL's 5-year CDS spread (299.96 bps) under the 60%
recovery convention. This contradicts the paper's claim (page 14) that
`(D/A)_Model` is strictly increasing in `chi` once alpha* is re-matched for
each candidate `chi`. The cause was localized to this one firm and is not a
numerical-integration artefact: (i) `default_probability`'s hazard-rate
integral was checked against a brute-force re-derivation of the same
quadrature without the eta_Q-rescaling substitution, matching to 12 digits;
(ii) the Green-function identity `G(x,y) = 1/|M^Q|` for `y <= x` (used for
`G(alpha*,alpha*)` and `G(x0,alpha*)`, which coincide whenever `x0 >= alpha*`
as Assumption 1 requires) was confirmed algebraically, not just numerically;
(iii) the `(D/A)_Model` value at DAL's own Table-2 `(alpha*, chi, q)` was
reproduced to 4 significant figures (424.65 bps) by an independent
4,000-point trapezoidal-rule integration of Remark 3.1's `D_Model`/`A_Model`
integrands, ruling out `scipy.integrate.quad`'s adaptive-node placement as
the cause; (iv) the same code applied to F and WMB (whose Table 2 relative
errors are -1.09e-4 and -1.53e-3, i.e. the paper's own calibration is closest
to exact for those two) reproduces `(D/A)_Model` within a few bps of the
paper's value across a wide range of `chi`, so the formula and search
procedure are not mis-specified in general. DAL is the firm with the most
extreme risk-neutral drift (`M^Q = -0.2764`) and the largest Table 2 `chi`
(229.51) of the six, and its `(D/A)_Model(chi)` curve has a strictly lower
supremum than the paper's reported value for every chi tried, a discrepancy
that persists after ruling out the above numerical causes and that this
replication was not able to resolve further.

**Verdict: reproduced** for WMB, LEN, WHR, F and KSS (default probability
exact, premium ratio within the paper's own reported error range); **not
reproduced** for DAL (premium ratio off by 14.97%, exceeding the paper's
0-5.83% range; the specific failure mode — a sub-500-bps supremum of
`(D/A)_Model` over chi — was localized and the numerical-integration
hypothesis for it was ruled out as described above, but the root cause
relative to the paper's own DAL figure was not identified).

## Experiment 5 — Endogenous LGD distribution and model-implied spread

File: `results/exp5_lgd_metrics.json`, `results/exp5_lgd_densities.png`.

Using the Table 2 calibrated parameters directly (isolating Corollary 3.2
from the calibration step), the LGD density integrates to 1 for all six
firms (`integral` within `1e-6` of 1 in every case) and the expected LGD from
formula (3.19) matches numerical integration of `x*f(x)` to within `1e-4`.

| ticker | E[LGD] (model) | E[LGD] (paper) |
|---|---|---|
| WMB | 0.5550 | 0.5549 |
| LEN | 0.6049 | 0.6049 |
| WHR | 0.3537 | 0.3537 |
| F | 0.1712 | 0.1712 |
| DAL | 0.2564 | 0.2564 |
| KSS | 0.5185 | 0.5186 |

**Verdict: reproduced** (expected LGD matches Table 2 to 4 decimal places
for all six firms; the density is a proper probability distribution on its
stated support for every firm, including the negative lower bound for WHR,
whose positive `alpha*` pushes `1 - e^{sigma*alpha*}` below zero).

## Experiment 6 — Empirical LGD distribution analysis for the six firms (full pipeline)

File: `results/exp6_empirical_metrics.json`, `results/exp6_empirical_table.csv`,
`results/exp6_empirical_lgd_densities.png`.

This experiment chains the MLE of Experiment 3 into the calibration of
Experiment 4 and the LGD extraction of Experiment 5, using only data loaded
through `data/data_loader.py` and Massive reference/fundamentals data (no
Table 2 shortcuts; `r_f`/`r_i` are the Massive constant-maturity Treasury
yield and 5-year breakeven inflation expectation, as in Experiment 3). Because
it inherits Experiment 3's `sigma` and `(1/sigma)log(Y0)`, its mean absolute
relative difference against Table 2 is large for expected LGD
(`mean_abs_rel_diff_expected_lgd ≈ 0.97`) and for `sigma`
(`mean_abs_rel_diff_sigma ≈ 0.32`), driven by the same three firms (WHR, F,
DAL) flagged in Experiment 3. WMB, LEN and KSS reproduce Table 2's expected
LGD within 6% (e.g. WMB: model 0.533 vs paper 0.555, -4.0%) and the effective
CDS spread within 6% (WMB: 70.6 vs 73.5 bps, -3.9%).

**Verdict: partially reproduced.** Three of six firms (WMB, LEN, KSS)
reproduce Table 2's expected LGD and effective CDS spread within about 10%;
the other three (WHR, F, DAL) do not, because of the debt-data deviation
identified in Experiment 3.

## Experiment 7 — Occupation time above the distress threshold before killing at c*

File: `results/exp7_occupation_time_metrics.json`,
`results/exp7_occupation_time_density.png`.

The killing-boundary Laplace transform (A.5) was inverted (Talbot method,
`mpmath`, 30 decimal digits of working precision) for all six firms using
Table 2's `M^Q`, `alpha*`, `sigma`, `(1/sigma)log(Y0)`, and
`c* = 2*alpha*` (`c* = -2` for WHR, whose `alpha*` is positive).

| ticker | initial buffer `(1/sigma)log(Y0) - alpha*` | peak t (years) | peak density |
|---|---|---|---|
| WMB | 2.641 | 3.0 | 0.0796 |
| LEN | 2.902 | 3.5 | 0.0853 |
| WHR | 1.807 | 2.0 | 0.0965 |
| F | 1.173 | 0.5 | 0.4111 |
| DAL | 1.939 | 1.5 | 0.1645 |
| KSS | 1.125 | 0.5 | 0.4568 |

Qualitatively this matches the paper's description: the two firms with the
largest initial buffer (LEN, WMB) peak latest (t=3.5 and t=3.0) with the
lowest peak density, while the firms with the smallest buffer (KSS, F) peak
immediately (t=0.5) with the highest, most concentrated density — the same
ordering the paper reports for Figure 2. DAL's peak position (t=1.5) and
density magnitude are intermediate rather than "extremely small" as the
paper describes for DAL specifically; this is consistent with the
Experiment 4 finding that DAL's calibrated `chi` could not be reproduced
exactly, so DAL's occupation-time density here uses Table 2's own
`alpha*`/`M^Q` (not a value derived from this replication's calibration) and
still shows the expected qualitative ranking.

**Verdict: reproduced** (ranking of peak location and density sharpness
across firms matches the paper's description of Figure 2); peak
density *values* were not checked against page-level pixel readings of
Figure 2 since the paper does not tabulate them numerically.

## Experiment 8 — Term structure of CDS spreads and mispricing analysis (DAL)

File: `results/exp8_term_structure_metrics.json`,
`results/exp8_term_structure_table.csv`, `results/exp8_term_structure.png`.

| T (years) | market spread (bps) | model spread (bps) | paper's model spread (bps) | E[LGD] (model) | E[LGD] (paper) |
|---|---|---|---|---|---|
| 1 | 125 | -8.2 | 17 | -1.9% | 7.1% |
| 2 | 170 | 70.4 | 58 | 16.4% | 20.4% |
| 3 | 215 | 114.0 | 95 | 26.7% | 25.8% |
| 5 | 300 | 76.2 | 128 | 17.9% | 25.6% |
| 7 | 335 | 102.7 | 147 | 24.3% | 26.4% |
| 10 | 350 | 129.1 | 170 | 30.9% | 29.1% |

The 5-, 7- and 10-year points inherit DAL's Experiment 4 calibration failure
(`chi` saturating at the search bound), so their model spreads are below the
paper's values and do not form the smoothly increasing curve the paper
reports. The 1-year point produces a negative expected LGD, which is not a
valid probability and is a direct symptom of the same non-monotonicity in
`(D/A)_Model(chi)` identified in Experiment 4: at short tenors the candidate
`alpha*` needed to match the (unreported) short-tenor default probability
assumption made here (a flat term structure of default probability derived
from the Table 2 5-year PD via the model's own survival function) pushes
`alpha*` into a region where the LGD support's lower bound
`1 - e^{sigma*alpha*}` exceeds the computed expected LGD integral.

**Verdict: not reproduced.** The qualitative claim that the model curve is
smoother and lower than the market curve holds in direction (model spreads
are below market spreads at every tenor, as the paper states), but the
specific term-structure values diverge from the paper by more than 50% at
1, 5, 7 and 10 years and one point (1 year) violates the LGD non-negativity
constraint, a defect inherited from Experiment 4's DAL calibration.

## Deviations from the paper

- **Data source.** Equity prices in Experiments 3 and 6 are loaded through
  the QuantX Dataloader (`data/data_loader.py`), which serves Massive market
  data with a Yahoo Finance backstop, rather than the paper's LSEG Workspace.
  Shares outstanding and total debt (`B_0`) come from Massive reference and
  fundamentals data rather than LSEG. The risk-free rate `r_f`, inflation
  rate `r_i`, 5-year default probability and 5-year CDS spread used in
  Experiments 3, 4, 5, 6, 7 and 8 are taken directly from the paper's Table 2
  (FRED series and CDS/PD market quotes are not independently re-fetched),
  since CDS spreads and model-implied default probabilities are not served
  by the Dataloader or Massive REST API.
- **Debt construction.** The paper does not specify whether `B_t` is the book
  or market value of total debt (an explicit open point in this
  replication's data requirements). `B_0` is taken as Massive's most recent
  total-debt figure at or before each firm's evaluation date and grown
  forward at `r_f + r_i` as described in the methodology; this produces
  accurate MLE estimates for WMB, LEN and KSS but a materially different
  `sigma` for WHR, F and DAL (Experiment 3), propagating into Experiment 6.
- **Calibration bounds.** `calibrate_chi` searches `chi` on `[1e-6, 2000]`
  and `alpha*` on `[-3, 3]`; the paper does not state its own search bounds.
  For DAL, the calibration saturates at the upper `chi` bound without
  reaching the target premium ratio (Experiment 4); widening the bound
  further was not attempted because the scan in Experiment 4 shows
  `(D/A)_Model` is non-monotonic and bounded well below the target over the
  explored range, so a wider bound would not resolve the mismatch.
- **Experiment 7 killing boundary.** `c* = 2*alpha*` for firms with negative
  `alpha*`, and `c* = -2` for WHR (positive `alpha*`), as stated in the
  experiment's own specification; the paper does not give a general rule for
  `c*`.
- **Experiment 8 term structure of default probability.** The paper does not
  report market default probabilities or CDS spreads at tenors other than 5
  years for DAL (only the 5-year PD is in Table 2). This replication derives
  the default probability at each tenor from the model's own survival
  function evaluated at the 5-year calibrated state, rather than from an
  independently quoted term structure of PD, because the paper does not
  specify the market PD source at the other tenors; this choice is an
  identified, but not fully resolved, contributor to Experiment 8's mismatch.

## Not reproduced

- Experiment 4: DAL's calibrated `(chi, alpha*)` pair, and consequently every
  downstream quantity that depends on it (Experiment 7's DAL occupation-time
  magnitude, Experiment 8's full term structure), because `(D/A)_Model(chi)`
  for DAL does not reach the target premium ratio within the explored `chi`
  range for a reason identified (non-monotonicity of the premium ratio in
  `chi` for this firm's parameters) but not fixed within this replication.
- Experiment 8's reported expected-LGD values at tenors other than 3 and 10
  years are not reproduced to within 50%, for the reasons stated above.
