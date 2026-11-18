"""Experiment parameters for the replication of Egami (2026),
"Loss-Given-Default Modeling by Post-Last Passage Time Process".

All experiment parameters live here as module-level UPPER_CASE variables.
Each has a comment naming its meaning, unit, and source (paper section/
equation, or "not specified in the paper").
"""

# ---------------------------------------------------------------------------
# Experiment 1: Laplace transform of the occupation time before last passage
# ---------------------------------------------------------------------------
EXP1_M = -0.27          # drift of normalized asset-to-debt ratio process (Sec. 3, dimensionless); illustrative, not specified in the paper
EXP1_ALPHA = 0.0        # distress threshold alpha for the synthetic Brownian motion with drift (Sec. 2.1)
EXP1_X0 = 0.5           # starting position x >= alpha (Sec. 2.1); not specified in the paper, chosen above alpha
EXP1_Q_GRID = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0)  # Laplace parameters q>0 used to validate (2.4) against Monte Carlo
EXP1_C = -1.0           # interior killing boundary c < alpha used for formula (A.5); not specified in the paper
EXP1_MC_PATHS = 20000   # Monte Carlo paths for occupation-time validation; not specified in the paper
EXP1_MC_DT = 0.002      # Euler time step (years) for Monte Carlo simulation; not specified in the paper
EXP1_MC_T_MAX = 60.0    # simulation horizon (years), long enough for the last-passage time to occur a.s.; not specified in the paper
EXP1_MC_BATCH_SIZE = 1000  # path batch size for memory-bounded Monte Carlo (not a modeling parameter; not specified in the paper)
EXP1_INVLAP_T_GRID_MAX = 15.0   # upper bound of the t-grid for the inverse Laplace transform PDF (Fig. 2 style grid, Sec. 4.2)
EXP1_INVLAP_N_POINTS = 61       # number of points on the t-grid for the inverse Laplace PDF

# ---------------------------------------------------------------------------
# Experiment 2: Post-last-passage process dynamics / entrance boundary
# ---------------------------------------------------------------------------
EXP2_M = -0.27          # drift M of the normalized process (Sec. 3); same illustrative value as experiment 1
EXP2_ALPHA = 0.0        # threshold alpha* (Sec. 3.2)
EXP2_Z0_OFFSET = -0.01  # starting offset z0 = alpha - |offset| used in Prop. 2.3 simulation; not specified in the paper
EXP2_SIM_DT = 1e-4      # Euler-Maruyama time step; not specified in the paper
EXP2_SIM_T = 2.0        # simulation horizon (years) for the post-last-passage process; not specified in the paper
EXP2_N_PATHS = 2000     # number of simulated paths for the entrance-boundary check; not specified in the paper
EXP2_EPS_CLIP = 1e-6    # minimum distance to alpha enforced in the Euler scheme to regularize coth(.) (numerical safeguard)
EXP2_DENSITY_T = 0.5    # horizon t (years) at which the transition density p^{h_alpha}(t;z,y) is compared (Prop. 2.2)
EXP2_DENSITY_Z = -0.05  # starting point z < alpha for the transition-density comparison
EXP2_DENSITY_N_PATHS = 20000  # paths for the transition-density Monte Carlo histogram; not specified in the paper
EXP2_MC_BATCH_SIZE = 1000  # path batch size for memory-bounded Monte Carlo (not a modeling parameter; not specified in the paper)

# ---------------------------------------------------------------------------
# Experiments 3-8: structural model calibrated to six NYSE firms (Table 2, Sec. 4)
# ---------------------------------------------------------------------------
FIRM_TICKERS = ["WMB", "LEN", "WHR", "F", "DAL", "KSS"]  # the six firms analyzed in Section 4.1

OPTION_MATURITY_YEARS = 1.0   # T_m, option maturity used in Duan's MLE (Sec. 3.3.1), following Lehar (2005)
TRADING_DAYS_PER_YEAR = 252   # Delta_t convention: 1/252 per business day (Sec. 3.3.1)
MLE_WINDOW_DAYS = 126         # trailing business-day window for MLE estimation (Sec. 4.1)
MLE_SIGMA_BOUNDS = (0.05, 1.5)  # bracket for scipy.optimize.minimize_scalar over sigma (not specified in the paper)
CDS_HORIZON_YEARS = 5.0       # T, the CDS5 contract horizon used for calibration in Section 4 (Sec. 4.2)
RECOVERY_CONVENTION = 0.6     # standard market LGD convention (60%) used to convert CDS spread to a hazard-rate proxy S/0.6 (Sec. 3.3.3)

CALIBRATION_ALPHA_BOUNDS = (-3.0, 3.0)   # bracket for Brent's method root-finding on alpha* (Sec. 3.3.2); not specified in the paper
CALIBRATION_CHI_BOUNDS = (1e-6, 2000.0)  # bracket for Brent's method root-finding on chi (Sec. 3.3.3); not specified in the paper
CALIBRATION_XTOL = 1e-10      # Brent's method convergence tolerance; not specified in the paper

TERM_STRUCTURE_TENORS_YEARS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)  # tenors shown in Figure 3 for DAL's CDS term structure (Sec. 4.2)

# Table 2 (Sec. 4.2): paper-reported inputs and calibration outputs per firm.
# This is the paper's published data table (not a tunable configuration knob);
# it is used both as calibration input (evaluation date, r_f, r_i, 1/sigma*log(Y0),
# sigma, market PD, market CDS spread) and as the reference values that
# experiments 4-8 compare their computed outputs against.
FIRM_TABLE2 = {
    "WMB": {
        "evaluation_date": "2020-07-02", "r_f": 0.0029, "r_i": 0.0121,
        "log_y0_over_sigma": 1.8326, "sigma": 0.3818, "M_Q": -0.2225,
        "alpha_star": -0.8087, "chi": 68.4931, "q": 0.9190,
        "expected_lgd": 0.5549, "market_pd_pct": 6.522, "model_pd_pct": 6.522,
        "s_over_06_bps": 132.2500, "da_model_bps": 132.4524,
        "market_cds_bps": 79.35, "effective_cds_bps": 73.5078,
        "mu": -0.1274, "minimum_boundary": 0.2656,
    },
    "LEN": {
        "evaluation_date": "2020-06-03", "r_f": 0.0038, "r_i": 0.0088,
        "log_y0_over_sigma": 2.2821, "sigma": 0.5034, "M_Q": -0.2691,
        "alpha_star": -0.6201, "chi": 67.9011, "q": 0.9983,
        "expected_lgd": 0.6049, "market_pd_pct": 7.241, "model_pd_pct": 7.241,
        "s_over_06_bps": 144.9000, "da_model_bps": 147.0421,
        "market_cds_bps": 86.94, "effective_cds_bps": 88.9460,
        "mu": 0.1487, "minimum_boundary": 0.2681,
    },
    "WHR": {
        "evaluation_date": "2023-03-30", "r_f": 0.0366, "r_i": 0.0240,
        "log_y0_over_sigma": 1.8887, "sigma": 0.3505, "M_Q": -0.2437,
        "alpha_star": 0.0816, "chi": 33.7436, "q": 0.9190,
        "expected_lgd": 0.3537, "market_pd_pct": 13.715, "model_pd_pct": 13.715,
        "s_over_06_bps": 264.5333, "da_model_bps": 279.9755,
        "market_cds_bps": 158.72, "effective_cds_bps": 99.0366,
        "mu": 0.3227, "minimum_boundary": -0.0290,
    },
    "F": {
        "evaluation_date": "2021-01-13", "r_f": 0.0048, "r_i": 0.0204,
        "log_y0_over_sigma": 1.1566, "sigma": 0.1867, "M_Q": -0.2026,
        "alpha_star": -0.0161, "chi": 45.0904, "q": 1.8781,
        "expected_lgd": 0.1712, "market_pd_pct": 18.726, "model_pd_pct": 18.726,
        "s_over_06_bps": 407.4833, "da_model_bps": 407.4385,
        "market_cds_bps": 244.49, "effective_cds_bps": 69.7731,
        "mu": 0.0405, "minimum_boundary": 0.0030,
    },
    "DAL": {
        "evaluation_date": "2021-03-26", "r_f": 0.0085, "r_i": 0.0257,
        "log_y0_over_sigma": 1.6564, "sigma": 0.4347, "M_Q": -0.2764,
        "alpha_star": -0.2822, "chi": 229.5061, "q": 11.5237,
        "expected_lgd": 0.2564, "market_pd_pct": 19.680, "model_pd_pct": 19.680,
        "s_over_06_bps": 499.9333, "da_model_bps": 499.9333,
        "market_cds_bps": 299.96, "effective_cds_bps": 128.2030,
        "mu": 0.7681, "minimum_boundary": 0.1154,
    },
    "KSS": {
        "evaluation_date": "2022-07-07", "r_f": 0.0305, "r_i": 0.0253,
        "log_y0_over_sigma": 1.0501, "sigma": 0.5362, "M_Q": -0.3153,
        "alpha_star": -0.0751, "chi": 13.5562, "q": 0.8788,
        "expected_lgd": 0.5186, "market_pd_pct": 27.779, "model_pd_pct": 27.779,
        "s_over_06_bps": 603.7333, "da_model_bps": 617.6460,
        "market_cds_bps": 362.24, "effective_cds_bps": 320.3120,
        "mu": -0.4873, "minimum_boundary": 0.0395,
    },
}

# Experiment 7: occupation time above alpha* before killing at c* (Sec. 4.2, Fig. 2)
# c* = 2*alpha* for firms with alpha* < 0; c* = -2 for WHR (alpha* > 0) (Sec. 4.2)
OCCUPATION_TIME_T_GRID_MAX = 15.0   # upper bound of the t-grid used for Figure 2 (Sec. 4.2)
OCCUPATION_TIME_N_POINTS = 31       # number of points on the t-grid (0, 0.5, ..., 15)

# Experiment 8: DAL term-structure market CDS spreads (bps) and market PD (%) by tenor.
# Market CDS spreads are read directly from Figure 3 (Sec. 4.2); the paper does not
# report market default probabilities at tenors other than 5Y, so the PD term
# structure is interpolated/extrapolated from the 5Y market PD under a constant
# hazard-rate (flat intensity) assumption -- this is an explicit assumption, not a
# paper-stated value, and is recorded as such in results/RESULTS.md.
DAL_TERM_STRUCTURE_MARKET_SPREADS_BPS = {
    1.0: 125.0, 2.0: 170.0, 3.0: 215.0, 5.0: 300.0, 7.0: 335.0, 10.0: 350.0,
}  # Figure 3, Sec. 4.2 (market, blue curve)
DAL_TERM_STRUCTURE_MODEL_SPREADS_BPS_PAPER = {
    1.0: 17.0, 2.0: 58.0, 3.0: 95.0, 5.0: 128.0, 7.0: 147.0, 10.0: 170.0,
}  # Figure 3, Sec. 4.2 (model, red curve) -- reference values for comparison only
DAL_TERM_STRUCTURE_EXPECTED_LGD_PAPER = {
    1.0: 0.07097, 2.0: 0.20409, 3.0: 0.25793, 5.0: 0.25644, 7.0: 0.26375, 10.0: 0.29076,
}  # Figure 3 caption, Sec. 4.2 -- reference values for comparison only

# ---------------------------------------------------------------------------
# Data acquisition (experiments 3 and 6 only)
# ---------------------------------------------------------------------------
IDENTITY_DATE_BUFFER_DAYS = 400  # calendar-day lookback added before the evaluation date when requesting bars/fundamentals, to guarantee >=126 business days of history
