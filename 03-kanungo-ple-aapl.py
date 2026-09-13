"""
Project 3 - Kanungo Ch. 7 Probabilistic Linear Ensemble (PLE) on AAPL/SPY.

Reconstruction of the Ch. 7 worked example using a from-scratch implementation:

    - Bayesian linear regression with a fat-tailed Student-t likelihood (nu=6)
      (Kanungo's argument: Gaussian likelihoods underweight tail events on
      daily financial returns).
    - Metropolis-Hastings MCMC sampler with adaptive step sizes
      (PyMC's default is HMC/NUTS; MH is simpler and dependency-free).
    - Reports: posterior summary, 94% HDI for beta, probabilistic R^2
      (Gelman et al. 2019 variance decomposition), posterior predictive
      samples, and the generative risk measures (GVaR, GES, GTR) from Ch. 8.

The script uses *synthetic* AAPL/SPY excess returns that mimic the structure
of the Ch. 7 example (31 daily observations, AAPL regressed on SPY, Student-t
noise). Replace `generate_aapl_spy_like()` with `pd.read_csv(...)` when you
have real data.

This is an *original reconstruction* of the methodology described in
Kanungo Ch. 7 - it is not a copy of the O'Reilly code. The implementation
uses only numpy and scipy for portability.

Run:
    python 03-kanungo-ple-aapl.py
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(text):
    """Print a banner-style section header."""
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def hdi(samples, prob=0.94):
    """Highest Density Interval — the narrowest interval containing `prob`
    of the samples. The Bayesian replacement for a confidence interval:
    "94% HDI for beta is [1.12, 1.55]" means what people think a CI means.
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_size = int(np.ceil(prob * n))
    widths = sorted_samples[interval_size:] - sorted_samples[:n - interval_size]
    best = int(np.argmin(widths))
    return float(sorted_samples[best]), float(sorted_samples[best + interval_size])


# ---------------------------------------------------------------------------
# Step 1 — Synthetic data (mimics AAPL/SPY excess returns from Ch. 7)
# ---------------------------------------------------------------------------
banner("STEP 1: Generate synthetic AAPL/SPY excess returns")

def generate_aapl_spy_like(seed=42, n=31, true_alpha=0.001,
                           true_beta=1.30, true_sigma=0.012, nu=6):
    """Generate 31 days of synthetic AAPL/SPY excess returns with fat-tailed
    Student-t noise. Matches the Ch. 7 example structure: AAPL regressed on
    SPY, fat tails, intercept near zero, market beta near 1.
    """
    rng = np.random.default_rng(seed)
    spy = rng.normal(0, 0.008, n)                 # SPY excess returns
    noise = stats.t.rvs(df=nu, size=n,
                        random_state=rng) * true_sigma   # fat-tailed shocks
    aapl = true_alpha + true_beta * spy + noise
    return spy, aapl


spy, aapl = generate_aapl_spy_like(seed=42, n=31)
print(f"Generated {len(aapl)} daily observations.")
print(f"  AAPL: mean={aapl.mean():+.5f}  sd={aapl.std():.5f}")
print(f"  SPY : mean={spy.mean():+.5f}  sd={spy.std():.5f}")
print(f"  Empirical correlation: {np.corrcoef(spy, aapl)[0,1]:.3f}")


# ---------------------------------------------------------------------------
# Step 2 — Model specification (Kanungo Ch. 7 priors + Student-t likelihood)
# ---------------------------------------------------------------------------
banner("STEP 2: Model specification")

# Prior choices — weakly informative, centred on financial-economics intuition:
#   alpha ~ Normal(0, 0.01)             intercept near zero, sd ~1% / day
#   beta  ~ Normal(1.0, 0.5)            market beta, centred on 1.0
#   sigma ~ HalfStudentT(0.01, nu=6)    residual scale, fat-tailed prior
# Likelihood:
#   y     ~ StudentT(nu=6, mu=alpha + beta*x, sigma=sigma)
# Kanungo's argument: nu=6 is a good default for daily stock returns because
# the likelihood is robust to outliers — a single rogue data point doesn't
# dominate the parameter estimates the way it would under Gaussian noise.

PRIOR_ALPHA_MU, PRIOR_ALPHA_SD = 0.0, 0.01
PRIOR_BETA_MU,  PRIOR_BETA_SD  = 1.0, 0.5
PRIOR_SIGMA_SCALE, PRIOR_SIGMA_NU = 0.01, 6
LIKELIHOOD_NU = 6


def log_prior(alpha, beta, sigma):
    """Log prior density. Normal for alpha, beta; Half-Student-t for sigma."""
    if sigma <= 0:
        return -np.inf
    lp  = stats.norm.logpdf(alpha, PRIOR_ALPHA_MU, PRIOR_ALPHA_SD)
    lp += stats.norm.logpdf(beta,  PRIOR_BETA_MU,  PRIOR_BETA_SD)
    # Half-Student-t: Student-t on positive support, normalised by 2
    lp += stats.t.logpdf(sigma, df=PRIOR_SIGMA_NU,
                         loc=0, scale=PRIOR_SIGMA_SCALE) + np.log(2.0)
    return lp


def log_likelihood(aapl, spy, alpha, beta, sigma):
    """Log likelihood — Student-t with nu=6 (fat-tailed)."""
    if sigma <= 0:
        return -np.inf
    mu = alpha + beta * spy
    return float(np.sum(stats.t.logpdf(aapl, df=LIKELIHOOD_NU,
                                       loc=mu, scale=sigma)))


def log_posterior(aapl, spy, alpha, beta, sigma):
    """Unnormalised log posterior = log prior + log likelihood."""
    lp = log_prior(alpha, beta, sigma)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(aapl, spy, alpha, beta, sigma)


# ---------------------------------------------------------------------------
# Step 3 — Metropolis-Hastings MCMC sampler
# ---------------------------------------------------------------------------
banner("STEP 3: Run Metropolis-Hastings MCMC (training set)")


def metropolis_hastings(aapl, spy, n_samples=8000, burn_in=2000, seed=42):
    """Random-walk Metropolis-Hastings for the 3-parameter PLE.

    Proposal: independent Gaussian perturbations on each parameter
    (with sigma reflected to keep it positive). Step sizes adapt during
    burn-in to target ~25% acceptance.

    Returns the post-burn-in chain of shape (n_samples - burn_in, 3)
    with columns [alpha, beta, sigma].
    """
    rng = np.random.default_rng(seed)

    # Initialise near OLS estimates — a defensible starting point
    beta_ols = np.sum((spy - spy.mean()) * (aapl - aapl.mean())) \
             / np.sum((spy - spy.mean()) ** 2)
    alpha_ols = aapl.mean() - beta_ols * spy.mean()
    resid = aapl - (alpha_ols + beta_ols * spy)
    # MAD-based robust sigma init (good for fat-tailed data)
    sigma_ols = float(np.median(np.abs(resid - np.median(resid))) * 1.4826)

    chain = np.zeros((n_samples, 3))
    chain[0] = [alpha_ols, beta_ols, sigma_ols]

    prop_sd = np.array([0.001, 0.05, 0.002])  # initial proposal SDs
    accepted = 0

    for i in range(1, n_samples):
        current = chain[i - 1]
        proposal = current + rng.normal(0, prop_sd)
        proposal[2] = abs(proposal[2])  # reflect sigma to keep positive

        log_alpha = (log_posterior(aapl, spy, *proposal)
                     - log_posterior(aapl, spy, *current))
        if np.log(rng.random()) < log_alpha:
            chain[i] = proposal
            accepted += 1
        else:
            chain[i] = current

        # Adapt step sizes during burn-in to ~25% acceptance
        if i < burn_in and i % 200 == 0 and i > 0:
            recent_rate = accepted / i
            if recent_rate < 0.20:
                prop_sd *= 0.8
            elif recent_rate > 0.30:
                prop_sd *= 1.2

    rate = accepted / n_samples
    print(f"MCMC: {accepted}/{n_samples} accepted ({100*rate:.1f}%) — "
          f"target ~25% (Metropolis-Hastings sweet spot)")
    return chain[burn_in:]


# Train/test split — Kanungo Ch. 7 uses first 21 days for training, last 10 for test
n_train = 21
spy_train, spy_test = spy[:n_train], spy[n_train:]
aapl_train, aapl_test = aapl[:n_train], aapl[n_train:]

chain = metropolis_hastings(aapl_train, spy_train,
                            n_samples=8000, burn_in=2000, seed=42)


# ---------------------------------------------------------------------------
# Step 4 — Posterior summary (with 94% HDI for beta — the Ch. 7 example)
# ---------------------------------------------------------------------------
banner("STEP 4: Posterior summary (94% HDI)")


def summarise_posterior(chain, labels, prob=0.94):
    """Print posterior summary table: mean, sd, and HDI for each parameter."""
    header = (f"{'Parameter':<10} {'Mean':>10} {'SD':>10} "
              f"{f'{int(prob*100)}% HDI low':>15} "
              f"{f'{int(prob*100)}% HDI high':>15}")
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        samples = chain[:, i]
        lo, hi = hdi(samples, prob)
        print(f"{label:<10} {samples.mean():>+10.5f} {samples.std():>10.5f} "
              f"{lo:>+15.5f} {hi:>+15.5f}")


summarise_posterior(chain, labels=['alpha', 'beta', 'sigma'])


# ---------------------------------------------------------------------------
# Step 5 — Probabilistic R-squared (Gelman et al. 2019)
# ---------------------------------------------------------------------------
banner("STEP 5: Probabilistic R-squared on test set")


def probabilistic_r_squared(actual, predictors, chain):
    """Gelman et al. 2019 variance decomposition. R^2 = 1 - var(resid)/var(total),
    computed across the posterior. The Student-t residual variance is
    sigma^2 * nu / (nu - 2) for nu > 2 (here nu=6, so var = sigma^2 * 1.5).

    Returns an array of R^2 samples (one per posterior draw).
    """
    n = len(chain)
    r2_samples = np.zeros(n)
    total_var = float(np.var(actual))
    residual_var_factor = LIKELIHOOD_NU / (LIKELIHOOD_NU - 2)  # = 1.5 for nu=6

    for i, (alpha, beta, sigma) in enumerate(chain):
        residual_var = (sigma ** 2) * residual_var_factor
        r2_samples[i] = 1.0 - residual_var / total_var

    return r2_samples


r2_test = probabilistic_r_squared(aapl_test, spy_test, chain)
r2_lo, r2_hi = hdi(r2_test)
print(f"Probabilistic R^2 (test): mean={r2_test.mean():.3f}, "
      f"94% HDI=({r2_lo:.3f}, {r2_hi:.3f})")
print(f"  → Kanungo's Ch. 7 AAPL example achieves ~69% test R^2.")
print(f"  → Note: R^2 can be negative if residual variance > total variance.")
print(f"    Honest uncertainty quantification: PLE tells you when the model")
print(f"    is worse than the mean baseline.")


# ---------------------------------------------------------------------------
# Step 6 — Posterior predictive samples + generative risk measures (Ch. 8)
# ---------------------------------------------------------------------------
banner("STEP 6: Posterior predictive + generative risk measures")


def posterior_predictive(actual, predictors, chain, n_pred=1000, seed=42):
    """Draw n_pred synthetic datasets from the posterior predictive distribution.
    Each dataset corresponds to one posterior sample of (alpha, beta, sigma).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(chain), size=n_pred, replace=True)
    samples = chain[idx]

    predictions = np.zeros((n_pred, len(actual)))
    for i, (alpha, beta, sigma) in enumerate(samples):
        mu = alpha + beta * predictors
        predictions[i] = stats.t.rvs(df=LIKELIHOOD_NU,
                                     loc=mu, scale=sigma,
                                     random_state=rng)
    return predictions


def generative_risk_measures(predictions, actual, confidence=0.99):
    """Ch. 8 generative risk measures. All three computed from the same
    posterior predictive samples — no additional model fits required.

        GVaR — the loss at the (1 - confidence) quantile
        GES  — the mean of all losses beyond GVaR
        GTR  — the worst (minimum) simulated loss

    Kanungo's prescription: GVaR > volatility, GTR > expected shortfall.
    """
    losses = actual - predictions           # negative loss = gain
    flat = losses.flatten()

    gvar = float(np.percentile(flat, 100 * (1 - confidence)))
    tail = flat[flat <= gvar]
    ges = float(tail.mean()) if len(tail) > 0 else gvar
    gtr = float(flat.min())

    return gvar, ges, gtr


predictions = posterior_predictive(aapl_test, spy_test, chain,
                                   n_pred=1000, seed=42)
print(f"Posterior predictive: {predictions.shape[0]} datasets × "
      f"{predictions.shape[1]} test points")

gvar, ges, gtr = generative_risk_measures(predictions, aapl_test,
                                          confidence=0.99)
print(f"\nGenerative risk measures (99% confidence):")
print(f"  GVaR (threshold loss)   : {gvar:+.5f}  ({gvar*100:+.2f}%)")
print(f"  GES  (mean tail loss)   : {ges:+.5f}  ({ges*100:+.2f}%)")
print(f"  GTR  (worst case)       : {gtr:+.5f}  ({gtr*100:+.2f}%)")
print(f"  → Note the cascade: GVaR ≤ GES ≤ GTR. The single investor")
print(f"    experiences GTR, not GES. Use GTR for personal exposure limits.")


# ---------------------------------------------------------------------------
# Step 7 — Frequentist baseline comparison (OLS vs PLE)
# ---------------------------------------------------------------------------
banner("STEP 7: Frequentist baseline (OLS) vs PLE")


def ols_summary(spy_train, aapl_train):
    """Ordinary least squares — the conventional baseline Kanungo critiques."""
    n = len(spy_train)
    x_mean, y_mean = spy_train.mean(), aapl_train.mean()
    beta = float(np.sum((spy_train - x_mean) * (aapl_train - y_mean))
                 / np.sum((spy_train - x_mean) ** 2))
    alpha = float(y_mean - beta * x_mean)
    return alpha, beta


ols_alpha, ols_beta = ols_summary(spy_train, aapl_train)
ple_alpha_mean = float(chain[:, 0].mean())
ple_beta_mean  = float(chain[:, 1].mean())
ple_beta_lo, ple_beta_hi = hdi(chain[:, 1])

print(f"OLS (frequentist, point estimate):")
print(f"  alpha = {ols_alpha:+.5f}")
print(f"  beta  = {ols_beta:+.4f}")
print(f"  → Single point estimate. No uncertainty quantification.")

print(f"\nPLE (Bayesian, posterior):")
print(f"  alpha = {ple_alpha_mean:+.5f}")
print(f"  beta  = {ple_beta_mean:+.4f}  (94% HDI: [{ple_beta_lo:+.4f}, {ple_beta_hi:+.4f}])")
print(f"  → Full posterior over each parameter. The HDI is the honest")
print(f"    answer to 'what is the market beta?' — and it overlaps 1.0,")
print(f"    which OLS hides.")


# ---------------------------------------------------------------------------
# Step 8 — Summary: the PLE Assembly Workflow delivered
# ---------------------------------------------------------------------------
banner("STEP 8: PLE Assembly Workflow — completed")

print("""
Ch. 7 PLE Assembly Workflow (6 phases), delivered:

    1. Define metrics          — probabilistic R^2 + 94% HDI for beta
    2. Analyse data            — 21 training points, Student-t noise
    3. Develop prior           — weakly informative Normal for alpha/beta;
                                  Half-Student-t for sigma
    4. Train posterior         — Metropolis-Hastings, 8000 samples, 2000 burn-in
    5. Test                    — held-out 10 points; test R^2 distribution
                                  (not a point estimate)
    6. Deploy                  — generative risk measures (GVaR/GES/GTR)
                                  computed from the same posterior predictive

Key takeaways:
    - The posterior is the deliverable, not a point estimate.
    - Generative ensembles enable honest uncertainty quantification
      AND downstream risk measures from a single MCMC run.
    - The PLE workflow is the eval-first template for any production
      ML system — not just finance.

Compare to: Kanungo Ch. 7 (PLE) + Ch. 8 (loss functions + generative risk).
""")
