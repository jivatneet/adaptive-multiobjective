import numpy as np

class QuantileDist:
    """
    Distribution defined by (possibly noisy) quantiles.
    probs are in (0,1); defaults to 0.01, 0.02, ..., 0.99 for 99-quantile input.
    """
    def __init__(self, quantiles, probs=None, fix_monotone=True):
        q = np.asarray(quantiles, dtype=float).ravel()
        if probs is None:
            p = np.arange(1, len(q) + 1, dtype=float) / (len(q) + 1)  # 0.01..0.99 for 99 q's
        else:
            p = np.asarray(probs, dtype=float).ravel()
        # sort by probability just in case
        idx = np.argsort(p)
        self.probs = p[idx]
        self.quantiles = q[idx]
        # enforce nondecreasing quantiles (monotone rearrangement)
        if fix_monotone:
            self.quantiles = np.maximum.accumulate(self.quantiles)

    # Percent-point function (quantile function, inverse CDF)
    def ppf(self, u):
        u = np.asarray(u, dtype=float)
        # linear interpolation between provided quantiles; clamp in [min,max]
        return np.interp(u, self.probs, self.quantiles,
                         left=self.quantiles[0], right=self.quantiles[-1])

    # CDF at x: invert the mapping (x -> prob) by swapping axes and interpolating
    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.quantiles, self.probs, left=0.0, right=1.0)

    # Draw n samples using inverse transform
    def sample(self, n=1, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        u = rng.random(n)
        return self.ppf(u)


# --- Convenience wrappers ---

def make_quantile_dist(qs, probs=None, fix_monotone=True):
    """Factory that returns (sample_fn, cdf_fn)."""
    dist = QuantileDist(qs, probs=probs, fix_monotone=fix_monotone)
    return dist.sample, dist.cdf

def sample_from_quantiles(qs, n, probs=None, seed=None):
    """One-shot sampler: draw n samples from the quantile-defined distribution."""
    rng = np.random.default_rng(seed)
    return QuantileDist(qs, probs=probs).sample(n=n, rng=rng)

def cdf_from_quantiles(qs, probs=None):
    """Return CDF function F(x)=P(Y<=x) built from the quantiles."""
    return QuantileDist(qs, probs=probs).cdf