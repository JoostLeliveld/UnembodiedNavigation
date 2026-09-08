# reading_independence

**Question:** commissioning makes the position accurate, so why is the filter's stated
uncertainty still dishonest?

**Answer:** camera readings are not independent. Readings 0.2 s apart are 89% alike, so the
filter's sqrt(N) shrinkage is unearned by about 4.2x. The controlled test is the thinning
replay: discarding four readings in five raises 95% ellipse coverage by up to 9 points while
costing accuracy, which no covariance model can do.

- `measure_independence.py` - reads the frozen six-run selection, computes residual
  autocorrelation and the 5 Hz vs 1 Hz comparison. Fits nothing, launches nothing.
- `plot_independence.py` - the two-panel figure.

Results, interpretation and limitations: `logs/studies/reading_independence_20260907/`.
