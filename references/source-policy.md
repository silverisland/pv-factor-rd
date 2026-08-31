# Expert source policy

Use sources to create mechanisms and hypotheses, not to declare factors valid.

## Priority

1. Physical laws, standards, and official NREL/Sandia/IEA/PVPMC guidance.
2. Peer-reviewed out-of-time forecasting studies with factor ablations.
3. Reproducible papers and maintained official libraries such as pvlib and
   PVAnalytics.
4. Relevant competitions and well-documented engineering repositories.
5. Internal operator knowledge with a named role and review state.
6. LLM-generated ideas, which always start at evidence level `K0`.

For technical claims, prefer primary sources. Record the exact URL, year,
scope, forecast horizon, modality, and what the source supports. Do not infer
0-4 hour effectiveness from a day-ahead or daily-energy result without marking
the evidence indirect.

## Competition and code rules

- Treat leaderboard solutions as engineering hypotheses, not causal evidence.
- Check public/private split behavior and likely leakage.
- Record licenses before reusing code. Prefer independent implementation of a
  published formula.
- GitHub stars are not evidence. Inspect maintainership, tests, issues, time
  semantics, and reproducibility.
- Exclude image-dependent factors from this library. Structured numerical cloud
  or satellite products may be used only when already available as time series.

## Evidence levels

- `K0`: unverified idea;
- `K1`: coherent physical/statistical mechanism;
- `K2`: authoritative or published support;
- `K3`: one valid private reproduction;
- `K4`: stable across rolling windows and seeds;
- `K5`: stable across multiple held-out temporal regimes and seasons;
- `K6`: prospective or production evidence.

Source evidence alone cannot raise a factor above `K2`.
