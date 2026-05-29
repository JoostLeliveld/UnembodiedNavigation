# F19 Traversability Keep-In Landscape

- figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F19_traversability_keepin_landscape.png`
- samples: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F19_traversability_keepin_landscape.csv`

## Sample values

- start: (3.200, -1.000), signed_distance=0.475 m, penalty=2.02, violation=False
- goal: (1.000, 1.750), signed_distance=0.375 m, penalty=2.02, violation=False
- A3 center lower: (1.075, -1.000), signed_distance=0.450 m, penalty=2.02, violation=False
- A4 center lower: (3.125, -1.000), signed_distance=0.550 m, penalty=2.02, violation=False
- rack body gap x2 y0: (2.000, 0.000), signed_distance=-0.475 m, penalty=341.13, violation=True
- between A3/A4 y0: (2.100, 0.000), signed_distance=-0.475 m, penalty=341.13, violation=True
- box spot R4: (2.000, -1.250), signed_distance=-0.475 m, penalty=341.13, violation=True
- mid cross goal lane: (1.000, 1.750), signed_distance=0.375 m, penalty=2.02, violation=False

Interpretation: the current no-go layer is a soft keep-in penalty over the driveable-region union. It is not an obstacle mesh cost, but because it is finite, a trajectory can still trade brief non-driveable excursions against goal/risk/ambiguity unless invalid candidates are rejected or the penalty is made effectively hard.