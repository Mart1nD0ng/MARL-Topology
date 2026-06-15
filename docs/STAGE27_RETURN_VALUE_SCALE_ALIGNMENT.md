# Stage 27 Return / Value Scale Alignment

The normalizer is fitted on train returns only.

- raw mean: `-449.6699541247348`
- raw std: `323.47408039953655`
- normalized mean: `-1.1220408283052485e-15`
- normalized std: `0.9999999999690855`
- fitted on train only: `True`
- GAE policy: critic predicts normalized value, then denormalizes to raw return scale before GAE.
