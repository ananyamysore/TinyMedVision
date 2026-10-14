# Evaluation design

## Predictive metrics

Binary ROC-AUC and average precision use predicted probabilities. Sensitivity,
specificity, precision, negative predictive value, and confusion counts use a
threshold; the example experiment prespecifies 0.5 and does not tune it on the
held-out test set. Metrics requiring absent classes are represented as `null`
(Python `None`) instead of being silently invented. For imbalanced datasets,
report prevalence and average precision alongside ROC-AUC.

The optional ROC-AUC interval is a percentile bootstrap. If group IDs are
available, complete groups are resampled; otherwise images are sampled, which
assumes image independence and may understate uncertainty. The CI is descriptive,
not a substitute for a prespecified statistical analysis plan. The implementation
does not yet calculate intervals for every metric.

## Calibration

Brier score is mean squared probability error for the binary outcome. Expected
calibration error is the sample-weighted absolute difference between mean
predicted probability and observed frequency in equal-width bins. ECE depends on
binning and sample size and is not a proper scoring rule; inspect the reliability
curve and Brier score together. Calibration must be fitted/assessed on data
separate from model fitting if used to modify predictions. No post-hoc calibrator
is fitted automatically.

## Robustness

Corruptions apply the same deterministic operation and seed across models. They
are controlled image-quality stress tests, not real clinical distribution shifts.
The `compression` setting is pixel quantization and is explicitly not JPEG.
No external dataset is bundled and no external-validation claim is made.
