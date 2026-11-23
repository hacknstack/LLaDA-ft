All four combinations of threshold/low-confidence remasking and Monte Carlo/path
sampling accept finite, strictly positive temperatures with full-vocabulary decoding.
Zero is not a supported greedy limit.

For each active position i, candidate tokens follow p_i(v) = softmax(logits_i / T).
Confidence is c_i(v) = softmax(logits_i), without temperature scaling. This convention
is shared by all four implementations, including the CDF masses used by path sampling.
Low-confidence reveals the candidate with highest c_i(v); ties select the smallest
sequence index. Threshold reveals every candidate with c_i(v) >= confidence_threshold,
falling back to the same highest-confidence rule when no candidate qualifies.
Changing T changes the candidates, and therefore can change the reveal order or subset,
even though the confidence function itself stays untempered.

Within either decoder, both estimators target the same temperature-dependent success
probability. Monte Carlo averages success indicators from ordinary decoder trajectories.
Path sampling samples successful transitions and averages their probability weights.
Individual draws and finite-sample estimates need not agree. Temperature changes the
decoder distribution being measured; these are not estimates of its T=1 probability.
The two remasking schemes define different decoders, so their probabilities can differ.

check_temperature_consistency.py independently enumerates every candidate combination
for a state-dependent, two-mask, three-token toy model, including confidence ties. It
compares exact DP and both estimators at T=0.5, 1, 2, using threshold cutoffs 0.5 and 0.9.
Each case uses 4,000 successful-path samples and 80,000 ordinary MC draws. Exact DP
must match enumeration to 1e-10 relative tolerance; estimates must fall within seven
standard errors. Invalid temperatures 0, -1, infinity, and NaN must still be rejected.
Audit JSON files record results for both source folders. This is a CPU algorithm check,
not a new full-model GPU experiment. Very extreme temperatures can still encounter
finite-precision limitations despite being mathematically positive and finite.

Only validation and diagnostic gates were changed for this update. Estimator arithmetic,
the completed T=1 experiment, its exported CSVs, and archived code snapshots are unchanged.
