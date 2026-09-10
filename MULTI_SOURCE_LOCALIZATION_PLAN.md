# MULTI_SOURCE_LOCALIZATION_PLAN.md

Implementation plan for the K-source / N-receiver localization case. Written to slot into
`ARCHITECTURE.md`: same naming rules, same dependency graph, same isolation guarantee for
`src/spark/inverse/`.

**Revision 2** — reconciled against Grinstein et al., *Steered Response Power for Sound Source
Localization: A Tutorial Review* (2024). Section 7 lists what that reconciliation changed and why.

---

## 0. Band policy, decided from the LTAS

The overlay shows three facts that fix the band configuration.

**A codec brickwall at ~15.5 kHz, common to all provenances.** Everything above it is the encoder
noise floor at -60 to -70 dB, not fire.

**Per-provenance spectral shapes differing by up to 20 dB above 4 kHz.** In simulation this is
harmless — each source carries its own excerpt, and the source spectrum cancels exactly in the
per-source two-receiver level difference. It does mean band levels are never comparable *across*
sources, and it is a provenance signature for `grouped_split_manifest.py` to key on later.

**A 60-100 Hz hump varying by session.** Wind, handling, or preamp self-noise. Not fire.

### Two different band sets, for two different consumers

The delay stage and the level stage want opposite things, and conflating them is a mistake.

**SRP / delay estimation wants one wide continuous band.** GCC-PHAT peak width goes as the
reciprocal of effective bandwidth, so every hertz retained sharpens the peak and narrows the
localization ambiguity. Use **250 Hz - 11 kHz as a single passband**, not a band split.

**Level-based ranging wants the fractional-octave split.** Absorption at 20 C / 70 % RH, and what it
buys over a 100 m path:

| Band centre | alpha (dB/m) | Loss over 100 m | Range leverage dL/dr |
|---|---|---|---|
| 250 Hz | 0.0013 | 0.13 dB | negligible |
| 500 Hz | 0.0028 | 0.28 dB | negligible |
| 1 kHz | 0.0055 | 0.55 dB | 0.006 dB/m |
| 2 kHz | 0.011 | 1.1 dB | 0.011 dB/m |
| 4 kHz | 0.029 | 2.9 dB | 0.029 dB/m |
| 8 kHz | 0.098 | 9.8 dB | 0.098 dB/m |
| 10 kHz | 0.13 | 13 dB | 0.13 dB/m |
| 12.5 kHz | 0.19 | 19 dB | 0.19 dB/m |

Range information scales with alpha. At a per-band standard error of ~0.1 dB, the 8 kHz band resolves
about 1 m of range difference and the 1 kHz band about 18 m. Capping at 8 kHz would discard the
8-12.5 kHz region, worth ten to twenty times the 1 kHz band. Don't.

Let the codec set the cap instead:

- **1/3-octave** above 2 kHz. Full octaves are too wide up there — the 8 kHz octave spans
  5.6-11.3 kHz and alpha roughly doubles across it, so the energy-weighted effective alpha drifts
  downward as range grows and biases exactly the bands that matter.
- Drop any band whose upper edge `f_c * 2^(1/6)` exceeds `0.9 * measured_cutoff`. At a 15.5 kHz
  cutoff that is 13.95 kHz, so the 12.5 kHz centre (upper edge 14.03 kHz) is marginally out and
  **10 kHz is the top usable centre**.
- Keep the bottom at **250 Hz**. The low bands have no alpha leverage, but the model is
  `dL_b = -alpha_b * D - 20*log10(r2/r1)`: the second term is common to all bands, and the low bands
  where alpha is near zero are what pin it. Drop them and the geometric offset trades off against
  the absorption slope.
- Exclude below 250 Hz: the session-dependent 60-100 Hz hump carries no range information.

Final level-stage band set: **1/3-octave centres, 250 Hz to 10 kHz** (17 bands), with per-band
usability decided at run time by an SNR test rather than hardcoded. Let `inverse_variance_fusion.py`
down-weight bands that turn out noise-limited — that is what it is for.

### The PHAT weighting must be band-limited

This follows from the LTAS combined with the definition of GCC-PHAT. The phase transform normalises
every frequency bin to unit magnitude. Above the codec cliff the content is encoder noise floor at
-60 dB, and PHAT would amplify it to full weight in the cross-correlation, injecting pure noise into
the GCC with the same influence as real fire content. The stability regulariser `gamma` softens this
but does not fix it. **Compute GCC-PHAT only over 250 Hz - 11 kHz** and zero the rest before the
inverse transform. This is not optional; skipping it will visibly degrade every delay estimate.

### Why the two stages cannot be merged

The conventional SRP signal model assumes frequency-independent attenuation, and PHAT discards
magnitude entirely. SRP therefore cannot see the absorption tilt that carries your absolute range —
by construction. This is the structural reason for the two-stage design: SRP recovers geometry from
phase, band levels recover absolute range from magnitude, and they are fused at the end. Note that
PHAT-beta with `beta < 1` (Sec. 3) partially retains magnitude, which means some level information
leaks into the SRP map. Treat that as a tuning knob, not as a substitute for the level stage, and
watch for it correlating the two stages' errors during fusion.

### Verify the cutoff per file

`detect_codec_cutoff_hz` measures it. The LTAS suggests one common encoder, but a per-file cap is one
line of code.

---

## 1. Architectural decision — resolved

Revision 1 proposed extracting a shared `src/spark/propagation/` leaf so `source_deflation.py` could
project an estimated source back through the assumed channel without importing `acoustic/`.

**That is no longer needed.** The established multi-source SRP methods deflate in the *correlation*
domain, not the waveform domain: Brutti's TDOA-domain notch filter de-emphasises a located source
directly in the cross-correlation function, and the orthogonal-subspace projection variant does the
same by projecting the observed GCCs onto a subspace orthogonal to the located source position — and
that variant is reported to outperform the notch filter, particularly when sources differ in power,
which is your expected hard case.

Correlation-domain deflation needs only receiver positions, an assumed speed of sound, and the GCCs.
No propagation operator, no shared leaf, no widening of the isolation rule. `inverse/`'s forbidden
list stays exactly as it is and the guard stays non-vacuous.

The propagation leaf remains a reasonable refactor on its own merits (it would let
`exponential_attenuation_channel.py` and `free_field_propagation.py` share primitives more cleanly),
but it is now an independent tidy-up, not a prerequisite for this work. Defer it.

**One consequence for the level stage.** Per-source band levels can no longer come from subtracting
waveforms. Get them from the matched-filter gain instead: with `s_k` the beamformed estimate of
source k, the contribution of source k at receiver m in band b is the band-restricted cross-spectral
magnitude of `s_k` and `x_m` normalised by the auto-spectrum of `s_k`. This stays entirely within
`audio/` and `inverse/` and needs no forward model.

---

## 2. Configuration

### `configs/geometry.toml` — widened

```toml
domain_extent_x_m = 100.0
domain_extent_y_m = 100.0
receiver_height_m = 1.5
source_height_m = 0.5

[receivers]
layout = "ring"
count = 6
ring_radius_m = 55.0
ring_start_bearing_rad = 0.0
minimum_separation_m = 5.0
maximum_collinearity = 0.98

[[sources]]
position_xy_m = [30.0, 40.0]
amplitude_scale = 1.0

[[sources]]
position_xy_m = [70.0, 65.0]
amplitude_scale = 0.5
```

`GeometryConfiguration` currently fixes exactly two receivers. It grows a
`ReceiverLayoutConfiguration` and a list of `SourceConfiguration`. Explicit
`receiver_positions_xyz` stays supported as the `layout = "explicit"` case so every existing
two-receiver config keeps working unmodified.

### `configs/data.toml` — per-source recordings

```toml
recording_directory = "data/wildfire/"
clip_duration_s = 50.0
clip_overlap_s = 0.0
excerpt_assignment = "distinct_provenance"
maximum_permitted_excerpt_coherence = 0.15
metrics_output_path = "results/metrics/multi_source_localization.json"
```

### `configs/localization.toml` — new sections

```toml
[correlation]
lowest_frequency_hz = 250.0
highest_frequency_hz = 11000.0
phase_transform_exponent = 0.7
phase_transform_regularization = 1e-8
use_analytic_envelope = true
window_duration_s = 0.5
window_overlap = 0.5
maximum_absolute_lag_margin = 1.1
peak_interpolation = "exponential"

[bands]
fraction_denominator = 3
lowest_centre_hz = 250.0
highest_centre_hz = 10000.0
codec_cutoff_margin = 0.9
minimum_band_signal_to_noise_ratio_db = 6.0

[steered_response_power]
pooling = "volumetric"
pairwise_combinator = "product"
coarse_grid_spacing_m = 2.0
refinement_levels = 4
refinement_factor = 4
refinement_retained_fraction = 0.05
low_frequency_seed_hz = 2000.0

[deflation]
method = "subspace_projection"
maximum_source_count = 6
peak_prominence_threshold_ratio = 0.35
minimum_source_separation_m = 3.0
residual_power_reduction_threshold = 0.02

[refinement]
maximum_gauss_newton_iterations = 30
position_convergence_tolerance_m = 1e-4
use_band_level_terms = true
```

`phase_transform_exponent` is the `beta` of the parameterised phase transform: `beta = 1` is
conventional PHAT, `beta = 0` is plain cross-correlation. Reported experimental work puts the useful
range at roughly 0.65-0.7 for general signals, with `beta = 1` showing the largest performance
fluctuation. Default to 0.7 and sweep it.

`pairwise_combinator` selects how pairwise maps are merged. Conventional SRP sums, so a single pair
with a strong spurious peak can carry the global map. The product requires agreement across all
pairs; simulated work reports it cutting localization RMS error by around 45 %. Keep `sum` available
as the baseline for comparison — that comparison is itself a result worth reporting.

---

## 3. New and modified files

Ordered so each is implementable and testable before the next. Module names follow the X-SRP
decomposition (`compute_signal_features`, `create_srp_map`, `grid_search`, `update_signal_features`,
`update_grid`) so the repository mirrors a published framework and can be cross-checked against its
reference implementation.

### `src/audio/` — leaf, no domain imports

#### `codec_bandwidth_detector.py` — new

**Owns:** `detect_codec_cutoff_hz(samples, sample_rate_hz, floor_drop_db) -> float` and
`compute_long_term_average_spectrum`, shared by the Stage 0 audit and the band selector.
**Never:** Filters or modifies the signal.

#### `octave_band_filter.py` — extended

**Owns, additionally:** `compute_fractional_octave_band_edges_hz(centre_frequencies_hz, fraction_denominator)`
generalising the existing octave edges to `f_c * 2^(-+1/(2*fraction))`, and
`build_fractional_octave_centre_frequencies_hz(lowest_hz, highest_hz, fraction_denominator)` from the
ISO preferred series. Existing octave behaviour is `fraction_denominator = 1`, so existing callers
are unaffected.

#### `usable_band_selector.py` — new

**Owns:** `select_usable_band_centres_hz(candidate_centres_hz, codec_cutoff_hz, margin)` applying the
upper-edge rule, and `filter_bands_by_signal_to_noise_ratio(band_levels_db, noise_floor_levels_db, minimum_ratio_db)`
applying the run-time SNR test.
**Never:** Knows about propagation or ranges; it is handed levels.

#### `excerpt_coherence.py` — new

**Owns:** `compute_maximum_normalized_cross_correlation` and
`compute_pairwise_excerpt_coherence_matrix(excerpts) -> ndarray (k, k)`. The pre-flight guard: if two
source excerpts are mutually correlated, every cross-correlation downstream contains cross-source
artefacts indistinguishable from real sources.
**Never:** Chooses excerpts.

#### `source_excerpt_selector.py` — new

**Owns:** `select_source_excerpts(recording_paths, source_count, clip_duration_s, policy, seed) -> list[SourceExcerpt]`.
Raises if the resulting coherence matrix exceeds the configured maximum.
**Never:** Applies gain, propagation, or normalization.

#### `matched_filter_band_level.py` — new

**Owns:** `compute_matched_filter_band_levels_db(reference_signal, receiver_signals, band_edges_hz, sample_rate_hz)`.
Recovers per-source, per-receiver band levels from a mixture given a beamformed reference, using the
band-restricted cross-spectrum normalised by the reference auto-spectrum. This is what replaces
waveform subtraction as the input to the level stage.
**Never:** Knows about geometry.

---

### `src/spark/acoustic/`

#### `receiver_layout.py` — new

**Owns:** `build_ring_receiver_positions_xyz`, `build_grid_receiver_positions_xyz`,
`build_random_receiver_positions_xyz` (Poisson-disc rejection against `minimum_separation_m`), and
two guards: `compute_minimum_pairwise_separation_m` and `compute_collinearity_measure` (largest
singular-value ratio of the centred position matrix). Layout builders raise when a guard fails.
**Never:** Renders anything.

The collinearity guard is the N-receiver generalisation of the perpendicular-bisector degeneracy in
`level_ratio_triangulation.py`: with all receivers on a line, a position and its mirror across that
line produce identical delays at every receiver.

#### `free_field_propagation.py` — extended

**Owns, additionally:** `render_multi_source_receiver_signals(source_signals, source_positions_xyz, receiver_positions_xyz, atmospheric_conditions, sample_rate_hz, reference_distance_m) -> ndarray (n_receivers, n_samples)`.
Superposition is linear: loop sources, call the existing per-pair operator, accumulate on a common
clock sized by the largest delay. Amplitude scales applied here.
**Never:** Adds noise.

The existing single-source `render_receiver_signals` becomes the `n_sources = 1` case, so the
single-source script keeps working.

---

### `src/spark/inverse/` — the core work

Isolation rule unchanged and now trivially satisfied: only receiver waveforms, receiver positions,
assumed air conditions.

#### `receiver_pair_index.py` — new

**Owns:** `enumerate_receiver_pairs(receiver_count) -> ndarray (n_pairs, 2)` in canonical `i < j`
order, plus `compute_pair_baseline_distances_m`. One definition of pair ordering, so the
`tau > 0 means receiver 1 hears it later` convention cannot be applied inconsistently.
**Never:** Touches signals.

#### `time_difference_of_arrival.py` — refactored

**Owns, restructured:** the primitive becomes the curve, not the peak.

```
compute_generalized_cross_correlation(
    signal_a, signal_b, sample_rate_hz, maximum_absolute_lag_s,
    lowest_frequency_hz, highest_frequency_hz,
    phase_transform_exponent, regularization, use_analytic_envelope
) -> GeneralizedCrossCorrelationCurve

accumulate_generalized_cross_correlation_over_windows(...) -> GeneralizedCrossCorrelationCurve

compute_pair_correlation_curves(
    receiver_signals, receiver_pairs, receiver_positions_xyz, speed_of_sound_m_per_s, ...
) -> list[GeneralizedCrossCorrelationCurve]

estimate_time_difference_of_arrival(curve, interpolation) -> DelayEstimate
```

Four changes from revision 1, all from the review:

**Band-limited PHAT.** Whitening is applied only within `[lowest_frequency_hz, highest_frequency_hz]`;
bins outside are zeroed. See Sec. 0.

**Parameterised phase transform.** The denominator becomes `|X_l X_m*|^beta + gamma` rather than the
plain magnitude product. `beta = 1` recovers conventional PHAT.

**Analytic-signal envelope.** A band-passed input drives GCC-PHAT toward a sinc, whose ripples put
spurious secondary peaks into the SRP map. Taking the magnitude of the analytic signal of the GCC
removes them. Since you are band-passing at 250 Hz and 11 kHz, this applies. Keep it configurable:
the envelope costs some sub-sample sharpness, so measure the trade rather than assuming.

**Exponential peak interpolation** rather than parabolic. Of parabolic, exponential and Fourier
interpolation evaluated for SRP-PHAT, exponential performed best.

Windowed accumulation still matters: fire is continuous noise, so the peak sharpens with the number
of averaged windows, and 50 s at 0.5 s windows with 50 % overlap gives about 200. Coherence is still
measured after alignment.
**Never:** Knows what the delay is for.

#### `steered_response_power.py` — new, substantially revised

**Owns:**

```
build_candidate_grid_xyz(extent_x_m, extent_y_m, spacing_m, height_m) -> ndarray (n_grid, 3)

compute_pair_delay_table_s(candidate_positions_xyz, receiver_positions_xyz,
                           receiver_pairs, speed_of_sound_m_per_s) -> ndarray (n_grid, n_pairs)

compute_cell_delay_bounds_s(candidate_positions_xyz, cell_extent_m, receiver_positions_xyz,
                            receiver_pairs, speed_of_sound_m_per_s)
    -> tuple[ndarray, ndarray]                      # (n_grid, n_pairs) lower and upper

compute_steered_response_power_map(correlation_curves, delay_bounds, pooling, pairwise_combinator)
    -> ndarray (n_grid,)

extract_map_peak(steered_response_power_map, candidate_positions_xyz) -> tuple[ndarray, float]
```

**The point-sampled grid of revision 1 was wrong, and this is the most important correction in this
document.** The TDOA gradient has magnitude at most `2/c`, so a 0.5 m cell spans up to 2.9 ms of
TDOA — about 129 samples at 44.1 kHz. With roughly 10 kHz of usable bandwidth the GCC peak is about
4 samples wide. A point-sampled 0.5 m grid therefore undersamples the TDOA axis by a factor of
around 30 and will routinely miss the peak entirely: the grid returns a confident maximum somewhere
else, with no symptom that anything went wrong. Point sampling would need about 1.7 cm spacing, i.e.
a 36-million-point grid, which is not affordable per deflation round.

Two fixes, used together:

**Volumetric pooling.** Instead of evaluating the GCC at the cell-centre delay, pool it over the
delay *interval* the cell spans. Because TDOA surfaces are hyperboloidal, the extremes over a cuboid
cell lie on its boundary, and exact bounds can be found from a small fixed set of boundary points —
precomputable once per cell and array geometry. Sum, mean and max pooling are all reported in the
literature; max-pooling is reported more robust to noise. Make it configurable and measure.

**Iterative refinement.** Start coarse (2 m), keep the top `refinement_retained_fraction` of cells,
subdivide by `refinement_factor`, repeat for `refinement_levels`. A useful seeding trick: SRP peak
width goes inversely with frequency, so computing the first level using only content below
`low_frequency_seed_hz` yields a deliberately smooth map whose basins are wide enough that coarse
cells cannot fall between them. Refine with the full band.

For the anechoic case — which is exactly your simulation — a branch-and-bound search with these
bounds carries a theoretical guarantee of never discarding the true maximum. That guarantee does not
hold in reverberant rooms, which is why most of the literature treats it as a special case; here it
is the normal case, so it is worth implementing rather than settling for the heuristic pruning.

**Pairwise combination.** Conventional SRP sums the pairwise maps, which means any single pair with a
strong spurious peak can dominate the global map. The product requires all pairs to agree; reported
simulations put the gain at roughly 45 % lower localization RMS error. Geometric and harmonic means
are reported to suppress sidelobes and improve source level estimation. Implement `sum`, `product`
and `harmonic_mean` behind the config switch.

Time-domain formulation is retained rather than frequency-domain: it is `L` times cheaper, and its
known weakness — integer-delay rounding — is a compact-array problem. Your array spans 100 m, so the
TDOA range is thousands of samples rather than a handful, and interpolation covers the remainder.
**Never:** Assumes a source count.

#### `multilateration.py` — new

**Owns:**

```
compute_predicted_time_differences_s(position_xyz, receiver_positions_xyz, receiver_pairs, speed_of_sound_m_per_s)
compute_time_difference_jacobian(...)
refine_position_gauss_newton(...) -> PositionEstimate
```

Weighted by inverse delay variance, using the variance `time_difference_of_arrival.py` already
derives from effective bandwidth and coherence. Covariance from the inverse weighted normal matrix at
the solution, restricted to the `(x, y)` block. The N-receiver analogue of
`level_ratio_triangulation.py`, producing the same error-ellipse output.

Two-step TDOA triangulation is known to be less robust than SRP under noise and reverberation, which
is why SRP seeds it here rather than the other way round. The residual `chi_squared_reduced` is the
validity test — but note the degrees of freedom: the cocycle constraint means N receivers give only
N-1 independent TDOAs against 2 unknowns, so there are N-3 degrees of freedom. At N = 3 the fit is
exact by construction and the residual is uninformative; **N >= 5 is where the residual gains real
discriminating power**, which is a concrete argument for pushing past the minimum array size.

Minimum geometry: N >= 3. With exactly three, the two hyperbola branches can intersect twice and a
two-fold ambiguity survives. Report N = 3 as the ambiguous baseline rather than hiding it.
**Never:** Touches signals.

#### `source_deflation.py` — new, revised

**Owns:**

```
apply_tdoa_notch_deflation(correlation_curves, source_position_xyz, receiver_positions_xyz,
                           receiver_pairs, speed_of_sound_m_per_s, notch_width_s)
    -> list[GeneralizedCrossCorrelationCurve]

apply_subspace_projection_deflation(correlation_curves, source_position_xyz, ...)
    -> list[GeneralizedCrossCorrelationCurve]

compute_delay_and_sum_beamformed_signal(receiver_signals, receiver_positions_xyz,
                                        source_position_xyz, sample_rate_hz, speed_of_sound_m_per_s)
    -> ndarray (n_samples,)

compute_residual_correlation_energy_ratio(original_curves, deflated_curves) -> float
```

Deflation acts on the GCCs, not the waveforms. The notch variant applies a TDOA-domain notch at each
pair's predicted delay for the located source; the subspace variant projects the observed GCCs onto a
subspace orthogonal to that source's position and is reported to outperform the notch, especially
when sources differ in power. Implement the notch first (simpler, well documented) and the projection
second; make the choice a config switch so the comparison is a measurable result.

`compute_delay_and_sum_beamformed_signal` remains, but only as the reference signal for the level
stage — it is no longer part of deflation.

**Documented failure mode, inherited:** the originators of the notch approach state that with three
sources the noise in the correlation function becomes prohibitive and recommend tracking instead.
Take that as a real ceiling on sequential deflation, not as pessimism. Sec. 5 gives the alternative.
**Never:** Reads ground truth or the forward model's per-source signals.

#### `multiple_source_estimator.py` — stub filled

**Owns:** `MultipleSourceEstimator` and the `localize_multiple_sources` driver, structured as the
X-SRP loop:

```
estimates = empty
grid = create_initial_candidate_grid(domain)
curves = compute_signal_features(receiver_signals)
while grid is not empty:
    srp_map = create_srp_map(receiver_positions, grid, curves)
    estimates, grid = grid_search(grid, srp_map, estimates)
    curves = update_signal_features(curves, estimates, receiver_positions)
    grid = update_grid(estimates, domain)
return estimates
```

`update_signal_features` is the deflation step and `update_grid` is the refinement step, so a single
loop covers both the coarse-to-fine search and the multi-source peeling. Stop when the peak fails
`peak_prominence_threshold_ratio` against the map median, when it lies within
`minimum_source_separation_m` of an accepted source, when `maximum_source_count` is reached, or when
the residual correlation energy stops dropping.

Returns a list of `PositionEstimate`, so `K_hat` is an output. Calibrate the prominence threshold on
a source-free null run rather than by eye.
**Never:** Requires the true source count.

#### `windowed_position_clustering.py` — new, second route to K

**Owns:** `estimate_positions_by_windowed_clustering(receiver_signals, ..., window_duration_s, clustering)`.
Runs single-source SRP independently on many short windows, collects the per-window maxima, and
clusters them spatially; each dense cluster is a source, cluster count is `K_hat`, cluster centroid
is the position and cluster spread is an empirical uncertainty.

This exists because it suits fire specifically. Much of the multi-source SRP literature leans on
speech sparsity — W-disjoint orthogonality, per-bin source dominance, voice activity detection — none
of which transfers to continuous, stationary, spectrally similar broadband noise. What does transfer
is impulsivity: fire crackle is a sequence of transients, so within a short window one source's
crackle frequently dominates, which is exactly the condition per-window clustering needs. Independent
of deflation, so its `K_hat` is a genuine cross-check rather than a restatement.
**Never:** Assumes windows are independent of each other in the fusion step.

#### `joint_position_refinement.py` — new

**Owns:** `refine_positions_with_band_levels(position_estimates, receiver_signals, receiver_positions_xyz, band_centres_hz, absorption_coefficients_db_per_m, ...)`.
Beamform toward each estimate, recover per-source per-receiver band levels via
`matched_filter_band_level.py`, then apply `band_level_difference.py` per source unchanged. Minimise
a single cost with TDOA residual terms and band-level residual terms, each weighted by its own
variance, fused through `inverse_variance_fusion.py`.

This is where the two observables earn their keep together. Delays constrain range *differences* very
precisely — your measured sigma_D of about 0.014 m is consistent with sub-sample GCC-PHAT — but a
compact array constrains absolute range poorly, so TDOA-only ellipses stretch radially. The
absorption slope supplies exactly that direction. If `phase_transform_exponent < 1`, check whether
the two stages' errors are correlated before treating the fusion weights as independent.
**Never:** Recomputes alpha — it is handed the coefficients, keeping `atmosphere/` the single source.

---

### `src/utils/metrics/` — new package

#### `localization_metrics.py` — new

**Owns:** `match_estimated_to_true_sources` (Hungarian assignment), 
`compute_optimal_subpattern_assignment_distance` (OSPA, configurable cutoff and order), per-source
position error, detection and false-alarm counts.

OSPA rather than plain RMSE, because RMSE over matched pairs silently ignores missed and spurious
sources — the failure mode the sweep exists to expose.
**Never:** Runs an estimator.

---

### `src/utils/visualization/`

#### `steered_response_power_plotter.py` — new

**Owns:** SRP map as a heat map over the domain, true and estimated positions, error ellipses,
receiver positions. One figure per deflation round and per refinement level, so both the peeling and
the coarse-to-fine search are inspectable.

#### `localization_error_plotter.py` — new

**Owns:** sweep curves — OSPA against source count, receiver count, SNR, inter-source separation,
receiver layout, `beta`, and pairwise combinator.

#### `receiver_signal_plotter.py` — extended

**Owns, additionally:** per-pair GCC curves with true delays marked, with and without the analytic
envelope; band-level difference regression against alpha_b per source.

---

### `scripts/`

#### `run_source_excerpt_audit.py` — new

Stage 0, and the formalisation of the LTAS figure already produced. Reads the recording pool, plots
the LTAS overlay coloured by provenance, detects each file's codec cutoff, selects K excerpts,
computes the mutual coherence matrix, and **fails loudly** if any off-diagonal entry exceeds the
maximum. Writes the resolved band lists (wide for correlation, fractional-octave for levels).

Run this before anything else.

#### `run_multi_source_localization.py` — new

Loads `configs/` -> selects excerpts -> builds the receiver layout -> renders the multi-source field
-> adds receiver noise -> runs `localize_multiple_sources` -> optional windowed-clustering
cross-check -> optional joint refinement -> matches against ground truth -> writes metrics and
figures. Takes `--configs <dir>` and nothing else.

#### `run_localization_sweep.py` — new

Loops the above over the parameter grid, one metrics record per cell.

---

## 4. Tests

| Test file | What it asserts |
|---|---|
| `audio/test_codec_bandwidth_detector.py` | Synthetic brickwalled noise: cutoff recovered within one FFT bin |
| `audio/test_octave_band_filter.py` (extended) | `fraction_denominator = 1` reproduces existing edges exactly; `= 3` matches ISO third-octave edges |
| `audio/test_usable_band_selector.py` | 15.5 kHz cutoff at 0.9 margin admits 10 kHz, rejects 12.5 kHz |
| `audio/test_excerpt_coherence.py` | Identical excerpts score 1.0; independent noise near zero |
| `audio/test_source_excerpt_selector.py` | Distinct-provenance policy never reuses a file; raises above the coherence limit |
| `audio/test_matched_filter_band_level.py` | Two synthetic sources at known gains: recovered levels within 0.5 dB |
| `spark/acoustic/test_receiver_layout.py` | Ring/grid/random honour separation; collinear layout raises |
| `spark/acoustic/test_free_field_propagation.py` (extended) | `n_sources = 1` matches the previous path; two sources equal the sum of two single renders |
| `spark/inverse/test_receiver_pair_index.py` | Canonical ordering, pair count n(n-1)/2 |
| `spark/inverse/test_time_difference_of_arrival.py` (extended) | Curve peak equals the previous scalar estimate at `beta = 1`; band-limited whitening ignores out-of-band noise; envelope removes sinc ripples on a band-passed input; sign convention still pinned |
| `spark/inverse/test_steered_response_power.py` | **Volumetric pooling recovers a source that point sampling at the same spacing misses** — the regression test for the revision-2 correction; delay bounds contain the true cell delay range; product and sum combinators both peak at truth noiselessly |
| `spark/inverse/test_multilateration.py` | Noiseless recovery to 1e-6 m; covariance grows with delay variance; N = 3 ambiguity flagged; chi-squared uninformative at N = 3 and informative at N = 5 |
| `spark/inverse/test_source_deflation.py` | Deflating one synthetic source removes its peak from every pair curve; deflating twice is idempotent; subspace variant outperforms notch at a 10 dB power ratio |
| `spark/inverse/test_windowed_position_clustering.py` | Two synthetic impulsive sources recovered as two clusters |
| `spark/inverse/test_multiple_source_estimator.py` | K = 2 and K = 3 noiseless recovery; `K_hat` correct; graceful SNR degradation |
| `spark/inverse/test_joint_position_refinement.py` | Adding level terms shrinks the radial semi-axis of the ellipse |
| `utils/metrics/test_localization_metrics.py` | Hungarian matching on a permuted set; OSPA penalises cardinality errors |
| `test_inverse_does_not_import_forward_model.py` | **Unchanged.** Correlation-domain deflation means no new import path was needed |

---

## 5. Milestones and acceptance criteria

**M0 — audit.** Acceptance: every off-diagonal coherence below 0.15, cutoff detected per file, both
band lists written.

**M1 — forward.** Acceptance: two-source render equals the sum of two single renders to machine
precision; the single-source script's existing metrics reproduce unchanged.

**M2 — SRP with K known.** Acceptance: K = 2, N = 4, noiseless, both positions within one final-level
grid spacing of truth — **and** the point-sampling-versus-volumetric regression test passes, so the
correction of Sec. 3 is demonstrably load-bearing rather than assumed.

**M3 — deflation with K unknown.** Acceptance: `K_hat = K` for K in {1, 2, 3} at 20 dB SNR with
sources at least 10 m apart; OSPA below 1 m. Cross-check `K_hat` against windowed clustering; a
disagreement is a finding, not a bug to suppress.

**M4 — level fusion.** Acceptance: median radial ellipse semi-axis shrinks by a factor of two or more
versus M3.

**M5 — sweep.** K in {1..4}, N in {3..8}, SNR in {0, 10, 20, 30} dB, three layouts, source separation
2 to 40 m, `beta` in {0.65, 0.7, 1.0}, combinator in {sum, product}. Deliverable: the two-source
resolution limit as a function of separation and N.

**If M3 fails at K = 3**, which the deflation literature predicts, the fallback is joint rather than
sequential estimation: fit the whole SRP map at once with a group-sparse generative model over the
candidate grid, so all sources are estimated simultaneously and the sequential error compounding
disappears. This has been demonstrated for multi-source SRP map fitting; its documented weakness is
hyperparameter sensitivity. Budget it as a contingency, not as the main path.

---

## 6. Numerical notes

**Grid sampling — the binding constraint.** TDOA gradient magnitude is at most `2/c = 5.83 ms/m`. A
cell of side `d` therefore spans up to `2d/c` of TDOA: 2.9 ms, or 129 samples at 44.1 kHz, for
`d = 0.5 m`. The GCC main lobe with 10 kHz of bandwidth is about 0.1 ms, or 4 samples. Point sampling
requires `d <= c/(2B) ~ 1.7 cm`. This is why Sec. 3 pools over cell delay bounds instead.

**Lag resolution.** One sample at 44.1 kHz is 22.7 us, i.e. 7.8 mm of range difference. Sub-sample
interpolation takes that below a centimetre noiselessly. So the final refinement grid can stay far
coarser than the delay precision — it only has to land in the correct basin for Gauss-Newton.

**Delay table size.** At the coarse level (2 m spacing, 2 500 cells) the bounds tables are two
`(2500, n_pairs)` float64 arrays — trivial. Rebuild per refinement level, not per deflation round;
the *curves* change under deflation, the geometry does not.

**Height handling.** Keep positions in 3D (receivers at 1.5 m, sources at 0.5 m) but solve for
`(x, y)` only. The height difference is free to include in the forward distance computation.

**Speed of sound.** From `atmosphere/atmospheric_conditions.py`, from the same `AtmosphericConditions`
the renderer used. For a sensitivity study, perturb the *estimator's* assumed conditions while leaving
the renderer's fixed — the honest version of the experiment, already supported since the estimator's
air conditions arrive as an argument.

**Outdoor caveats, for when this leaves simulation.** The literature on outdoor SRP flags three
problems your simulation does not have: intense low-frequency environmental noise forcing a
band-pass (hence the envelope fix above), search areas large enough to make grid cost binding, and
temperature, wind, terrain and sensor-position variation making the constant-`c` propagation model
unreliable. The third is the one that will hurt most in a field trial and has no simulation analogue
— worth naming explicitly as a limitation rather than discovering later. Distributed arrays also
require sample-clock synchronisation across devices, which the simulation grants for free.

**Where it will fail first.** Two sources closer than roughly the SRP main-lobe width merge into one
peak and the loop returns `K_hat = 1`. There is an analytical model in the literature relating signal
bandwidth to spatial resolution in SRP-PHAT maps — use it to predict the limit and let M5 confirm it,
rather than mapping it blind. Expect the amplitude ratio to matter as much as separation: every
multi-source SRP method surveyed degrades when sources differ greatly in power, and a source 10 dB
below its neighbour will likely be missed well before the geometric resolution limit is reached.

---

## 7. What the tutorial review changed, and what it confirmed

### Confirmed

- **SRP-PHAT plus source cancellation is the established multi-source approach**, not an improvisation.
- **Avoiding explicit peak-to-source association is the correct instinct**, and has a name: the
  principle of least commitment. Estimating delays first and discarding the rest of the correlation
  function is what makes two-step methods fragile.
- **Near-field, exact-range-difference formulation is required**, since a distributed array with
  sources at comparable distances is positional localization, not DOA estimation, and no plane-wave
  approximation is valid.
- **`K` must be estimated, not assumed** — the review states this outright for the multi-source case.
- **Power disparity between sources is the documented failure mode** across the whole family, which
  matches the earlier warning about a 10 dB weaker source being missed first.

### Changed

| Revision 1 | Revision 2 | Why |
|---|---|---|
| Point-sampled 0.5 m search grid | Volumetric pooling over cell delay bounds, plus iterative refinement and a low-frequency seed level | A 0.5 m cell spans ~129 samples of TDOA against a ~4-sample peak; the grid would miss sources silently |
| Deflation by beamform-and-subtract on waveforms | Deflation on the GCCs (TDOA notch, then subspace projection) | This is what the field does, it handles power disparity better, and it removes the need for a shared propagation leaf entirely |
| New `src/spark/propagation/` leaf required | Deferred as an optional tidy-up | No longer needed once deflation is correlation-domain; isolation rule holds unchanged |
| Conventional PHAT | Parameterised PHAT-beta, default 0.7 | `beta = 1` shows the largest performance fluctuation experimentally |
| PHAT over the full spectrum | Whitening band-limited to 250 Hz - 11 kHz | Whitening normalises the codec noise floor above 15.5 kHz to full weight |
| Sum over pairwise maps | Configurable sum / product / harmonic mean, default product | Product reported to cut localization RMS error by ~45 % |
| Parabolic sub-sample interpolation | Exponential | Reported best of parabolic, exponential and Fourier for SRP-PHAT |
| No treatment of band-pass artefacts | Analytic-signal envelope option on the GCC | Band-passed input drives GCC toward a sinc with ripples that become spurious map peaks |
| Deflation as the only multi-source route | Windowed clustering added as an independent second route to `K_hat`; group-sparse joint fitting as the K >= 3 contingency | Sequential deflation is documented as breaking down at three sources |
| Resolution limit to be mapped empirically | Analytical bandwidth-resolution model as the prediction, M5 as confirmation | The model exists; blind mapping wastes the sweep |
| Ad-hoc module decomposition | Aligned to the X-SRP block structure | Mirrors a published framework with an open reference implementation to check against |

### Noted as not transferable

A large part of the multi-source SRP literature depends on speech-specific structure: W-disjoint
orthogonality, one-source-dominant time-frequency bins, voice activity detection, speaker
verification. Fire is continuous, stationary, broadband and spectrally similar across sources, so
none of that carries over. The one speech-like property fire does have is impulsivity in the crackle
band, which is what `windowed_position_clustering.py` exploits. Worth stating explicitly in the
paper: the multi-source methods that work here are the ones that do not assume sparsity.
