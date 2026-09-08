# Acoustic Localization of a Fire — Single Source, Two Receivers

**Scope of this document.** Full, implementable methodology for locating **one** fire source
from **two** microphones on a 100 × 100 m domain (0.5 m mesh). This is the minimal case
(2 mics is the floor for 2-D range localization) and the base unit that later generalizes to
multiple sources. Everything here is the "fuse-before-invert" architecture: combine the
per-band evidence into one geometric term, then invert once.

---

## 0. Inputs and outputs

**Inputs the algorithm needs**
| Symbol | Meaning | Source |
|---|---|---|
| `y1, y2` | mono recordings at the two mics, same clock, same fs | your split 5 s clips |
| `fs` | sample rate (Hz) | from the audio |
| `s1, s2` | receiver positions (x, y) in metres, in grid frame | you provide |
| `T_air, RH, P` | air temperature (°C), relative humidity (%), pressure (kPa) | your simulation scenario |
| `c` | speed of sound (m/s), from `T_air` | computed, see §1 |

**Input used only for evaluation (not by the estimator)**
| `x_true` | true source position | you provide → used to report error, never fed to the solver |

**Outputs**
- `x_hat = (x, y)` — estimated source position (m)
- `Sigma_xy` — 2 × 2 covariance → error ellipse
- Diagnostics: per-band `G_b`, weighted `G_hat`, reduced chi-square `chi2_nu`, per-band residuals,
  TDOA `tau`, distances `r1, r2`.

The two-fold mirror ambiguity across the mic baseline is resolved by knowing which side of the
line `s1–s2` the domain lies on (place both mics on one edge → the mirror image falls outside
the 100 × 100 m grid).

---

## 1. Fixed physical constants and derived quantities

```python
c = 20.05 * np.sqrt(273.15 + T_air)     # speed of sound (m/s); ~340 at 15 C
```

Absorption coefficient per band from **ISO 9613-1** (validated, see §2 for values):

```python
def iso9613_alpha(f, T_celsius, RH_percent, P_kpa=101.325):
    """Atmospheric absorption (dB/m). f in Hz (scalar or array)."""
    f = np.asarray(f, float)
    T   = T_celsius + 273.15
    T0, T01, pr = 293.15, 273.16, 101.325
    pa  = P_kpa
    psat_pr = 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h   = RH_percent * psat_pr * (pr / pa)                      # % molar water vapour
    frO = (pa/pr) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pa/pr) * (T/T0)**-0.5 * (9 + 280*h*np.exp(-4.170*((T/T0)**(-1/3) - 1)))
    return 8.686 * f**2 * (
        1.84e-11 * (pr/pa) * (T/T0)**0.5
        + (T/T0)**-2.5 * (0.01275*np.exp(-2239.1/T)/(frO + f**2/frO)
                        + 0.1068*np.exp(-3352.0/T)/(frN + f**2/frN)))
```

Compute `alpha_b` **once per scenario** from your simulation's `T_air, RH, P` — do not use a
generic table, because `alpha` moves ~30 % across realistic conditions (see §2).

---

## 2. Band selection and alpha values

**Choice: 7 octave bands, 125 Hz → 8 kHz.** Rationale:
- Standard ISO centres → reproducible, tabulated, easy to filter.
- Covers your "100 Hz–8 kHz" span (125 Hz octave lower edge ≈ 88 Hz).
- Low bands (125–250 Hz) are near-zero-absorption **anchors** for the geometric term;
  high bands (4–8 kHz) carry the absorption leverage. The spread is what makes the model
  well-conditioned.

Do **not** go finer (1/3-octave, 21 bands) for the first implementation: each narrow band is
noisier and you gain nothing here because `D` is fixed from TDOA, not fitted from the bands.

**Center frequencies:** `125, 250, 500, 1000, 2000, 4000, 8000` Hz.

**alpha_b (dB/m), computed with the function above** — pick the row nearest your scenario, or
recompute exactly:

| band (Hz) | 15 °C / 70 % | 20 °C / 70 % | 10 °C / 80 % | 25 °C / 50 % |
|---|---|---|---|---|
| 125  | 0.000376 | 0.000335 | 0.000373 | 0.000394 |
| 250  | 0.001124 | 0.001124 | 0.001018 | 0.001313 |
| 500  | 0.002358 | 0.002791 | 0.001963 | 0.003223 |
| 1000 | 0.004079 | 0.004978 | 0.003566 | 0.005677 |
| 2000 | 0.008777 | 0.009039 | 0.008789 | 0.010204 |
| 4000 | 0.026608 | 0.023086 | 0.028966 | 0.025863 |
| 8000 | 0.094962 | 0.077633 | 0.104565 | 0.086573 |

**Scale sanity check (why absorption matters despite being the smaller term).** On this grid a
typical path difference is `D = r2 − r1 ≈ 20–40 m`. At `D = 30 m`, the absorption term
`alpha_b · D` is ~0.01 dB at 125 Hz (ignorable) but ~**2.8 dB** at 8 kHz — comparable to the
geometric term `G` itself. Omitting it biases exactly your highest-SNR crackle bands. Keep it.

**Ground-effect flag.** The 250–500 Hz bands sit where ground-reflection interference lives.
For this first version (free-field simulation, no impedance model) they are fine. The moment
your forward model includes ground, treat 250–500 Hz as suspect — the §6 residual check will
reveal it as a structured dip rather than random scatter.

---

## 3. Window length T and number of windows M

**Analysis window `T` (default 0.5 s, easily changed).**
RMS level converges when the time-bandwidth product `B·T ≫ 1`. The narrowest band (125 Hz
octave, `B ≈ 88 Hz`) gives `B·T = 44` at `T = 0.5 s` — comfortable. `T = 0.25 s` (`B·T = 22`)
is the aggressive floor; below that the low-band level estimate gets noisy. Expose `T` as a
parameter; **0.5 s** is the recommended default.

**Windowing within a clip:** Hann window, **50 % overlap** (`hop = T/2`).
```
N_windows = floor((L_clip − T) / hop) + 1
```

**How many windows `M` for a trustworthy variance.**
The relative error on an estimated standard deviation is `≈ 1 / sqrt(2 (M_eff − 1))`.
50 %-overlap windows are ~half-independent, so `M_eff ≈ 0.6 · N_windows`.

| target | rel. error on σ_b | need M_eff | plan |
|---|---|---|---|
| adequate | ~20 % | ~13 | **default** |
| good | ~16 % | ~19 | preferred |
| tight | ~14 % | ~25 | if variance looks unstable |

**Recommended defaults (map directly onto your 5 s clips):**

| clip length | T | hop | N_windows | M_eff | verdict |
|---|---|---|---|---|---|
| **5 s (one clip)** | 0.5 s | 0.25 s | **19** | ~11–13 | **default — use this** |
| 10 s (two clips joined) | 0.5 s | 0.25 s | 39 | ~23 | if σ_b unstable |
| 2 s (fast-moving fire) | 0.5 s | 0.25 s | 7 | ~4 | marginal, responsive |

**Fire-movement budget.** Surface-fire spread is ~0.01–0.3 m/s; over 5 s the front moves
≤ 1.5 m, over 10 s ≤ 3 m — below the few-metre error floor of the method. So a **single 5 s clip
is the default**; only shorten (toward 2 s) for fast crown-fire runs where the source genuinely
translates during the observation. **Start at 5 s / T = 0.5 s / M ≈ 19.**

---

## 4. Per-window, per-band computation

### 4.1 TDOA — once, broadband (GCC-PHAT)

```python
def gcc_phat(a, b, fs, max_tau=None, interp=16):
    n = a.size + b.size
    A = np.fft.rfft(a, n); B = np.fft.rfft(b, n)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12                     # PHAT weighting
    cc = np.fft.irfft(R, n * interp)
    max_shift = int(interp * n / 2)
    if max_tau: max_shift = min(int(interp*fs*max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    k = np.argmax(np.abs(cc))
    # parabolic sub-sample refinement
    if 0 < k < cc.size-1:
        y0,y1,y2 = np.abs(cc[k-1]),np.abs(cc[k]),np.abs(cc[k+1])
        k = k + 0.5*(y0-y2)/(y0-2*y1+y2)
    return (k - max_shift) / float(interp * fs)   # tau12 (s): >0 => arrives at mic1 later
```

`max_tau = B / c` (baseline / speed of sound) bounds the search to physically possible delays.

From the TDOA:
```
tau = gcc_phat(y1, y2, fs, max_tau=B/c)   # over the whole clip (or averaged over windows)
D   = -c * tau                            # path difference r2 - r1  (metres), FIXED hereafter
```

### 4.2 Band-pass filter (both channels)

4th-order Butterworth, **zero-phase** (`filtfilt`) so no extra delay is introduced:

```python
from scipy.signal import butter, filtfilt
def octave_bandpass(x, fc, fs, order=4):
    lo, hi = fc/np.sqrt(2), fc*np.sqrt(2)
    hi = min(hi, 0.99*fs/2)
    b,a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, x)
```

### 4.3 Time-align, then level per band

Shift channel 2 by the TDOA so both windows cover the same emission event, then RMS:

```python
def band_level_db(xb, w):                 # xb: band signal in window, w: Hann window
    p2 = np.mean((xb*w)**2) / np.mean(w**2)
    return 10*np.log10(p2 + 1e-20)         # p_ref folds out in the difference
```

For each window `m` and band `b`:
```
L1_bm = band_level_db(y1_b[window m],             w)
L2_bm = band_level_db(y2_b[window m + tau shift], w)
dL_bm = L1_bm - L2_bm
```

### 4.4 Per-window geometric estimate

```
G_bm = dL_bm - alpha_b * D          # each (band, window) gives one estimate of the same G
```

---

## 5. Fusion → single position

### 5.1 Per-band average and variance (over the M windows)

```
G_b       = mean_m ( G_bm )
sigma2_b  = var_m  ( G_bm )          # sample variance, ddof=1  -> this IS sigma_b^2
```
(Optionally floor `sigma2_b` using band coherence: `sigma2_b *= (1 + (1-gamma2_b)/max(gamma2_b,eps))`
to auto-down-weight low-coherence bands. Not required for the first version.)

### 5.2 Inverse-variance fusion of G

```
w_b     = 1 / sigma2_b
G_hat   = sum_b (w_b * G_b) / sum_b (w_b)
var_G   = 1 / sum_b (w_b)
```

### 5.3 Consistency check (do this **before** trusting the position)

```
chi2_nu = sum_b (G_b - G_hat)**2 / sigma2_b / (Nb - 1)
```
- `chi2_nu ≈ 1` → bands agree, free-field + ISO model adequate → trust the estimate.
- `chi2_nu ≫ 1` → inspect per-band residuals `G_b − G_hat`:
  - spike at 250–500 Hz → ground effect,
  - monotone rise with frequency → wrong `alpha` (bad T/RH) or forgotten absorption term,
  - one erratic band → filter/SNR problem in that band.

### 5.4 Invert to distances, then position

```python
k  = 10**(G_hat/20.0)
r1 = D / (k - 1.0)
r2 = k * D / (k - 1.0)

# place s1 at origin, s2 at (B,0) in a local frame aligned with the baseline:
B  = np.linalg.norm(s2 - s1)
x_loc = (r1**2 - r2**2 + B**2) / (2*B)
y_loc = np.sqrt(max(r1**2 - x_loc**2, 0.0))     # sign: + = domain side of baseline
# then rotate/translate (x_loc, y_loc) back into the grid frame using s1, s2.
```

**Near-singular guard:** if `|k − 1| < 0.05` the source is near the perpendicular bisector
(`D → 0`), where range is ill-determined — flag the estimate as low-confidence rather than
trusting the blown-up `r1, r2`.

### 5.5 Uncertainty on the final position

Two independent inputs: `var_G` (from §5.2) and `var_D` (TDOA precision,
`sigma_tau ≈ 1/(2π·B_eff·sqrt(SNR))`, so `var_D = c^2 · sigma_tau^2`). Push both through the
Jacobian of `(G, D) → (x, y)` — finite-difference it, it's a 2 × 2:

```python
def forward(G, D, s1, s2):
    k=10**(G/20); r1=D/(k-1); r2=k*D/(k-1)
    B=np.linalg.norm(s2-s1); xl=(r1**2-r2**2+B**2)/(2*B); yl=np.sqrt(max(r1**2-xl**2,0))
    # ... rotate/translate to grid frame, return (x,y)
    return np.array([x, y])

eps=1e-4
J = np.column_stack([(forward(G_hat+eps,D,s1,s2)-forward(G_hat-eps,D,s1,s2))/(2*eps),
                     (forward(G_hat,D+eps,s1,s2)-forward(G_hat,D-eps,s1,s2))/(2*eps)])
Sigma_xy = J @ np.diag([var_G, var_D]) @ J.T
```

`Sigma_xy` is your error ellipse — overlay it on the grid. Expect a thin ellipse across the
hyperbola, long along it; it widens as `r1·r2/D` grows (sources far from the baseline are worse).

---

## 6. End-to-end driver

```python
def localize(y1, y2, fs, s1, s2, T_air, RH, P,
             bands=(125,250,500,1000,2000,4000,8000),
             T_win=0.5, overlap=0.5):
    c      = 20.05*np.sqrt(273.15+T_air)
    alpha  = iso9613_alpha(np.array(bands), T_air, RH, P)      # dB/m per band
    B      = np.linalg.norm(np.subtract(s2,s1))
    tau    = gcc_phat(y1, y2, fs, max_tau=B/c)
    D      = -c*tau
    nwin   = int(T_win*fs); hop = int(nwin*(1-overlap)); w = np.hanning(nwin)
    shift  = int(round(tau*fs))
    G_b, s2_b = [], []
    for fc,ab in zip(bands, alpha):
        y1b, y2b = octave_bandpass(y1,fc,fs), octave_bandpass(y2,fc,fs)
        Gs=[]
        for start in range(0, len(y1b)-nwin-abs(shift), hop):
            seg1 = y1b[start:start+nwin]
            seg2 = y2b[start+shift:start+shift+nwin]
            dL   = band_level_db(seg1,w) - band_level_db(seg2,w)
            Gs.append(dL - ab*D)
        Gs=np.array(Gs); G_b.append(Gs.mean()); s2_b.append(Gs.var(ddof=1)+1e-6)
    G_b=np.array(G_b); s2_b=np.array(s2_b); wts=1/s2_b
    G_hat=(wts*G_b).sum()/wts.sum(); var_G=1/wts.sum()
    chi2_nu=((G_b-G_hat)**2/s2_b).sum()/(len(bands)-1)
    # invert (§5.4) + uncertainty (§5.5) ...
    return dict(x_hat=..., Sigma_xy=..., G_hat=G_hat, var_G=var_G,
                D=D, tau=tau, chi2_nu=chi2_nu, G_b=G_b, sigma2_b=s2_b)
```

Fill the `...` with the §5.4 inversion (with rotation to grid frame) and §5.5 covariance.

**Reported result:** `x_hat`, `Sigma_xy`. **Error vs. ground truth** (uses `x_true`):
`err = ||x_hat − x_true||`.

---

## 7. Validation ladder (run before touching real fire audio)

1. **Synthetic point source, no noise, free field.** Generate `y1, y2` by delaying/attenuating a
   known broadband signal with the exact `1/r` + `alpha_b` model. `err` should be ~0 and
   `chi2_nu ≈ 0`. Confirms the algebra and frame handling.
2. **Add white noise** at controlled SNR. Watch `err` and `Sigma_xy` grow; check the predicted
   ellipse actually contains `x_true` at the right rate (~68 % for 1-sigma).
3. **Sweep source over the grid.** Map `err` and ellipse area over the 100 × 100 m domain →
   this is your GDOP map. Expect a large bad lobe along the perpendicular bisector; use it to
   choose final mic placement (both on one edge).
4. **Vary T / M.** Confirm `T = 0.5 s`, `M ≈ 19` is on the knee: smaller M inflates `Sigma_xy`,
   larger M barely helps.
5. **Then** feed real 5 s clips; use `chi2_nu` as the go/no-go on each estimate.

---

## 8. Parameter summary (defaults)

| Parameter | Default | Change if… |
|---|---|---|
| Bands | 7 octaves, 125 Hz–8 kHz | need finer diagnostics → 1/3-octave |
| `alpha_b` | ISO 9613-1 at scenario T/RH/P | recompute per scenario, never generic |
| `T_win` | 0.5 s | 0.25 s aggressive / 1.0 s smoother |
| Overlap | 50 % | — |
| Clip length | 5 s (one clip) | 10 s if σ_b unstable; 2 s if fire fast |
| `M` (windows) | ~19 (→ M_eff ≈ 12) | ≥25 for tight variance |
| Mic placement | both on one domain edge | resolves mirror ambiguity |
| Singular guard | flag if `|k−1| < 0.05` | source near bisector |
