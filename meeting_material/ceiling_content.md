# Where the 14.5 dB lives — content stratification

27224 renders / 6806 objects. Object-level means throughout.

## val  (n=1806 objects)

### Correlation of margin with content

| covariate | corr with margin |
|---|---|
| sat_frac | +0.065 |
| mean_lum | -0.408 |
| hf_energy | +0.588 |
| edge_frac | +0.458 |
| fg_frac | -0.412 |
| crop_px | -0.278 |

### margin by sat_frac quintile

(all objects share one value of sat_frac)

### margin by hf_energy quintile

| hf_energy range | n | rec-GT dB | V17 dB | margin dB |
|---|---|---|---|---|
| 0.0215 – 0.0808 | 361 | 45.61 | 33.03 | **12.58** |
| 0.0808 – 0.1118 | 361 | 44.89 | 31.04 | **13.86** |
| 0.1118 – 0.1504 | 361 | 44.70 | 30.21 | **14.48** |
| 0.1504 – 0.1993 | 361 | 44.85 | 29.12 | **15.73** |
| 0.1993 – 0.3996 | 362 | 44.66 | 28.06 | **16.61** |

### margin by fg_frac quintile

| fg_frac range | n | rec-GT dB | V17 dB | margin dB |
|---|---|---|---|---|
| 0.0046 – 0.0308 | 361 | 45.06 | 28.88 | **16.19** |
| 0.0308 – 0.0488 | 361 | 45.34 | 30.07 | **15.27** |
| 0.0488 – 0.0729 | 361 | 45.34 | 30.80 | **14.54** |
| 0.0729 – 0.1070 | 361 | 44.56 | 30.37 | **14.19** |
| 0.1070 – 0.3271 | 362 | 44.41 | 31.34 | **13.07** |


## train  (n=3000 objects)

### Correlation of margin with content

| covariate | corr with margin |
|---|---|
| sat_frac | +0.052 |
| mean_lum | -0.479 |
| hf_energy | +0.644 |
| edge_frac | +0.465 |
| fg_frac | -0.429 |
| crop_px | -0.288 |

### margin by sat_frac quintile

(all objects share one value of sat_frac)

### margin by hf_energy quintile

| hf_energy range | n | rec-GT dB | V17 dB | margin dB |
|---|---|---|---|---|
| 0.0209 – 0.0790 | 600 | 45.51 | 33.98 | **11.53** |
| 0.0790 – 0.1137 | 600 | 44.90 | 31.62 | **13.28** |
| 0.1137 – 0.1512 | 600 | 44.83 | 30.59 | **14.24** |
| 0.1512 – 0.1977 | 600 | 44.91 | 29.82 | **15.09** |
| 0.1977 – 0.7261 | 600 | 44.95 | 28.51 | **16.44** |

### margin by fg_frac quintile

| fg_frac range | n | rec-GT dB | V17 dB | margin dB |
|---|---|---|---|---|
| 0.0044 – 0.0319 | 600 | 45.23 | 29.46 | **15.77** |
| 0.0319 – 0.0500 | 600 | 45.58 | 30.82 | **14.76** |
| 0.0500 – 0.0726 | 600 | 45.46 | 31.29 | **14.17** |
| 0.0726 – 0.1087 | 600 | 44.81 | 31.30 | **13.51** |
| 0.1087 – 0.3703 | 600 | 44.01 | 31.65 | **12.36** |


## unseen2x  (n=2000 objects)

### Correlation of margin with content

| covariate | corr with margin |
|---|---|
| sat_frac | +0.054 |
| mean_lum | -0.429 |
| hf_energy | +0.561 |
| edge_frac | +0.420 |
| fg_frac | -0.374 |
| crop_px | -0.261 |

### margin by sat_frac quintile

(all objects share one value of sat_frac)

### margin by hf_energy quintile

| hf_energy range | n | rec-GT dB | V17 dB | margin dB |
|---|---|---|---|---|
| 0.0163 – 0.0791 | 400 | 45.57 | 33.01 | **12.56** |
| 0.0791 – 0.1157 | 400 | 44.81 | 30.93 | **13.88** |
| 0.1157 – 0.1524 | 400 | 44.67 | 29.91 | **14.77** |
| 0.1524 – 0.2041 | 400 | 44.83 | 29.28 | **15.55** |
| 0.2041 – 0.6631 | 400 | 44.67 | 27.94 | **16.73** |

### margin by fg_frac quintile

| fg_frac range | n | rec-GT dB | V17 dB | margin dB |
|---|---|---|---|---|
| 0.0066 – 0.0309 | 400 | 44.93 | 28.76 | **16.17** |
| 0.0309 – 0.0482 | 400 | 45.40 | 30.09 | **15.31** |
| 0.0482 – 0.0699 | 400 | 45.00 | 30.17 | **14.83** |
| 0.0699 – 0.1049 | 400 | 44.94 | 30.89 | **14.05** |
| 0.1049 – 0.7433 | 400 | 44.27 | 31.16 | **13.11** |



---

# Disentangling detail from size (all 6,806 objects pooled)

`hf_energy` (+0.59) and `fg_frac` (−0.41) both correlate with the margin, but they are themselves
correlated at **−0.62** (small objects measure as higher-detail). The marginal correlations
therefore cannot be read separately. Two checks below.

## The emissive hypothesis is FALSIFIED

| threshold | objects | share | margin |
|---|---|---|---|
| any saturated pixel (`sat_frac > 0`) | 175 | 2.57 % | 15.79 dB |
| `sat_frac > 1e-4` | 16 | 0.24 % | 17.41 dB |
| `sat_frac > 0.01` | **0** | 0 % | — |
| *(all objects)* | 6806 | 100 % | **14.43 dB** |

Emissive content is a **2.6 % tail carrying only +1.4 dB** of extra margin. Removing it entirely
would move the overall mean by ~0.03 dB. The glowing-chest failures are real and spectacular but
statistically negligible — they dominated the tails only because I sorted *by* the tails.

## Margin by detail x size tercile — detail wins

|  | fg LOW (small) | fg MID | fg HIGH (big) |
|---|---|---|---|
| **hf LOW (smooth)** | 13.27 (n=91) | 13.05 (n=641) | 12.44 (n=1535) |
| **hf MID** | 14.41 (n=599) | 14.50 (n=1057) | 14.51 (n=616) |
| **hf HIGH (detail)** | 16.31 (n=1577) | 15.92 (n=574) | 15.30 (n=116) |

Reading **down** each column: detail costs **~+2.9 dB** consistently, in every size band.
Reading **across** each row: size is worth **−1.0 to +0.1 dB**, and vanishes entirely in the middle
band. **Fine-detail content is the driver; object size was mostly the confound.**

Standardised regression (R² = 0.436): `edge_frac` **+0.36**, `hf_energy` **+0.29**,
`fg_frac` −0.23 and `crop_px` +0.23 (these two are 0.75-correlated and nearly cancel).

## The key asymmetry

Across the hf quintiles, **rec-GT is flat at 44.7–45.6 dB while V17 falls 33.98 → 28.51 dB.**
The pruned+recovered input carries fine detail at a constant ~45 dB regardless of how much detail
the object has. The model's error grows monotonically with detail content. So the loss is in the
**model**, and it is specifically a **spatial-bandwidth** failure, not a data or input-quality one.
