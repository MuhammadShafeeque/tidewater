# Ocean-forced frontal ablation: language-neutral specification

This file defines every formula of `tidewater.physics` and `tidewater.laws`
in plain text, so that the scheme can be written in another language and
checked against `reference_vectors.csv` without running Python.

All arithmetic is IEEE double precision. `max`, `min` and `clip` act on
scalars. `x ** y` is the real power; it is only ever applied to `x >= 0`.

## 1. Symbols and units

| Symbol | Meaning | Unit |
|---|---|---|
| `h` | ice thickness at the calving front | m |
| `d` | water depth at the calving front | m |
| `w` | width of the calving front | m |
| `s` | surface elevation of the terminus | m |
| `b` | bed elevation | m |
| `z_sl` | sea level, in the vertical datum of `s` and `b` (default 0) | m |
| `k`, `k0`, `k_c` | calving constant | yr-1 |
| `TF` | ocean thermal forcing | degC |
| `TF_ref` | reference thermal forcing, > 0 | degC |
| `theta` | sea-water potential temperature | degC |
| `S` | practical salinity | psu |
| `z` | depth, positive downwards | m |
| `c` | sea-ice concentration | 0 to 1 |
| `f_ow` | open-water fraction | 0 to 1 |
| `q_sg` | subglacial discharge | m3 s-1 of water |
| `q` | subglacial discharge per unit of submerged front area | m d-1 |
| `mdot` | submarine melt rate | m d-1 |
| `u` | front-normal speed | m s-1 |
| `Q` | frontal ablation, a volume flux of ice | m3 s-1 |
| `rho_i`, `rho_o` | density of ice and of sea water | kg m-3 |

Constants:

    SEC_IN_DAY  = 86400
    SEC_IN_YEAR = 31536000        (365 days; an argument of every law)
    LAMBDA1 = -5.73e-2 degC psu-1
    LAMBDA2 =  8.32e-2 degC
    LAMBDA3 =  7.61e-4 degC m-1

Default parameter values:

    gamma = 1.18      a (tf_fraction) = 1.0
    A = 3e-4          B = 0.15        alpha = 0.39     beta = 1.18
    lam = 1.0         delta = 1.0
    threshold = 0.15  rho_o = 1028    rho_i = 900

A calving constant given in yr-1 is applied as `k_s = k / SEC_IN_YEAR`
(s-1). The division is done first, before any product with a length.

## 2. Thermodynamics

Freezing point (linear form):

    T_f(S, z) = LAMBDA1 * S + LAMBDA2 - LAMBDA3 * z

Thermal forcing of one level (not clipped here):

    TF(theta, S, z) = theta - T_f(S, z)

Open-water fraction:

    f_ow(c) = clip((threshold - c) / threshold, 0, 1)

Flotation thickness:

    h_f(d) = d * rho_o / rho_i

Water depth over a bed (negative above the water):

    d(b) = z_sl - b

Water depth at a front of known thickness and surface elevation:

    d(h, s) = h - (s - z_sl)

Front thickness at a prescribed water depth, freeboard plus depth:

    h(s, d) = max(s - z_sl, 0) + d

## 3. Depth-band average

Given levels `z[i]` with `theta[i]`, `S[i]`, a band `[top, bottom]` selects
the levels with `top <= z[i] <= bottom` (both ends included). It is an error
if no level is selected or if a selected value is not finite.

Weights over the selected levels:

    uniform:         wt[i] = 1 / n
    depth_weighted:  wt[i] = g[i] / sum(g over the selected levels)

where `g` is the centred difference of the whole depth axis (all `N` levels,
before the selection), taken per level index:

    g[0]   = z[1] - z[0]
    g[N-1] = z[N-1] - z[N-2]
    g[i]   = (z[i+1] - z[i-1]) / 2        for 0 < i < N-1

The band thermal forcing is the weighted mean of the level thermal forcing,
each level at its own depth:

    TF_band = sum(wt[i] * (theta[i] - T_f(S[i], z[i])))

The band temperature and salinity are the same weighted means of `theta`
and `S`.

A band flagged as a front band takes `bottom = max(d_front, min(z))` when a
front water depth `d_front` is given.

## 4. The ocean factor

    F(TF) = (max(TF, 0) / TF_ref) ** gamma

    scaling(TF) = F(TF)                     if a == 1
                = (1 - a) + a * F(TF)       otherwise

with `0 ** 0 = 1`, so `gamma = 0` gives `F = 1` for every `TF`.

Power mean of a series (the `TF_ref` at which `F` averages to one). Drop the
non-finite values, clip the rest at zero, then

    TF_pm = mean(TF ** gamma) ** (1 / gamma)     if gamma != 0
          = mean(TF)                             if gamma == 0

and `nan` when no finite value is left.

Calving constant of an inversion, from a period-mean thermal forcing
`TF_mean` (when no `TF_ref` is given, `TF_ref = TF_mean`; `TF_ref` must be
finite and > 0):

    k_inv = k_ref * scaling(TF_mean)

## 5. Submarine melt

Discharge per unit of submerged front area:

    q = max(q_sg, 0) * SEC_IN_DAY / max(d * w, 1e-3)          [m d-1]

Melt rate (Rignot et al., 2016, Eq. 1):

    mdot = (A * max(d, 0) * q ** alpha + B) * TF ** beta      [m d-1]

In the stand-alone function `submarine_melt_rate`, `mdot` is `nan` when
`TF <= 0` or when `d` or `TF` is not finite, and a non-finite `q` counts as
zero. Inside the laws, `TF` is clipped first (`TF = max(TF, 0)`), so the
melt rate is zero at and below the freezing point, and `q = 0` when the law
has no discharge record.

Melt speed applied by the laws:

    u_m = lam * mdot / SEC_IN_DAY                              [m s-1]

## 6. The laws

Every law gives a pair of front-normal speeds `(u_c, u_m)` in m s-1. The
flux follows from the submerged area:

    if d <= 0 or h <= 0:   Q = 0
    else:                  Q = max((u_c + u_m) * d * w, 0)     [m3 s-1]

The melt share of `Q` is `u_m / (u_c + u_m)` clipped to [0, 1], and 0 when
`u_c + u_m <= 0`.

`constant` is the exception: it applies no test on `d` or `h` and no
clipping, and multiplies in this order:

    constant:      Q = k_s * d * h * w

    tf_power:      u_c = k_s * scaling(max(TF, 0)) * h
                   u_m = 0

    melt_calving:  u_c = k_s * h
                   u_m = lam * mdot / SEC_IN_DAY

    sea_ice:       u_c = k_s * h
                   u_m = max(f_ow, 0) ** delta * (lam * mdot / SEC_IN_DAY)

    larger_of:     with u_c, u_m of melt_calving:
                   (u_c, 0)   if u_c >= u_m
                   (0, u_m)   otherwise

Nesting: `tf_power` is `constant` at `TF == TF_ref`, at `gamma = 0` and at
`a = 0`; `melt_calving` and `larger_of` are `constant` at `lam = 0`;
`sea_ice` is `melt_calving` at `delta = 0`.

Time: a law holds its forcing on a time axis in fractional years, a monthly
value at `year + (month - 0.5) / 12`. The value at a model time is the
linear interpolation between the neighbouring values, and the first or last
value outside the record (no extrapolation). The clipping `max(TF, 0)` is
applied after the interpolation.

Other rate bases: `Q [m3 yr-1] = Q [m3 s-1] * SEC_IN_YEAR`,
`Q [km3 yr-1] = Q [m3 yr-1] / 1e9`, `Q [Gt yr-1] = Q [m3 yr-1] * rho_i / 1e12`.

## 7. Keeping a calibrated total

A constant `k` calibrated against an observed frontal ablation on a front of
thickness `h` already holds the melt-driven part. With `u_tot = k_s * h`:

Sum laws (`melt_calving`, `sea_ice`), from the period mean `u_m` of the melt
speed the law applies:

    if u_tot <= 0:       k_c = 0,  fraction = 0
    if u_m >= u_tot:     k_c = 0,  fraction = 1
    otherwise:           k_c = (u_tot - u_m) / h * SEC_IN_YEAR
                         fraction = u_m / u_tot

`larger_of`, from the `n` melt speeds `u[t]` of the period, clipped at zero
and sorted in ascending order (`u[0] <= ... <= u[n-1]`): find `c` with
`mean(max(c, u[t])) = u_tot`.

    if u_tot <= 0 or n == 0:     k_c = max(k, 0),  fraction = 0
    if mean(u) >= u_tot:         k_c = 0,          fraction = 1
    otherwise, for j = 1 .. n:
        c = (n * u_tot - sum(u[j], ..., u[n-1])) / j
        stop at the first j < n with c <= u[j], else at j = n
    k_c = c / h * SEC_IN_YEAR
    fraction = sum(u[t] for u[t] > c) / n / u_tot

## 8. Flux of the constant-k law at a prescribed depth

    h = max(s - z_sl, 0) + d
    Q = max(k * h * d * w / 1e9, 0)        [km3 yr-1, k in yr-1]

The products are taken from left to right, the division last. The result
also carries the freeboard `h - d` and the flotation thickness `h_f(d)`.

## 9. reference_vectors.csv

One row per case. `function` names the formula, the input columns hold its
arguments (an empty cell: not read by that function), `out1` and `out2` hold
the results (`nan`: no second result). Numbers are printed with the shortest
text that reads back to the same double. The three `series` columns hold
values separated by `;`.

| function | inputs | out1 | out2 |
|---|---|---|---|
| `freezing_point` | so, depth | T_f, degC | |
| `thermal_forcing` | thetao, so, depth | TF, degC | |
| `open_water_fraction` | siconc, threshold | f_ow | |
| `flotation_thickness` | water_depth, rho_ocean, rho_ice | h_f, m | |
| `thermal_forcing_band` | series (theta), series2 (S), series3 (z), band_top, band_bottom, weighting | TF_band, degC | |
| `ocean_factor` | tf, tf_ref, gamma | F | |
| `tf_scaling` | tf, tf_ref, gamma, tf_fraction | scaling | |
| `tf_power_mean` | series (TF), gamma | TF_pm, degC | |
| `inversion_k_from_ocean` | k, tf (the mean), tf_ref, gamma, tf_fraction | k_inv, yr-1 | |
| `discharge_per_front_area` | q_sg, water_depth, width | q, m d-1 | |
| `submarine_melt_rate` | water_depth, q_sg_per_area (q), tf, A, B, alpha, beta | mdot, m d-1 | |
| `calving_law_flux` | water_depth, k, width, surface_h, sea_level (empty: 0) | Q, km3 yr-1 | h, m |
| `water_depth` | bed_elevation, sea_level | d, m | |
| `submerged_depth` | thick, surface_h, sea_level | d, m | |
| `front_thickness` | surface_h, water_depth, sea_level | h, m | |
| `law_constant` | k, thick, water_depth, width | Q, m3 s-1 | melt part of Q (0) |
| `law_tf_power` | k, thick, water_depth, width, tf, tf_ref, gamma, tf_fraction | Q, m3 s-1 | melt part of Q (0) |
| `law_melt_calving` | k, thick, water_depth, width, tf, q_sg, A, B, alpha, beta, lam | Q, m3 s-1 | melt part of Q |
| `law_sea_ice` | as `law_melt_calving`, open_water, delta | Q, m3 s-1 | melt part of Q |
| `law_larger_of` | as `law_melt_calving` | Q, m3 s-1 | melt part of Q |
| `partition_calving_constant` | k, melt_rate (mdot in m s-1), thick, lam | k_c, yr-1 | fraction |
| `larger_of_calving_constant` | k, series (u in m s-1), thick | k_c, yr-1 | fraction |

In the `law_*` rows the forcing is constant in time, `q_sg` is in m3 s-1 and
an empty `q_sg` means a law without a discharge record. The melt part of `Q`
is `Q` times the melt share of section 6.

An implementation agrees with the reference when every `out1` and `out2` is
reproduced to a relative difference of 1e-12 (and exactly where the
reference is 0 or `nan`). The file is written by `make_vectors.py` and
checked against the Python code by `tests/test_spec.py`.

## 10. Sources

| Section | Equation or constant | Source |
|---|---|---|
| 2 | thermal forcing; linear freezing point and its coefficients | Jenkins (2011), as used by Slater et al. (2020) |
| 2 | TEOS-10 freezing point | IOC, SCOR and IAPSO (2010) |
| 4 | ocean factor, power mean, power law of the calving constant | this specification |
| 5 | submarine melt rate and its coefficients `A`, `B`, `alpha`, `beta` | Rignot et al. (2016), Eq. 1 |
| 6 | constant-k law | Oerlemans and Nick (2005) |
| 6 | melt plus calving, open-water law | this specification |
| 6 | larger-of rule | Malles et al. (2025) |
| 7 | keeping a calibrated total | this specification |

- IOC, SCOR and IAPSO (2010). The international thermodynamic equation of
  seawater - 2010: Calculation and use of thermodynamic properties. UNESCO.
- Jenkins, A. (2011). Convection-driven melting near the grounding lines of
  ice shelves and tidewater glaciers. Journal of Physical Oceanography, 41,
  2279-2294. https://doi.org/10.1175/JPO-D-11-03.1
- Malles, J.-H., Marzeion, B., and Myers, P. G. (2025). Freshwater input from
  glacier melt outside Greenland alters modeled northern high-latitude ocean
  circulation. Earth System Dynamics, 16, 347-377.
  https://doi.org/10.5194/esd-16-347-2025
- Oerlemans, J. and Nick, F. M. (2005). A minimal model of a tidewater
  glacier. Annals of Glaciology, 42, 1-6.
  https://doi.org/10.3189/172756405781813023
- Rignot, E., Xu, Y., Menemenlis, D., Mouginot, J., Scheuchl, B., Li, X.,
  Morlighem, M., Seroussi, H., van den Broeke, M., Fenty, I., Cai, C., An, L.,
  and de Fleurian, B. (2016). Modeling of ocean-induced ice melt rates of five
  west Greenland glaciers over the past two decades. Geophysical Research
  Letters, 43, 6374-6382. https://doi.org/10.1002/2016GL068784
- Slater, D. A., Felikson, D., Straneo, F., Goelzer, H., Little, C. M.,
  Morlighem, M., Fettweis, X., and Nowicki, S. (2020). Twenty-first century
  ocean forcing of the Greenland ice sheet for modelling of sea level
  contribution. The Cryosphere, 14, 985-1008.
  https://doi.org/10.5194/tc-14-985-2020
