; tidewater: the formulas of spec/SPEC.md in IDL (also runs in GDL).
; Units as in SPEC.md: m, degC, psu, yr-1 for calving constants, m3 s-1 for fluxes.
; Call the procedure tidewater_laws once to compile every function of this file.

; A parameter, or its default when it is undefined or NaN.
function tw_opt, value, default
  compile_opt idl2
  if n_elements(value) eq 0 then return, default
  if ~finite(value[0]) then return, default
  return, double(value)
end

function tw_freezing_point, s, z
  compile_opt idl2
  return, -5.73d-2 * s + 8.32d-2 - 7.61d-4 * z
end

function tw_thermal_forcing, theta, s, z
  compile_opt idl2
  return, theta - tw_freezing_point(s, z)
end

function tw_open_water_fraction, siconc, threshold
  compile_opt idl2
  t = tw_opt(threshold, 0.15d)
  return, (((t - siconc) / t) > 0d) < 1d
end

function tw_flotation_thickness, d, rho_ocean, rho_ice
  compile_opt idl2
  return, d * tw_opt(rho_ocean, 1028d) / tw_opt(rho_ice, 900d)
end

function tw_water_depth, b, z_sl
  compile_opt idl2
  return, tw_opt(z_sl, 0d) - b
end

function tw_submerged_depth, h, s, z_sl
  compile_opt idl2
  return, h - (s - tw_opt(z_sl, 0d))
end

function tw_front_thickness, s, d, z_sl
  compile_opt idl2
  return, ((s - tw_opt(z_sl, 0d)) > 0d) + d
end

; Weighted mean thermal forcing of the levels with top <= z <= bottom.
function tw_thermal_forcing_band, theta, s, z, top, bottom, weighting, front_depth = front_depth
  compile_opt idl2
  n = n_elements(z)
  bot = bottom
  if n_elements(front_depth) gt 0 then bot = front_depth > min(z)
  g = dblarr(n) + 1d
  if n gt 1 then begin
    g[0] = z[1] - z[0]
    g[n - 1] = z[n - 1] - z[n - 2]
    if n gt 2 then g[1 : n - 2] = (z[2 : n - 1] - z[0 : n - 3]) / 2d
  endif
  sel = where(z ge top and z le bot, ns)
  if ns eq 0 then message, 'no level in the band'
  if total(finite(theta[sel]) and finite(s[sel])) lt ns then message, 'a level in the band is not finite'
  if weighting eq 'uniform' then wt = dblarr(ns) + 1d / ns $
  else if weighting eq 'depth_weighted' then wt = g[sel] / total(g[sel]) $
  else message, 'unknown weighting ' + weighting
  return, total(wt * (theta[sel] - tw_freezing_point(s[sel], z[sel])))
end

; F = (max(TF, 0) / TF_ref)^gamma, with F = 1 at gamma = 0.
function tw_ocean_factor, tf, tf_ref, gamma
  compile_opt idl2
  g = tw_opt(gamma, 1.18d)
  if g eq 0 then return, 1d + 0d * tf
  return, ((tf > 0d) / tf_ref) ^ g
end

function tw_tf_scaling, tf, tf_ref, gamma, tf_fraction
  compile_opt idl2
  a = tw_opt(tf_fraction, 1d)
  f = tw_ocean_factor(tf, tf_ref, gamma)
  if a eq 1 then return, f
  return, (1d - a) + a * f
end

; Ocean factor of a model year: the weighted mean of the monthly factors.
function tw_yearly_ocean_factor, tf_months, tf_ref, gamma, tf_fraction, weights = weights
  compile_opt idl2
  w = n_elements(weights) gt 0 ? double(weights) : dblarr(n_elements(tf_months)) + 1d
  return, total(w * tw_tf_scaling(tf_months, tf_ref, gamma, tf_fraction)) / total(w)
end

function tw_tf_power_mean, series, gamma
  compile_opt idl2
  ok = where(finite(series), n)
  if n eq 0 then return, !values.d_nan
  v = series[ok] > 0d
  g = tw_opt(gamma, 1.18d)
  if g eq 0 then return, total(v) / n
  return, (total(v ^ g) / n) ^ (1d / g)
end

function tw_inversion_k_from_ocean, k_ref, tf_mean, tf_ref, gamma, tf_fraction
  compile_opt idl2
  ref = n_elements(tf_ref) gt 0 ? double(tf_ref) : double(tf_mean)
  if ~finite(ref) || ref le 0 then message, 'tf_ref must be finite and > 0'
  return, k_ref * tw_tf_scaling(tf_mean, ref, gamma, tf_fraction)
end

; Subglacial discharge per unit of submerged front area, m d-1.
function tw_discharge_per_front_area, q_sg, d, w
  compile_opt idl2
  return, (q_sg > 0d) * 86400d / ((d * w) > 1d-3)
end

function tw_rignot, d, q, tf, a_, b_, alpha, beta
  compile_opt idl2
  return, (tw_opt(a_, 3d-4) * (d > 0d) * q ^ tw_opt(alpha, 0.39d) + tw_opt(b_, 0.15d)) $
    * tf ^ tw_opt(beta, 1.18d)
end

; Melt rate (Rignot et al., 2016), m d-1; NaN at TF <= 0.
function tw_submarine_melt_rate, d, q, tf, a_coef = a_coef, b_coef = b_coef, alpha = alpha, $
  beta = beta
  compile_opt idl2
  if ~finite(tf) || ~finite(d) || tf le 0 then return, !values.d_nan
  qq = finite(q) ? double(q) : 0d
  return, tw_rignot(d, qq, tf, a_coef, b_coef, alpha, beta)
end

; Flux of a law at constant forcing TF, m3 s-1; MELT returns the melt part.
; LAW: 'constant', 'tf_power', 'melt_calving', 'sea_ice' or 'larger_of'.
function tw_law_flux, law, k, h, d, w, tf, tf_ref = tf_ref, gamma = gamma, $
  tf_fraction = tf_fraction, q_sg = q_sg, open_water = open_water, a_coef = a_coef, b_coef = b_coef, $
  alpha = alpha, beta = beta, lam = lam, delta = delta, sec_in_year = sec_in_year, melt = melt
  compile_opt idl2
  k_s = k / tw_opt(sec_in_year, 31536000d)
  melt = 0d
  if law eq 'constant' then return, k_s * d * h * w
  if d le 0 || h le 0 then return, 0d
  if total(law eq ['tf_power', 'melt_calving', 'sea_ice', 'larger_of']) eq 0 then $
    message, 'unknown law ' + string(law)
  if law eq 'tf_power' then begin
    ; in the order of 'constant', so gamma = 0 or a = 0 is the same number
    k_t = k_s * tw_tf_scaling(tf > 0d, tf_ref, gamma, tf_fraction)
    return, (k_t * d * h * w) > 0d
  endif else begin
    q = n_elements(q_sg) gt 0 ? tw_discharge_per_front_area(q_sg, d, w) : 0d
    u_c = k_s * h
    u_m = tw_opt(lam, 1d) * tw_rignot(d, q, tf > 0d, a_coef, b_coef, alpha, beta) / 86400d
    if law eq 'sea_ice' then u_m = (open_water > 0d) ^ tw_opt(delta, 1d) * u_m
    if law eq 'larger_of' then begin
      if u_c ge u_m then u_m = 0d else u_c = 0d
    endif
  endelse
  u_tot = u_c + u_m
  share = u_tot gt 0 ? ((u_m / u_tot) > 0d) < 1d : 0d
  flux = (u_tot * d * w) > 0d
  melt = flux * share
  return, flux
end

; Constant-k flux at a prescribed depth, km3 yr-1 (k in yr-1); H returns the thickness.
function tw_calving_law_flux, d, k, w, s, z_sl, h = h
  compile_opt idl2
  h = ((s - tw_opt(z_sl, 0d)) > 0d) + d
  return, (k * h * d * w / 1d9) > 0d
end

; Calving constant of a sum law keeping a calibrated total (SPEC section 7).
; MELT_RATE in m s-1; FRACTION returns the melt-driven share.
function tw_partition_calving_constant, k, melt_rate, h, lam, sec_in_year = sec_in_year, $
  fraction = fraction
  compile_opt idl2
  sy = tw_opt(sec_in_year, 31536000d)
  u_tot = k / sy * h
  u_m = tw_opt(lam, 1d) * melt_rate
  fraction = 0d
  if u_tot le 0 then return, 0d
  fraction = 1d
  if u_m ge u_tot then return, 0d
  fraction = u_m / u_tot
  return, (u_tot - u_m) / h * sy
end

; Calving constant of larger_of keeping a calibrated total; U: melt speeds, m s-1.
function tw_larger_of_calving_constant, k, u, h, sec_in_year = sec_in_year, fraction = fraction
  compile_opt idl2
  sy = tw_opt(sec_in_year, 31536000d)
  u_tot = k / sy * h
  n = n_elements(u)
  fraction = 0d
  if u_tot le 0 || n eq 0 then return, k > 0d
  v = double(u) > 0d
  v = v[sort(v)]
  fraction = 1d
  if total(v) / n ge u_tot then return, 0d
  for j = 1L, n do begin
    rest = j lt n ? total(v[j : n - 1]) : 0d
    c = (n * u_tot - rest) / j
    if j eq n then break
    if c le v[j] then break
  endfor
  above = where(v gt c, na)
  fraction = na gt 0 ? total(v[above]) / n / u_tot : 0d
  return, c / h * sy
end

pro tidewater_laws
  compile_opt idl2
end
