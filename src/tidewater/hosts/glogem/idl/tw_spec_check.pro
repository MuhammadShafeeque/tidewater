; Check tidewater_laws.pro against spec/reference_vectors.csv.
; Prints one line per failing row and a summary line starting with 'tw_spec_check'.

function tw_cell, cells, columns, name
  compile_opt idl2
  i = where(columns eq name)
  text = cells[i[0]]
  if text eq '' || strlowcase(text) eq 'nan' then return, !values.d_nan
  return, double(text)
end

function tw_series, cells, columns, name
  compile_opt idl2
  i = where(columns eq name)
  return, double(strsplit(cells[i[0]], ';', /extract))
end

function tw_spec_case, fn, c, cols
  compile_opt idl2
  nan = !values.d_nan
  v = tw_cell(c, cols, 'k') & h = tw_cell(c, cols, 'thick')
  d = tw_cell(c, cols, 'water_depth') & w = tw_cell(c, cols, 'width')
  tf = tw_cell(c, cols, 'tf') & sl = tw_cell(c, cols, 'sea_level')
  if ~finite(sl) then sl = 0d
  case fn of
    'freezing_point': return, [tw_freezing_point(tw_cell(c, cols, 'so'), tw_cell(c, cols, 'depth')), nan]
    'thermal_forcing': return, [tw_thermal_forcing(tw_cell(c, cols, 'thetao'), $
      tw_cell(c, cols, 'so'), tw_cell(c, cols, 'depth')), nan]
    'open_water_fraction': return, [tw_open_water_fraction(tw_cell(c, cols, 'siconc'), $
      tw_cell(c, cols, 'threshold')), nan]
    'flotation_thickness': return, [tw_flotation_thickness(d, tw_cell(c, cols, 'rho_ocean'), $
      tw_cell(c, cols, 'rho_ice')), nan]
    'thermal_forcing_band': begin
      i = where(cols eq 'weighting')
      return, [tw_thermal_forcing_band(tw_series(c, cols, 'series'), tw_series(c, cols, 'series2'), $
        tw_series(c, cols, 'series3'), tw_cell(c, cols, 'band_top'), $
        tw_cell(c, cols, 'band_bottom'), c[i[0]]), nan]
    end
    'ocean_factor': return, [tw_ocean_factor(tf, tw_cell(c, cols, 'tf_ref'), tw_cell(c, cols, 'gamma')), nan]
    'tf_scaling': return, [tw_tf_scaling(tf, tw_cell(c, cols, 'tf_ref'), tw_cell(c, cols, 'gamma'), $
      tw_cell(c, cols, 'tf_fraction')), nan]
    'tf_power_mean': return, [tw_tf_power_mean(tw_series(c, cols, 'series'), tw_cell(c, cols, 'gamma')), nan]
    'inversion_k_from_ocean': begin
      ref = tw_cell(c, cols, 'tf_ref')
      if finite(ref) then return, [tw_inversion_k_from_ocean(v, tf, ref, tw_cell(c, cols, 'gamma'), $
        tw_cell(c, cols, 'tf_fraction')), nan]
      return, [tw_inversion_k_from_ocean(v, tf, undefined_ref, tw_cell(c, cols, 'gamma'), $
        tw_cell(c, cols, 'tf_fraction')), nan]
    end
    'discharge_per_front_area': return, [tw_discharge_per_front_area(tw_cell(c, cols, 'q_sg'), d, w), nan]
    'submarine_melt_rate': return, [tw_submarine_melt_rate(d, tw_cell(c, cols, 'q_sg_per_area'), tf, $
      a_coef = tw_cell(c, cols, 'A'), b_coef = tw_cell(c, cols, 'B'), alpha = tw_cell(c, cols, 'alpha'), $
      beta = tw_cell(c, cols, 'beta')), nan]
    'calving_law_flux': begin
      q = tw_calving_law_flux(d, v, w, tw_cell(c, cols, 'surface_h'), sl, h = hf)
      return, [q, hf]
    end
    'water_depth': return, [tw_water_depth(tw_cell(c, cols, 'bed_elevation'), sl), nan]
    'submerged_depth': return, [tw_submerged_depth(h, tw_cell(c, cols, 'surface_h'), sl), nan]
    'front_thickness': return, [tw_front_thickness(tw_cell(c, cols, 'surface_h'), d, sl), nan]
    'partition_calving_constant': begin
      kc = tw_partition_calving_constant(v, tw_cell(c, cols, 'melt_rate'), h, $
        tw_cell(c, cols, 'lam'), fraction = f)
      return, [kc, f]
    end
    'larger_of_calving_constant': begin
      kc = tw_larger_of_calving_constant(v, tw_series(c, cols, 'series'), h, fraction = f)
      return, [kc, f]
    end
    else: begin
      if strmid(fn, 0, 4) ne 'law_' then message, 'unknown function ' + fn
      q_sg = tw_cell(c, cols, 'q_sg')
      if ~finite(q_sg) then undefine_q = temporary(q_sg)
      q = tw_law_flux(strmid(fn, 4), v, h, d, w, tf, tf_ref = tw_cell(c, cols, 'tf_ref'), $
        gamma = tw_cell(c, cols, 'gamma'), tf_fraction = tw_cell(c, cols, 'tf_fraction'), $
        q_sg = q_sg, open_water = tw_cell(c, cols, 'open_water'), a_coef = tw_cell(c, cols, 'A'), $
        b_coef = tw_cell(c, cols, 'B'), alpha = tw_cell(c, cols, 'alpha'), beta = tw_cell(c, cols, 'beta'), $
        lam = tw_cell(c, cols, 'lam'), delta = tw_cell(c, cols, 'delta'), melt = m)
      return, [q, m]
    end
  endcase
end

pro tw_spec_check, path, rtol = rtol, failures = failures
  compile_opt idl2
  tidewater_laws
  tol = n_elements(rtol) gt 0 ? double(rtol) : 1d-12
  n = file_lines(path)
  lines = strarr(n)
  openr, lun, path, /get_lun
  readf, lun, lines
  free_lun, lun
  cols = strsplit(lines[0], ',', /extract, /preserve_null)
  failures = 0L
  worst = 0d
  rows = 0L
  for r = 1L, n - 1 do begin
    if strtrim(lines[r], 2) eq '' then continue
    c = strsplit(lines[r], ',', /extract, /preserve_null)
    fn = c[(where(cols eq 'function'))[0]]
    got = tw_spec_case(fn, c, cols)
    want = [tw_cell(c, cols, 'out1'), tw_cell(c, cols, 'out2')]
    rows++
    for o = 0, 1 do begin
      if ~finite(want[o]) then ok = ~finite(got[o]) $
      else if want[o] eq 0 then ok = got[o] eq 0 $
      else begin
        rel = abs(got[o] - want[o]) / abs(want[o])
        if rel gt worst then worst = rel
        ok = rel le tol
      endelse
      if ~ok then begin
        failures++
        print, 'FAIL ', c[0], ' ', fn, ' out', strtrim(o + 1, 2), ' got ', $
          string(got[o], format = '(e25.17)'), ' want ', string(want[o], format = '(e25.17)')
      endif
    endfor
  endfor
  print, 'tw_spec_check rows=' + strtrim(rows, 2) + ' failures=' + strtrim(failures, 2) + $
    ' max_rel_diff=' + strtrim(string(worst, format = '(e10.3)'), 2)
end
