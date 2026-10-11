; tidewater: read a per-glacier, per-model-year table of tidewater.export and look
; up the value of one glacier and model year, for GloGEM's calving model.

; Read a table: '#' header lines of key=value pairs, a line 'id <years>', one row per glacier.
function tw_read_table, path
  compile_opt idl2
  n = file_lines(path)
  lines = strarr(n)
  openr, lun, path, /get_lun
  readf, lun, lines
  free_lun, lun
  meta = {variable: '', units: '', region: '', start_month: '', label: '', $
    rgi_version: '', outside: '', missing: -9999d}
  tags = strlowcase(tag_names(meta))
  i = 0L
  while i lt n && strmid(lines[i], 0, 1) eq '#' do begin
    items = strsplit(strmid(lines[i], 1), ' ', /extract)
    for j = 0, n_elements(items) - 1 do begin
      kv = strsplit(items[j], '=', /extract, /preserve_null)
      if n_elements(kv) ne 2 then continue
      t = where(tags eq kv[0], ct)
      if ct eq 0 then continue
      if kv[0] eq 'missing' then meta.missing = double(kv[1]) else meta.(t[0]) = kv[1]
    endfor
    i++
  endwhile
  head = strsplit(lines[i], ' ', /extract)
  years = long(head[1 : *])
  body = lines[i + 1 : *]
  body = body[where(strtrim(body, 2) ne '', ng)]
  if ng eq 0 then message, 'no glacier in ' + path
  ids = strarr(ng)
  values = dblarr(n_elements(years), ng)
  for g = 0L, ng - 1 do begin
    a = strsplit(body[g], ' ', /extract)
    ids[g] = a[0]
    values[*, g] = double(a[1 : *])
  endfor
  bad = where(values eq meta.missing, cb)
  if cb gt 0 then values[bad] = !values.d_nan
  return, create_struct(meta, 'path', path, 'ids', ids, 'years', years, 'values', values)
end

; Calendar month of GloGEM's bal_month (a month, or a day of a 365-day year when daily).
function tw_glogem_start_month, bal_month, time_resolution
  compile_opt idl2
  if time_resolution ne 'daily' then return, fix(bal_month)
  ends = total([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], /cumulative)
  return, fix((where(bal_month le ends))[0] + 1)
end

; Value of GLACIER_ID (GloGEM's within-region id) in model YEAR, from the table of REGION
; (the two-digit RGI region) in DIRECTORY. Glaciers absent from the table get 1.
; OUTSIDE: 'stop', 'one' or 'nearest', for a year or value the table lacks; the default
; is the table's own setting (outside= in its header), else 'stop'.
; BAL_MONTH, TIME_RESOLUTION and RGI_VERSION, when given, are checked against the header.
function tw_table_factor, directory, region, glacier_id, year, variable = variable, $
  bal_month = bal_month, time_resolution = time_resolution, rgi_version = rgi_version, $
  outside = outside
  compile_opt idl2
  common tw_table_cache, tw_path, tw_table, tw_absent
  if n_elements(region) eq 0 then message, 'no region: run GloGEM with region_id_loop from region_batch.dat'
  var = n_elements(variable) gt 0 ? variable : 'ocean_factor'
  reg = size(region, /type) eq 7 ? strtrim(region, 2) : string(region, format = '(i2.2)')
  path = directory + path_sep() + var + '_' + reg + '.dat'
  if n_elements(tw_path) eq 0 || tw_path ne path then begin
    tw_path = path
    tw_absent = ~file_test(path)
    if tw_absent then begin
      print, 'tidewater: no table ' + path + ', factor 1 for the region'
    endif else begin
      tw_table = tw_read_table(path)
      if tw_table.label ne '' && tw_table.label ne 'end' then $
        message, path + ': model years must be labelled by their last calendar year (label=end)'
      if n_elements(bal_month) gt 0 && tw_table.start_month ne '' && tw_table.start_month ne 'per_glacier' then begin
        res = n_elements(time_resolution) gt 0 ? time_resolution : 'monthly'
        want = tw_glogem_start_month(bal_month, res)
        if fix(tw_table.start_month) ne want then message, path + ': start_month=' + $
          tw_table.start_month + ' but the run starts its year in month ' + strtrim(want, 2)
      endif
      if n_elements(rgi_version) gt 0 && tw_table.rgi_version ne '' then begin
        if strmid(tw_table.rgi_version, 0, 1) ne strmid(strtrim(rgi_version, 2), 0, 1) then $
          message, path + ': RGI ' + tw_table.rgi_version + ' ids, the run uses RGI ' + strtrim(rgi_version, 2)
      endif
    endelse
  endif
  if tw_absent then return, 1d
  g = where(tw_table.ids eq strtrim(glacier_id, 2), cg)
  if cg eq 0 then return, 1d
  how = n_elements(outside) gt 0 ? outside : tw_table.outside
  if how eq '' then how = 'stop'
  if total(how eq ['stop', 'one', 'nearest']) eq 0 then message, path + ': outside=' + how
  y = where(tw_table.years eq year, cy)
  value = cy gt 0 ? tw_table.values[y[0], g[0]] : !values.d_nan
  if finite(value) then return, value
  if how eq 'one' then return, 1d
  if how eq 'nearest' then begin
    ok = where(finite(tw_table.values[*, g[0]]), cok)
    if cok gt 0 then begin
      void = min(abs(tw_table.years[ok] - year), j)
      return, tw_table.values[ok[j], g[0]]
    endif
  endif
  message, path + ': no value for glacier ' + strtrim(glacier_id, 2) + ' in year ' + strtrim(year, 2)
end
