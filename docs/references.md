# Equations, constants and their sources

Every equation and every numerical constant in `tidewater`, with where it is in
the code and where it comes from. "This package" marks a formulation or a
value defined here; those are written out in the specification
(`src/tidewater/spec/SPEC.md`). Full references are at the end.

Values marked *regional* were derived for Greenland glaciers and are arguments
to set elsewhere; see [regions.md](regions.md).

## Frontal ablation laws

| Equation or constant | Form or value | In the code | Source |
|---|---|---|---|
| Constant-k (water-depth) law | `q = k d h w` | `laws.ConstantK`, `laws.calving_law_flux` | Oerlemans and Nick (2005); the calving law of OGGM and of PyGEM |
| Power law of thermal forcing | `k(t) = k0 ((1 - a) + a (TF / TF_ref)^gamma)` | `laws.TFPower`, `laws.tf_scaling` | this package |
| Ocean factor | `F = (max(TF, 0) / TF_ref)^gamma` | `laws.ocean_factor` | this package |
| Reference thermal forcing | power mean, `TF_ref = mean(max(TF, 0)^gamma)^(1/gamma)` | `laws.tf_power_mean` | this package |
| Exponent `gamma` (*regional*) | 1.18 | `OceanParams.calving_tf_exponent` | the thermal-forcing exponent of the submarine melt rate of Rignot et al. (2016) |
| Share `a` of the constant that follows the ocean | 1 | `OceanParams.calving_tf_fraction` | this package |
| Submarine melt rate | `m = (A d q_sg^alpha + B) TF^beta`, m d-1 | `laws.submarine_melt_rate` | Rignot et al. (2016), Eq. 1; related: Jenkins (2011), Xu et al. (2013) |
| Melt coefficients (*regional*) | `A` = 3e-4, `B` = 0.15, `alpha` = 0.39, `beta` = 1.18 | `OceanParams.calving_melt_A`, `_B`, `_alpha`, `_beta` | Rignot et al. (2016) |
| Subglacial discharge per unit of front area | `q_sg = Q_sg / (d w)`, m d-1 | `laws.discharge_per_front_area` | as the melt rate of Rignot et al. (2016) takes it |
| Melt plus calving | `q = w d (k_c h + lambda m)` | `laws.MeltPlusCalving` | this package, with the melt rate of Rignot et al. (2016) |
| Undercut efficiency `lambda` | 1 | `OceanParams.calving_undercut_efficiency` | this package |
| Open-water law | melt term multiplied by `f_ow^delta` | `laws.SeaIceModulated` | this package; motivated by the observations of Bendtsen et al. (2017) |
| Open-water fraction | `f_ow = clip((c* - c) / c*, 0, 1)` | `physics.open_water_fraction` | this package |
| Open-water exponent `delta` | 1 | `OceanParams.calving_openwater_exponent` | this package |
| Larger-of rule | `q = w d max(k_c h, lambda m)` | `laws.LargerOf` | Malles et al. (2025) |
| Keeping a calibrated total | `k_c` from `k0` and the period-mean melt speed | `laws.partition_calving_constant`, `laws.larger_of_calving_constant` | this package |

## Sea-water thermodynamics and thermal forcing

| Equation or constant | Form or value | In the code | Source |
|---|---|---|---|
| Thermal forcing | `TF = theta - T_f(S, z)` | `physics.thermal_forcing` | Jenkins (2011); Slater et al. (2020) |
| Freezing point, linear form | `T_f = lambda1 S + lambda2 - lambda3 z` | `physics.freezing_point` | Jenkins (2011), as used by Slater et al. (2020), Eq. 6 |
| Freezing-point coefficients | `lambda1` = -5.73e-2 degC psu-1, `lambda2` = 8.32e-2 degC, `lambda3` = 7.61e-4 degC m-1 | `physics.LAMBDA1`, `LAMBDA2`, `LAMBDA3` | Jenkins (2011), as used by Slater et al. (2020) |
| Freezing point, TEOS-10 form | from the `gsw` toolbox | `physics.freezing_point(teos10=True)` | IOC, SCOR and IAPSO (2010) |
| Absolute to practical salinity | divide by 35.16504 / 35 | `forcing.io.SA_PER_SP` | reference-composition salinity of TEOS-10: IOC, SCOR and IAPSO (2010); for model output see McDougall et al. (2021) |
| Bias offset onto a reference product | one additive offset per glacier and depth band over a reference period | `forcing.bias` | Slater et al. (2020), Eq. 5 |
| Clip at zero | `max(TF, 0)` before any power | `laws.ocean_factor` | this package |

## Depth ranges and the horizontal footprint

| Equation or constant | Form or value | In the code | Source |
|---|---|---|---|
| Depth range of the ISMIP6 retreat parameterization (*regional*) | 200 to 500 m | band `ismip6` in `OceanParams.ocean_depth_bands` | Slater et al. (2019, 2020) |
| Shallow range (*regional*) | 0 to 60 m | bands `shallow_0_60` and `front` | the depth measured at a tidewater outlet by Bendtsen et al. (2017); see also Luckman et al. (2015) for the 20 to 60 m range |
| Reference period of the thermal forcing (*regional*) | 1995 to 2014 | `OceanParams.ocean_tf_ref_period` | this package (the period of the application it was developed on) |
| Band average | weighted mean of the level thermal forcing, uniform or by depth | `physics.thermal_forcing_bands` | this package |
| Footprint target area | 1,000 km2 of the nearest wet cells | `forcing.footprint.reduce_site` | this package |
| Search radii | 15 km, growing by a factor of 1.5 to 150 km | `forcing.footprint.radius_schedule` | this package |
| Largest number of cells in a footprint | 64 | `forcing.footprint.reduce_site` | this package |
| Great-circle distance | haversine formula on a sphere | `forcing.footprint.haversine_km` | standard formula |
| Earth radius | 6371.0 km | `forcing.footprint.R_EARTH_KM` | conventional mean radius |
| Sea-ice concentration above which no water is open | 0.15 | `OceanParams.ocean_open_water_threshold` | the threshold in common use for sea-ice extent; set it to the product in use |

## Geometry at the front

| Equation or constant | Form or value | In the code | Source |
|---|---|---|---|
| Water depth | bed elevation below sea level, positive downwards | `physics.water_depth` | definition, as in Oerlemans and Nick (2005) |
| Flotation thickness | `h_f = d rho_w / rho_i` | `physics.flotation_thickness` | hydrostatic equilibrium |
| Ice density | 900 kg m-3 | `OceanParams.ice_density` | the default of OGGM (Maussion et al., 2019) |
| Sea-water density | 1028 kg m-3 | `OceanParams.ocean_water_density` | a common value for sea water |
| Sea level | 0 m | `OceanParams.ocean_sea_level` | datum of the bed product in use |
| Water depth from a gridded bed | deepest ocean cells next to the outline | `physics.terminus_water_depth_from_bed` | this package; first used with BedMachine (Morlighem et al., 2017, 2025) |

## Calibration

| Equation or constant | Form or value | In the code | Source |
|---|---|---|---|
| Static constant | observed flux divided by the flux of the constant-k law at unit `k` | `calibration.static` | this package |
| Dynamic constant | `k` multiplied by target over modelled flux, repeated per front | `calibration.dynamic` | this package |
| Tolerance, largest step, passes | 5 %, a factor of 4, 10 | `calibration.dynamic.step`, `calibrate_dynamic_k` | this package |
| Spread of the constants, robust form | 1.4826 times the median absolute deviation | `calibration.dynamic.log_spread` | standard consistency factor of the median absolute deviation for a normal distribution |
| Constant of glaciers without an observation | covariate model on log `k`, scored by leave-one-out error | `calibration.covariates` | this package |
| Permutation test | 500 permutations by default | `calibration.permutation.permutation_test` | this package (a standard permutation test) |
| Frontal-ablation targets | any dataset through a column mapping | `calibration.targets` | developed with Kochtitzky et al. (2022) |
| Surface mass balance at a calving glacier | fitted to the geodetic mass change plus frontal ablation | host adapters | Hugonnet et al. (2021) for the geodetic data; the separate frontal term as in Malles et al. (2023) |

## Time and units

| Equation or constant | Form or value | In the code | Source |
|---|---|---|---|
| Seconds in a year | 365 days | `config.SEC_IN_YEAR` | the convention of OGGM (Maussion et al., 2019), kept so that constants in yr-1 are exchangeable |
| Start of the hydrological year | October in the north, April in the south | `timeconv.HYDRO_START_NORTH`, `HYDRO_START_SOUTH` | the convention of OGGM (Maussion et al., 2019); GloGEM's own starts are in `hosts.glogem` |
| Kelvin to degrees Celsius | subtract 273.15 | `units.to_celsius` | definition of the Celsius scale |
| Volume to mass | `1 Gt = 1e12 kg`, `1 km3 = 1e9 m3`, with the ice density as an argument | `units.convert_flux` | definitions |
| Yearly ocean factor | mean of the monthly factors, weighted by days | `timeconv.yearly_ocean_factor` | this package |
| Glacier ids | RGI 6 and RGI 7 forms | `ids` | RGI 7.0 Consortium (2023) for the version 7 form |

## Values without a published source

Four values above rest on common practice, not on a publication cited here:
the sea-ice threshold of 0.15, the Earth radius of 6371 km, the factor 1.4826
and the sea-water density of 1028 kg m-3. They are arguments or module
constants and can be changed.

## References

- Bendtsen, J., Mortensen, J., Lennert, K., Ehn, J. K., Boone, W., Galindo,
  V., Hu, Y.-B., Dmitrenko, I. A., Kirillov, S. A., Kjeldsen, K. K.,
  Kristoffersen, Y., Barber, D. G., and Rysgaard, S. (2017). Sea ice breakup
  and marine melt of a retreating tidewater outlet glacier in northeast
  Greenland (81°N). *Scientific Reports*, 7, 4941.
  https://doi.org/10.1038/s41598-017-05089-3
- Hugonnet, R., McNabb, R., Berthier, E., Menounos, B., Nuth, C., Girod, L.,
  Farinotti, D., Huss, M., Dussaillant, I., Brun, F., and Kääb, A. (2021).
  Accelerated global glacier mass loss in the early twenty-first century.
  *Nature*, 592, 726-731. https://doi.org/10.1038/s41586-021-03436-z
- IOC, SCOR and IAPSO (2010). *The international thermodynamic equation of
  seawater - 2010: Calculation and use of thermodynamic properties.* UNESCO.
- Jenkins, A. (2011). Convection-driven melting near the grounding lines of
  ice shelves and tidewater glaciers. *Journal of Physical Oceanography*, 41,
  2279-2294. https://doi.org/10.1175/JPO-D-11-03.1
- Kochtitzky, W., Copland, L., Van Wychen, W., Hugonnet, R., Hock, R.,
  Dowdeswell, J. A., Benham, T., Strozzi, T., Glazovsky, A., Lavrentiev, I.,
  Rounce, D. R., Millan, R., Cook, A., Dalton, A., Jiskoot, H., Cooley, J.,
  Jania, J., and Navarro, F. (2022). The unquantified mass loss of Northern
  Hemisphere marine-terminating glaciers from 2000-2020. *Nature
  Communications*, 13, 5835. https://doi.org/10.1038/s41467-022-33231-x
- Luckman, A., Benn, D. I., Cottier, F., Bevan, S., Nilsen, F., and Inall, M.
  (2015). Calving rates at tidewater glaciers vary strongly with ocean
  temperature. *Nature Communications*, 6, 8566.
  https://doi.org/10.1038/ncomms9566
- Malles, J.-H., Marzeion, B., and Myers, P. G. (2025). Freshwater input from
  glacier melt outside Greenland alters modeled northern high-latitude ocean
  circulation. *Earth System Dynamics*, 16, 347-377.
  https://doi.org/10.5194/esd-16-347-2025
- Malles, J.-H., Maussion, F., Ultee, L., Kochtitzky, W., Copland, L., and
  Marzeion, B. (2023). Exploring the impact of a frontal ablation
  parameterization on projected 21st-century mass change for Northern
  Hemisphere glaciers. *Journal of Glaciology*, 69, 1317-1332.
  https://doi.org/10.1017/jog.2023.19
- Maussion, F., Butenko, A., Champollion, N., Dusch, M., Eis, J., Fourteau,
  K., Gregor, P., Jarosch, A. H., Landmann, J., Oesterle, F., Recinos, B.,
  Rothenpieler, T., Vlug, A., Wild, C. T., and Marzeion, B. (2019). The Open
  Global Glacier Model (OGGM) v1.1. *Geoscientific Model Development*, 12,
  909-931. https://doi.org/10.5194/gmd-12-909-2019
- McDougall, T. J., Barker, P. M., Holmes, R. M., Pawlowicz, R., Griffies,
  S. M., and Durack, P. J. (2021). The interpretation of temperature and
  salinity variables in numerical ocean model output and the calculation of
  heat fluxes and heat content. *Geoscientific Model Development*, 14,
  6445-6466. https://doi.org/10.5194/gmd-14-6445-2021
- Morlighem, M., et al. (2025). *IceBridge BedMachine Greenland, Version 6.* NASA
  National Snow and Ice Data Center Distributed Active Archive Center.
  https://doi.org/10.5067/6B6B225B8V2D
- Morlighem, M., and 31 others (2017). BedMachine v3: Complete bed topography
  and ocean bathymetry mapping of Greenland from multibeam echo sounding
  combined with mass conservation. *Geophysical Research Letters*, 44, 11051-11061.
  https://doi.org/10.1002/2017GL074954
- Oerlemans, J. and Nick, F. M. (2005). A minimal model of a tidewater
  glacier. *Annals of Glaciology*, 42, 1-6.
  https://doi.org/10.3189/172756405781813023
- RGI 7.0 Consortium (2023). *Randolph Glacier Inventory - a dataset of global
  glacier outlines, Version 7.0.* NASA National Snow and Ice Data Center
  Distributed Active Archive Center. https://doi.org/10.5067/F6JMOVY5NAVZ
- Rignot, E., Xu, Y., Menemenlis, D., Mouginot, J., Scheuchl, B., Li, X.,
  Morlighem, M., Seroussi, H., van den Broeke, M., Fenty, I., Cai, C., An, L.,
  and de Fleurian, B. (2016). Modeling of ocean-induced ice melt rates of five
  west Greenland glaciers over the past two decades. *Geophysical Research
  Letters*, 43, 6374-6382. https://doi.org/10.1002/2016GL068784
- Slater, D. A., Felikson, D., Straneo, F., Goelzer, H., Little, C. M.,
  Morlighem, M., Fettweis, X., and Nowicki, S. (2020). Twenty-first century
  ocean forcing of the Greenland ice sheet for modelling of sea level
  contribution. *The Cryosphere*, 14, 985-1008.
  https://doi.org/10.5194/tc-14-985-2020
- Slater, D. A., Straneo, F., Felikson, D., Little, C. M., Goelzer, H.,
  Fettweis, X., and Holte, J. (2019). Estimating Greenland tidewater glacier
  retreat driven by submarine melting. *The Cryosphere*, 13, 2489-2509.
  https://doi.org/10.5194/tc-13-2489-2019
- Xu, Y., Rignot, E., Fenty, I., Menemenlis, D., and Flexas, M. M. (2013).
  Subaqueous melting of Store Glacier, west Greenland from three-dimensional,
  high-resolution numerical modeling and ocean observations. *Geophysical
  Research Letters*, 40, 4648-4653. https://doi.org/10.1002/grl.50825

## Datasets of the examples

The datasets of the example case (see [examples](examples.md)); those already
listed above (BedMachine, RGI 7.0, Kochtitzky et al., 2022, Hugonnet et al.,
2021) are not repeated.

- E.U. Copernicus Marine Service Information (2018). *Global Ocean Physics
  Reanalysis* (GLOBAL_MULTIYEAR_PHY_001_030). Mercator Ocean International.
  https://doi.org/10.48670/moi-00021
- Hersbach, H., and 42 others (2020). The ERA5 global reanalysis. *Quarterly
  Journal of the Royal Meteorological Society*, 146, 1999-2049.
  https://doi.org/10.1002/qj.3803
- Hugonnet, R., McNabb, R., Berthier, E., Menounos, B., Nuth, C., Girod, L.,
  Farinotti, D., Huss, M., Dussaillant, I., Brun, F., and Kääb, A. (2021).
  *Accelerated global glacier mass loss in the early twenty-first century -
  Dataset.* Theia. https://doi.org/10.6096/13
- Jungclaus, J., and 47 others (2019). *MPI-M MPI-ESM1.2-HR model output
  prepared for CMIP6 CMIP historical.* Earth System Grid Federation.
  https://doi.org/10.22033/ESGF/CMIP6.6594
- Kochtitzky, W. and Copland, L. (2021). *Decadal Northern Hemisphere
  marine-terminating glacier positions, 2000-2020.* Canadian Cryospheric
  Information Network. https://doi.org/10.21963/13257
- Lellouche, J.-M., Greiner, E., Bourdallé-Badie, R., Garric, G., Melet, A.,
  Drévillon, M., Bricaud, C., Hamon, M., Le Galloudec, O., Regnier, C.,
  Candela, T., Testut, C.-E., Gasparin, F., Ruggiero, G., Benkiran, M.,
  Drillet, Y., and Le Traon, P.-Y. (2021). The Copernicus global 1/12°
  oceanic and sea ice GLORYS12 reanalysis. *Frontiers in Earth Science*, 9,
  698876. https://doi.org/10.3389/feart.2021.698876
- Maffezzoli, N. (2025). *Ice thickness of the world's glaciers (IceBoost
  model)*, version iceboost-v2.0. Zenodo. https://doi.org/10.5281/zenodo.17724512
- Maffezzoli, N., Rignot, E., Barbante, C., Petersen, T., and Vascon, S.
  (2025). A gradient-boosted tree framework to model the ice thickness of the
  world's glaciers (IceBoost v1.1). *Geoscientific Model Development*, 18,
  2545-2568. https://doi.org/10.5194/gmd-18-2545-2025
- Müller, W. A., Jungclaus, J. H., Mauritsen, T., Baehr, J., Bittner, M.,
  Budich, R., Bunzel, F., Esch, M., Ghosh, R., Haak, H., Ilyina, T.,
  Kleine, T., Kornblueh, L., Li, H., Modali, K., Notz, D., Pohlmann, H.,
  Roeckner, E., Stemmler, I., Tian, F., and Marotzke, J. (2018). A
  higher-resolution version of the Max Planck Institute Earth System Model
  (MPI-ESM1.2-HR). *Journal of Advances in Modeling Earth Systems*, 10,
  1383-1413. https://doi.org/10.1029/2017MS001217
- RGI Consortium (2017). *Randolph Glacier Inventory - A Dataset of Global
  Glacier Outlines, Version 6.* National Snow and Ice Data Center.
  https://doi.org/10.7265/4m1f-gd79
- Schupfner, M., and 50 others (2019). *DKRZ MPI-ESM1.2-HR model output
  prepared for CMIP6 ScenarioMIP.* Earth System Grid Federation.
  https://doi.org/10.22033/ESGF/CMIP6.2450
- Natural Earth (no date). *1:50m physical vectors, land.* Public domain.
  https://www.naturalearthdata.com
