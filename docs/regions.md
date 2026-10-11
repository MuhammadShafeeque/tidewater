# Parameters for a region

The laws themselves have no geography. Some default values do: they were
derived for, or first used on, glaciers in Greenland. Outside Greenland they
are a starting point to be checked, not a calibrated value. `tidewater` ships
no parameter set for any other region.

## The values with a Greenland origin

| Parameter | Default | Where it is set | Origin |
|---|---|---|---|
| melt coefficients `A`, `B`, `alpha`, `beta` | 3e-4, 0.15, 0.39, 1.18 | `OceanParams.calving_melt_A`, `_B`, `_alpha`, `_beta` | the submarine melt rate of Rignot et al. (2016), fitted on Greenland glaciers |
| exponent `gamma` of `tf_power` | 1.18 | `OceanParams.calving_tf_exponent` | borrowed from `beta` of that melt rate |
| reference thermal forcing `TF_ref` | the power mean of the record over the reference period | `OceanParams.ocean_tf_ref` (None), `OceanParams.ocean_tf_ref_period` | |
| reference period | 1995 to 2014, both included | `OceanParams.ocean_tf_ref_period` | |
| depth bands | `shallow_0_60:0:60`, `ismip6:200:500`, `front:0:60`, all uniform | `OceanParams.ocean_depth_bands` | `ismip6` is the Greenland band of ISMIP6 |
| the band the laws read | `shallow_0_60` | `OceanParams.ocean_tf_band` | |
| footprint search radii | 15 km, growing by 1.5 up to 150 km | `forcing.footprint.reduce_site` (`r0_km`, `r_grow`, `r_max_km`) | kept from the first Greenland application, to be reviewed |
| footprint target area and cap | 1000 km2, at most 64 cells | `forcing.footprint.reduce_site` (`target_area_km2`, `max_cells`) | as above |

`tidewater.config.REGIONAL_FIELDS` lists the `OceanParams` fields of this
table. Two further defaults depend on the data products, not on the region:
`ocean_open_water_threshold` (0.15, sea-ice concentration at and above which
no water is open) and `ocean_sea_level` (0 m, in the vertical datum of the bed
and surface products).

A fjord shallower than the top of a band has no water in that band. Choose
bands the region's fronts reach, and read the law from one of them
(`ocean_tf_band`).

## Setting them for another region

Change some values of the defaults:

```python
from tidewater.config import DEFAULTS
params = DEFAULTS.replace(ocean_tf_ref_period=(1990, 2009),
                          ocean_depth_bands=("shallow_0_60:0:60:uniform",
                                             "front:0:60:uniform"))
```

Or register a named set once and select it by name; fields not given keep
their defaults, and keyword arguments of `for_region` override the set:

```python
from tidewater.config import OceanParams, register_region
register_region("my_region", calving_tf_exponent=1.0,
                ocean_tf_ref_period=(1990, 2009))
params = OceanParams.for_region("my_region", ocean_sea_level=0.4)
```

`register_region("greenland", ...)` is the only set shipped, and it holds the
defaults above. `OceanParams.from_mapping(mapping)` reads the same fields from
a host's configuration.

The footprint arguments are arguments of `forcing.footprint.reduce_site`, not
of `OceanParams`; pass them per call.

## What to check per region

- the melt coefficients and `gamma`: calibrate them, or state that the
  Greenland values were used;
- the reference period: the ocean product must cover it, and the period
  should match the period of the frontal-ablation observations the constant
  is calibrated on;
- the depth bands against the water depth at the fronts and the sill depths;
- the footprint radii against the distance from the fronts to the first
  ocean cells of the product (fjords narrower than the product's grid);
- the open-water threshold against the sea-ice product.
