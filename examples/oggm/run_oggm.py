"""OGGM on the example case: the forcing file, the calibration and two runs.

Without OGGM, it writes the glacier's ocean forcing file and prints the law at
the front. With ``--workdir`` it also runs OGGM (the branch with the
tidewater hooks) on the glacier directory in that OGGM working directory,
which must already hold the glacier's flowlines, inversion and a ``gcm_data``
climate file from the calibration period to 2100: the dynamic calibration of
``k`` on the first target, then a control and an ocean-forced run.

    python examples/oggm/run_oggm.py [results dir] [--workdir DIR] [--scenario ssp126]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example_case import RESULTS, as_forcing_dataset, joined, load_case, read_forcing  # noqa: E402

from tidewater.forcing.io import write_forcing  # noqa: E402
from tidewater.laws import ocean_calving_law  # noqa: E402
from tidewater.periods import run_bounds  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="?", default=RESULTS / "synthetic")
    p.add_argument("--scenario", default="ssp126")
    p.add_argument("--workdir")
    p.add_argument("--out", default="ocean_data.nc")
    a = p.parse_args(argv)
    case = load_case()
    table, meta = read_forcing(a.results)
    g = case["glacier"]
    ds = as_forcing_dataset(joined(table, meta, a.scenario), meta["band"],
                            meta["band_top_m"], meta["band_bottom_m"])
    law = ocean_calving_law(ds, "tf_power", band=meta["band"], tf_ref=meta["tf_ref_degc"],
                            gamma=meta["gamma"], k0=1.0)
    h = g["front_thickness_m"]
    for year in (2005, 2050, 2095):
        q = law.flux_per_year(h, g["front_water_depth_m"], g["front_width_m"], year + 0.5)
        print(f"{year}: flux at k0 = 1 yr-1 and a {h:g} m front {q / 1e6:.3f} 1e6 m3 yr-1")
    if not a.workdir:
        write_forcing(a.out, ds, glacier_lon=g["lon"], glacier_lat=g["lat"])
        print(f"wrote {a.out}")
        return law

    from oggm import GlacierDirectory, cfg

    import tidewater.hosts.oggm
    from tidewater.hosts.oggm import tasks

    cfg.initialize()
    cfg.PATHS["working_dir"] = str(Path(a.workdir).resolve())
    tidewater.hosts.oggm.register()
    gdir = GlacierDirectory(g["rgi7_id"])
    write_forcing(gdir.get_filepath("ocean_data"), ds, glacier_lon=g["lon"],
                  glacier_lat=g["lat"])
    target = meta["frontal_ablation_targets"][0]
    tasks.run_dynamic_calving_k_calibration(
        gdir, ref_fa=target["gt_yr"], ref_fa_err=target["error_gt_yr"],
        ref_period=run_bounds(target["period"]))
    y0, y1 = meta["periods"]["calibration"][0], run_bounds(meta["periods"]["projection"])[1]
    for run, law_name in (("control", "constant"), ("ocean", "tf_power")):
        tasks.run_with_ocean_forcing(gdir, calving_law=law_name, band=meta["band"],
                                     tf_ref=meta["tf_ref_degc"],
                                     law_kwargs={"gamma": meta["gamma"]},
                                     ys=y0, ye=y1, output_filesuffix=f"_{run}")
    print(f"ran {gdir.rgi_id}: control and ocean, {y0}-{y1}")
    return law


if __name__ == "__main__":
    main()
