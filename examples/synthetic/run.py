"""The synthetic example: calibrate, evaluate and project the toy host on three routes.

Writes one results file per route next to the inputs of ``make_inputs.py``,
in the columns every host of the example case uses.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_inputs  # noqa: E402
import toy_host  # noqa: E402
from example_case import RESULTS, joined, read_forcing, write_results  # noqa: E402

from tidewater.calibration.dynamic import calibrate_dynamic_k  # noqa: E402
from tidewater.calibration.static import fit_calving_k, shape_flux  # noqa: E402

RUNS = ("control", "ocean")


def static_k(glacier, target_gt_yr):
    """The constant at which the starting front gives the target."""
    h, d, w = glacier.front
    shape = shape_flux(pd.DataFrame({"thickness_m": [h], "water_depth_m": [d],
                                     "width_m": [w]}, index=["toy"]),
                       flux_unit="Gt yr-1", ice_density=toy_host.ICE_DENSITY)
    k, _ = fit_calving_k(shape, {"toy": target_gt_yr}, flux_unit="Gt yr-1")
    return k


def main(out=RESULTS / "synthetic", quick=False):
    out = Path(out)
    if not (out / "forcing.csv").exists():
        make_inputs.main(out)
    table, meta = read_forcing(out)
    gamma, tf_ref = meta["gamma"], meta["tf_ref_degc"]
    cal, (_, last_hist) = meta["periods"]["calibration"], meta["periods"]["evaluation"]
    target = meta["frontal_ablation_targets"][0]
    proj_last = 2040 if quick else meta["periods"]["projection"][1]
    forcing = {s: joined(table, meta, s) for s in ("ssp126", "ssp585")}

    def tf_of(run, scenario="ssp126"):
        return None if run == "control" else forcing[scenario]

    def run_flux(ks):
        q = {}
        for key, k in ks.items():
            route, run = key.split("|")
            rows = toy_host.run(toy_host.ToyGlacier(), route, k, tuple(cal), "historical",
                                tf_of(run), tf_ref, gamma)
            q[key] = sum(r["frontal_ablation_gt_yr"] for r in rows) / len(rows)
        return q

    keys = [f"{route}|{run}" for route in toy_host.ROUTES for run in RUNS]
    k0 = static_k(toy_host.ToyGlacier(), target["gt_yr"])
    calib = calibrate_dynamic_k(run_flux, dict.fromkeys(keys, k0),
                                dict.fromkeys(keys, target["gt_yr"]),
                                dict.fromkeys(keys, target["error_gt_yr"]))

    summary = []
    for route in toy_host.ROUTES:
        rows = []
        for run in RUNS:
            k = float(calib.loc[f"{route}|{run}", "calving_k"])
            glacier = toy_host.ToyGlacier()
            hist = toy_host.run(glacier, route, k, (cal[0], last_hist), "historical",
                                tf_of(run), tf_ref, gamma)
            rows += [dict(r, run=run) for r in hist]
            for scenario in ("ssp126", "ssp585"):
                proj = toy_host.run(glacier.copy(), route, k, (last_hist + 1, proj_last),
                                    scenario, tf_of(run, scenario), tf_ref, gamma)
                rows += [dict(r, run=run) for r in proj]
            summary.append({"host": f"toy: {route}", "run": run, "calving_k_yr": k,
                            "k_static_yr": k0, **{c: calib.loc[f"{route}|{run}", c]
                                                  for c in ("flux", "ratio", "passes",
                                                            "converged")}})
        name = route.replace(" ", "_")
        write_results([dict(r, host=f"toy: {route}") for r in rows], out / f"toy_{name}.csv")
    pd.DataFrame(summary).to_csv(out / "calibration.csv", index=False, float_format="%.6g")
    return pd.DataFrame(summary)


if __name__ == "__main__":
    print(main(*sys.argv[1:]).to_string(index=False))
