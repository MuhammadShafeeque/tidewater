"""PyGEM on the example case: one law per glacier, then PyGEM's own run script.

Without PyGEM, it builds the law and the yearly factors and prints them. With
``--run`` it also runs ``run_simulation`` for the glacier inside
``ocean_dynamics``; that needs PyGEM and its inputs for the glacier (outline,
thickness, climate) where PyGEM's configuration points. The glacier is the
RGI 7.0 one (branch ``pygem-tidewater-rgi7``), or with ``--rgi6`` the RGI 6.0
one (released PyGEM or the branch on RGI 6.0).

    python examples/pygem/run_pygem.py [results dir] [--run] [--rgi6] [--scenario ssp126]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example_case import RESULTS, as_forcing_dataset, joined, load_case, read_forcing  # noqa: E402

from tidewater.hosts.pygem import glacno, laws_from_forcing, yearly_factors  # noqa: E402


def laws_for(case, table, meta, scenario, rgi6=False):
    """The tf_power law of the glacier and its yearly factors (calendar years)."""
    gid = case["glacier"]["rgi6_id" if rgi6 else "rgi7_id"]
    tf = joined(table, meta, scenario)
    ds = as_forcing_dataset(tf, meta["band"], meta["band_top_m"], meta["band_bottom_m"])
    laws = laws_from_forcing({gid: ds}, "tf_power", band=meta["band"],
                             tf_ref=meta["tf_ref_degc"], gamma=meta["gamma"])
    factors = yearly_factors(tf.to_frame(gid), meta["tf_ref_degc"], gamma=meta["gamma"],
                             incomplete="drop")
    return laws, factors


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="?", default=RESULTS / "synthetic")
    p.add_argument("--scenario", default="ssp126")
    p.add_argument("--run", action="store_true")
    p.add_argument("--rgi6", action="store_true")
    a = p.parse_args(argv)
    case = load_case()
    table, meta = read_forcing(a.results)
    laws, factors = laws_for(case, table, meta, a.scenario, a.rgi6)
    (gid, law), = laws.items()
    print(f"{gid}: law {law.name}, TF_ref {law.tf_ref:.3f} degC, gamma {law.gamma}")
    for year in (2005, 2050, 2095):
        print(f"{year}: k / k0 {law.scaling(year + 0.5):.3f} in mid-year, "
              f"{factors.loc[year].iloc[0]:.3f} as the mean of the year")
    if not a.run:
        return laws, factors

    from pygem.bin.run import run_simulation
    from tidewater.hosts.pygem.dynamics import ocean_dynamics

    sys.argv = ["run_simulation", "-rgi_glac_number", glacno(gid),
                "-option_dynamics", "OGGM"]
    with ocean_dynamics(laws) as runs:
        run_simulation.main()
    print(f"{gid}: {len(runs)} model run(s) with the law")
    return laws, factors


if __name__ == "__main__":
    main()
