"""GloGEM on the example case: the yearly ocean factor tables and the config lines.

Writes one table directory per scenario. GloGEM, on its branch with the
calving factor hook, then runs as usual with the three printed lines in its
``config.pro``; that run needs IDL and GloGEM's own inputs for the glacier.

    python examples/glogem/run_glogem.py [results dir] [output dir]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example_case import RESULTS, joined, load_case, read_forcing  # noqa: E402

from tidewater.hosts import glogem  # noqa: E402

SCENARIOS = ("ssp126", "ssp585")


def main(results=RESULTS / "synthetic", out="glogem_tables"):
    case = load_case()
    table, meta = read_forcing(results)
    gid = case["glacier"]["rgi6_id"]
    out = Path(out)
    factors = {}
    for scenario in SCENARIOS:
        tf = joined(table, meta, scenario).to_frame(gid)
        factor, _ = glogem.write_ocean_factor_tables(
            out / scenario, tf, meta["tf_ref_degc"], gamma=meta["gamma"],
            incomplete="drop", outside="stop")
        factors[scenario] = factor[gid]
    y0, y1 = meta["periods"]["calibration"]
    cal = factors[SCENARIOS[0]].loc[y0:y1].mean()
    print(f"GloGEM model years (October to September) {factor.index[0]}-{factor.index[-1]}")
    print(f"mean factor {y0}-{y1}: {cal:.3f}")
    for scenario, f in factors.items():
        print(f"mean factor 2081-2100, {scenario}: {f.loc[2081:2100].mean():.3f}")
    print("config.pro:")
    print(f"  !PATH = !PATH + ':{glogem.IDL_DIR}'")
    print("  calving_factor_function = 'tw_table_factor'")
    print(f"  calving_factor_dir = '{(out / SCENARIOS[0]).resolve()}'")
    return factors


if __name__ == "__main__":
    main(*sys.argv[1:])
