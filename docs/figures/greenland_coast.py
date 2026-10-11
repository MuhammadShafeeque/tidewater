"""Cut the Greenland coastline of the locator map out of Natural Earth.

Run once with the path of ``ne_50m_land.geojson`` (Natural Earth, public
domain); it writes ``data/greenland_ne50m.csv``.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "greenland_ne50m.csv"


def main(geojson, step=4):
    with open(geojson) as f:
        features = json.load(f)["features"]
    rings = [ring for feat in features if feat["geometry"]["type"] == "Polygon"
             for ring in feat["geometry"]["coordinates"][:1]]
    rings += [poly[0] for feat in features if feat["geometry"]["type"] == "MultiPolygon"
              for poly in feat["geometry"]["coordinates"]]
    inside = [np.array(r) for r in rings
              if min(p[0] for p in r) < -42 < max(p[0] for p in r)
              and min(p[1] for p in r) < 72 < max(p[1] for p in r)]
    ring = max(inside, key=len)[::step]
    ring = np.vstack([ring, ring[:1]])
    np.savetxt(OUT, ring, fmt="%.3f", delimiter=",", header="lon,lat", comments="")
    print(OUT, len(ring))


if __name__ == "__main__":
    main(*sys.argv[1:])
