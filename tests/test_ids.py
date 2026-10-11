import pandas as pd
import pytest

from tidewater import ids


def test_parse_rgi6():
    g = ids.parse_id("RGI60-01.00570")
    assert (g.version, g.region, g.number, g.kind) == ("6.0", 1, 570, "G")
    assert g.is_rgi and g.raw == "RGI60-01.00570"
    assert ids.rgi_version("RGI50-17.01234") == "5.0"


def test_parse_rgi7():
    g = ids.parse_id("RGI2000-v7.0-G-03-01450")
    assert (g.version, g.region, g.number, g.kind) == ("7.0", 3, 1450, "G")
    c = ids.parse_id(" RGI2000-v7.0-C-19-00012 ")
    assert (c.version, c.region, c.number, c.kind) == ("7.0", 19, 12, "C")


@pytest.mark.parametrize("raw", ["GLIMS-G012345E78901N", "glacier 7", "1.00570",
                                 "RGI60-1.570", "RGI2000-v7.0-X-01-00001", 42])
def test_opaque_ids_pass_through(raw):
    g = ids.parse_id(raw)
    assert not g.is_rgi and g.raw == str(raw).strip()
    assert g.region is None and g.number is None
    assert ids.rgi_version(raw) is None
    assert ids.to_pygem_glacno(raw) == str(raw)
    assert ids.to_glogem(raw) == (None, str(raw))


def test_format_rgi_round_trip():
    for raw in ("RGI60-01.00570", "RGI2000-v7.0-G-03-01450",
                "RGI2000-v7.0-C-19-00012"):
        g = ids.parse_id(raw)
        assert ids.format_rgi(g.region, g.number, g.version, g.kind) == raw
    with pytest.raises(ValueError):
        ids.format_rgi(1, 1, "7.0", kind="X")


def test_pygem_glacno():
    assert ids.to_pygem_glacno("RGI60-01.00570") == "1.00570"
    assert ids.to_pygem_glacno("RGI60-17.00005") == "17.00005"
    assert ids.from_pygem_glacno("1.00570") == "RGI60-01.00570"
    assert ids.from_pygem_glacno(1.0057) == "RGI60-01.00570"
    assert ids.from_pygem_glacno("17.00005") == "RGI60-17.00005"
    assert ids.to_pygem_glacno("RGI2000-v7.0-G-05-09345") == "05-09345"
    assert ids.from_pygem_glacno("05-09345") == "RGI2000-v7.0-G-05-09345"
    with pytest.raises(ValueError, match="map the id first"):
        ids.to_pygem_glacno("RGI2000-v7.0-C-01-00570")
    with pytest.raises(ValueError):
        ids.from_pygem_glacno("abc")


def test_glogem():
    assert ids.to_glogem("RGI60-01.01450") == (1, "01450")
    assert ids.to_glogem("RGI2000-v7.0-G-05-00007") == (5, "00007")
    names = {1: "Alaska", 5: "Greenland"}
    assert ids.to_glogem("RGI60-01.01450", names) == ("Alaska", "01450")
    assert ids.from_glogem(1, "01450") == "RGI60-01.01450"
    assert ids.from_glogem("Greenland", 7, version="7.0",
                           region_names=names) == "RGI2000-v7.0-G-05-00007"


def link_table():
    return pd.DataFrame({
        "v6": ["RGI60-01.00001", "RGI60-01.00002", "RGI60-01.00002", None],
        "v7": ["RGI2000-v7.0-G-01-00011", "RGI2000-v7.0-G-01-00021",
               "RGI2000-v7.0-G-01-00022", "RGI2000-v7.0-G-01-00030"],
        "overlap": [1.0, 0.2, 0.8, 0.5]})


def test_map_ids_with_weight():
    out = ids.map_ids(["RGI60-01.00002", "RGI60-01.00001", "other"],
                      link_table(), "v6", "v7", weight="overlap")
    assert out == ["RGI2000-v7.0-G-01-00022", "RGI2000-v7.0-G-01-00011", "other"]
    back = ids.map_ids(["RGI2000-v7.0-G-01-00021"], link_table(), "v7", "v6")
    assert back == ["RGI60-01.00002"]


def test_map_ids_ambiguous_and_missing():
    link = link_table()
    with pytest.raises(ValueError, match="several candidates"):
        ids.map_ids(["RGI60-01.00002"], link, "v6", "v7")
    assert ids.map_ids(["RGI60-01.00001"], link, "v6", "v7") == [
        "RGI2000-v7.0-G-01-00011"]
    assert ids.map_ids(iter(["RGI60-01.00002"]), link, "v6", "v7",
                       multiple="first") == ["RGI2000-v7.0-G-01-00021"]
    assert ids.map_ids(["x"], link, "v6", "v7", missing="none") == [None]
    with pytest.raises(KeyError):
        ids.map_ids(["x"], link, "v6", "v7", missing="raise")
    with pytest.raises(ValueError):
        ids.map_ids(["x"], link, "v6", "v7", missing="skip")


def test_region_of():
    out = ids.region_of(["RGI60-01.00570", "RGI2000-v7.0-G-19-00001", "x"])
    assert out["RGI60-01.00570"] == 1 and out.iloc[1] == 19
    assert pd.isna(out["x"])
