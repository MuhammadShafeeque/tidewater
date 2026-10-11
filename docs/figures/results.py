"""The figures of the example case, made from a results directory.

``python docs/figures/results.py <results dir>`` reads ``forcing.csv``,
``forcing.json``, ``map/`` and one CSV per host (the columns of
``examples/example_case.py``) and writes SVG and PNG files to ``docs/img/``.
Every number and curve comes from those files. A directory whose
``forcing.json`` says ``"synthetic": true`` gets a label on every figure.
A directory without host CSVs gives the map and the forcing figure only.
Names after the directory pick figures: ``results.py <dir> map forcing``.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import style as s

HERE = Path(__file__).resolve().parent
COAST = HERE / "data" / "greenland_ne50m.csv"
KM_PER_DEG = 111.195
RUNS = ("control", "ocean")
SCENARIOS = ("ssp126", "ssp585")
SCENARIO_LABEL = {"historical": "historical", "ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}
# hosts first, then the toy routes that stand in for them
ORDER = {"oggm": (0, s.BLUE), "pygem": (1, s.ORANGE), "glogem": (2, s.GREEN),
         "toy: law each step": (0, s.BLUE), "toy: yearly constant": (1, s.ORANGE),
         "toy: factor table": (2, s.GREEN)}
NAMES = {"oggm": "OGGM", "pygem": "PyGEM", "glogem": "GloGEM"}
HOST_COLUMNS = {"host", "run", "scenario", "year", "frontal_ablation_gt_yr"}
ICE_GT_PER_KM3 = 0.9
# fluxes are over the model year, states at its start
FLUXES = {"frontal_ablation_gt_yr", "mass_change"}
SMOOTH_YEARS = 11


class Results:
    """The files of one results directory."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.forcing = pd.read_csv(self.dir / "forcing.csv", parse_dates=["date"])
        with open(self.dir / "forcing.json") as f:
            self.meta = json.load(f)
        frames = [pd.read_csv(p) for p in sorted(self.dir.glob("*.csv"))]
        frames = [f for f in frames if HOST_COLUMNS <= set(f.columns)]
        runs = pd.concat(frames) if frames else pd.DataFrame(columns=sorted(HOST_COLUMNS))
        self.runs = runs[runs["run"].isin(RUNS)]
        self.hosts = sorted(self.runs["host"].unique(),
                            key=lambda h: ORDER.get(h.lower(), (9, None))[0])
        self.synthetic = bool(self.meta.get("synthetic", False))
        self.notes = {k.lower(): v for k, v in self.meta.get("host_notes", {}).items()}

    def color(self, host):
        return ORDER.get(host.lower(), (9, s.PURPLE))[1]

    def label(self, host):
        note, name = self.notes.get(host.lower()), NAMES.get(host.lower(), host)
        return f"{name}\n({note})" if note else name

    def run(self, host, run, scenario):
        r = self.runs
        return r[(r["host"] == host) & (r["run"] == run)
                 & (r["scenario"] == scenario)].sort_values("year")

    def path(self, host, run, scenario):
        """Historical rows followed by the scenario's rows."""
        return pd.concat([self.run(host, run, "historical"),
                          self.run(host, run, scenario)])

    def mass_change(self, host, run):
        """Yearly mass change, Gt/yr, from the volumes of consecutive historical years."""
        v = self.run(host, run, "historical").set_index("year")["volume_km3"]
        return ((v.shift(-1) - v) * ICE_GT_PER_KM3).dropna()

    def stamp(self, fig):
        if self.synthetic:
            s.stamp(fig, "SYNTHETIC DATA: toy host, not the glacier's results")


def _local_km(lon, lat, lon0, lat0):
    return ((np.asarray(lon) - lon0) * KM_PER_DEG * np.cos(np.radians(lat0)),
            (np.asarray(lat) - lat0) * KM_PER_DEG)


def _ortho(lon, lat, lon0=-42.0, lat0=72.0):
    lon, lat, lon0, lat0 = map(np.radians, (np.asarray(lon), np.asarray(lat), lon0, lat0))
    x = np.cos(lat) * np.sin(lon - lon0)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
    return x, y


def figure_map(res, out):
    m = res.dir / "map"
    outline = pd.read_csv(m / "outline.csv")
    flowline = pd.read_csv(m / "flowline.csv") if (m / "flowline.csv").exists() else None
    front = pd.read_csv(m / "front.csv")
    cells = pd.read_csv(m / "ocean_cells.csv")
    depth = xr.open_dataset(m / "water_depth.nc")
    lon0, lat0 = front["lon"].mean(), front["lat"].mean()

    fig = s.plt.figure(figsize=(s.FULL, 3.4))
    loc = fig.add_axes([0.0, 0.3, 0.18, 0.56])
    big = fig.add_axes([0.25, 0.14, 0.37, 0.76])
    zoom = fig.add_axes([0.665, 0.14, 0.24, 0.76])
    cax = fig.add_axes([0.915, 0.14, 0.012, 0.76])

    coast = pd.read_csv(COAST)
    x, y = _ortho(coast["lon"], coast["lat"])
    loc.fill(x, y, color=s.LIGHT, ec=s.GREY, lw=0.5)
    gx, gy = _ortho(lon0, lat0)
    loc.plot(gx, gy, "o", ms=5, mfc=s.VERMILION, mec=s.BLACK, mew=0.5)
    loc.set_aspect("equal")
    loc.axis("off")
    loc.set_title(f"Greenland\n{abs(lat0):.2f}° N, {abs(lon0):.2f}° W",
                  fontsize=s.BASE_PT, loc="left")

    dx, dy = _local_km(depth["lon"], depth["lat"], lon0, lat0)
    water = depth["water_depth"].where(depth["water_depth"] > 0)
    vmax = float(np.nanmax(water))
    for ax in (big, zoom):
        ax.set_facecolor(s.LIGHT)
        mesh = ax.pcolormesh(dx, dy, water, cmap="Blues", vmin=0,
                             vmax=vmax, shading="auto", rasterized=True)
        ox, oy = _local_km(outline["lon"], outline["lat"], lon0, lat0)
        ax.fill(ox, oy, fc=s.ICE, ec=s.GREY, lw=0.7)
        ax.set_aspect("equal")
        ax.set_xlabel("km east of the front")
    for _, c in cells.iterrows():
        cx, cy = _local_km([c.lon - c.dlon / 2, c.lon + c.dlon / 2],
                           [c.lat - c.dlat / 2, c.lat + c.dlat / 2], lon0, lat0)
        big.add_patch(s.plt.Rectangle((cx[0], cy[0]), cx[1] - cx[0], cy[1] - cy[0],
                                      fill=False, ec=s.VERMILION, lw=0.8))
    big.plot(0, 0, "o", ms=4, color=s.VERMILION)
    big.set_ylabel("km north of the front")
    big.set_title("Ocean footprint and water depth", fontsize=s.BASE_PT, loc="left")
    if flowline is not None:
        fx, fy = _local_km(flowline["lon"], flowline["lat"], lon0, lat0)
        zoom.plot(fx, fy, color=s.BLUE, lw=1.0, label="flowline")
    tx, ty = _local_km(front["lon"], front["lat"], lon0, lat0)
    if len(front) > 1:
        zoom.plot(tx, ty, color=s.VERMILION, lw=2.2)
    else:
        zoom.plot(tx, ty, "o", ms=4, color=s.VERMILION)
    zoom.set_xlim(min(ox.min(), -2.5) - 1.5, max(ox.max(), 2.5) + 1.5)
    zoom.set_ylim(min(oy.min(), -2.5) - 1.5, max(oy.max(), 2.5) + 1.5)
    handles = [s.plt.Line2D([], [], ls="none", marker="o", ms=4, color=s.VERMILION),
               s.plt.Line2D([], [], ls="none", marker="s", ms=6, mfc="none", mec=s.VERMILION),
               s.Patch(fc=s.ICE, ec=s.GREY, lw=0.7), s.Patch(fc=s.LIGHT, lw=0)]
    labels = ["calving front", f"ocean cells ({len(cells)})", "glacier outline", "land"]
    if flowline is not None:
        handles.append(s.plt.Line2D([], [], color=s.BLUE, lw=1.0))
        labels.append("flowline")
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.02))
    zoom.set_title("Glacier", fontsize=s.BASE_PT, loc="left")
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("water depth, m")
    res.stamp(fig)
    return s.save(fig, "example-map", out)


def _annual(series):
    return series.groupby(series.index.year).mean()


def figure_forcing(res, out):
    f, meta = res.forcing, res.meta
    ref_name, model_name = meta["reference_product"], meta["model_product"]
    if res.hosts:
        fig, (ax, bx) = s.plt.subplots(2, 1, figsize=(s.FULL, 3.9), sharex=True,
                                       gridspec_kw={"height_ratios": [1.6, 1]})
        fig.subplots_adjust(left=0.08, right=0.76, top=0.92, bottom=0.1, hspace=0.3)
    else:
        fig, ax = s.plt.subplots(figsize=(s.FULL, 2.7))
        fig.subplots_adjust(left=0.08, right=0.76, top=0.9, bottom=0.16)
        ax.set_xlabel("year")

    def yearly(product, scenario, corrected):
        rows = f[(f["product"] == product) & (f["scenario"] == scenario)
                 & (f["corrected"] == corrected)]
        return rows.groupby(rows["date"].dt.year)["thermal_forcing_degc"].mean()

    p0, p1 = meta["bias_correction_period"]
    ax.axvspan(p0, p1 + 1, color=s.LIGHT, lw=0)
    ax.text(p0 + 0.5, 0.97, "bias correction\nreference period", transform=
            ax.get_xaxis_transform(), va="top", fontsize=s.MIN_PT, color=s.GREY)
    ref = yearly(ref_name, "historical", 0)
    ax.plot(ref.index + 0.5, ref, color=s.BLACK, lw=1.4, label=ref_name)
    for scenario in ("historical",) + SCENARIOS:
        color = s.GREY if scenario == "historical" else s.SCENARIO[scenario]
        raw, cor = yearly(model_name, scenario, 0), yearly(model_name, scenario, 1)
        ax.plot(raw.index + 0.5, raw, color=color, lw=0.8, ls=":")
        if len(cor):
            ax.plot(cor.index + 0.5, cor, color=color, lw=1.0,
                    label=f"{model_name},\n{SCENARIO_LABEL[scenario]}")
    if len(f[(f["product"] == model_name) & (f["corrected"] == 0)]):
        ax.plot([], [], color=s.GREY, ls=":", lw=0.8, label="before the correction")
    ax.axhline(meta["tf_ref_degc"], color=s.GREY, lw=0.8, ls="--",
               label=f"$TF_{{ref}}$ = {meta['tf_ref_degc']:.2f} °C")
    offset = meta["bias_offset_degc"][meta["band"]]
    ax.set_ylabel("thermal forcing, °C")
    ax.set_title(f"Yearly mean thermal forcing, {meta['band_top_m']:g}-"
                 f"{meta['band_bottom_m']:g} m; correction offset {offset:+.2f} °C",
                 loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), fontsize=s.MIN_PT)

    if not res.hosts:
        res.stamp(fig)
        return s.save(fig, "example-forcing", out)
    host = res.hosts[0]
    for scenario in SCENARIOS:
        r = res.path(host, "ocean", scenario)
        bx.plot(r["year"] + 0.5, r["ocean_factor"], color=s.SCENARIO[scenario], lw=1.0,
                label=SCENARIO_LABEL[scenario])
    bx.axhline(1, color=s.GREY, lw=0.6, ls="--")
    bx.set_ylabel("ocean factor $F$")
    bx.set_xlabel("year")
    bx.set_title(f"Yearly ocean factor, as {res.label(host)} used it "
                 f"($\\gamma$ = {meta['gamma']:g})", loc="left")
    bx.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    res.stamp(fig)
    return s.save(fig, "example-forcing", out)


OBSERVABLES = (("frontal_ablation_targets", "frontal ablation", "calibration"),
               ("mass_change_targets", "mass change", "mass_balance_calibration"))
PERIOD_MARKER = {"calibration": "o", "evaluation": "^", "mass_balance_calibration": "s"}


def observables(res):
    return [(key, name, res.meta["periods"].get(cal))
            for key, name, cal in OBSERVABLES if res.meta.get(key)]


def period_names(res):
    return {tuple(v): k for k, v in res.meta["periods"].items()}


def period_label(period, calibrated):
    role = "calibration" if list(period) == list(calibrated or []) else "evaluation"
    return f"{period[0]}-{period[1]}\n{role}"


def period_values(res, host, run, key, period):
    """The yearly values of a period, empty when the run does not cover it."""
    first, last = period
    if key == "mass_change_targets":
        return res.mass_change(host, run).loc[first:last]
    series = res.run(host, run, "historical").set_index("year")["frontal_ablation_gt_yr"]
    return series.loc[first:last] if last in series.index else series.iloc[:0]


def period_mean(res, host, run, key, period):
    return period_values(res, host, run, key, period).mean()


def legend(res, ax, markers, observed, **errorbar):
    """Hosts by colour, then the markers, then the observation as it is drawn."""
    for host in res.hosts:
        ax.fill_between([], [], color=res.color(host), label=res.label(host))
    for label, kw in markers:
        if "yerr" in kw:
            ax.errorbar([], [], color=s.GREY, capsize=2, label=label, **kw)
        else:
            ax.plot([], [], color=s.GREY, ls="none", label=label, **kw)
    ax.errorbar([], [], color=s.BLACK, capsize=2, label=observed, **errorbar)
    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))


def figure_calibration(res, out):
    shown = observables(res)
    fig, axes = s.plt.subplots(1, len(shown), figsize=(s.FULL, 2.9), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.76, bottom=0.2, top=0.88, wspace=0.4)
    n = len(res.hosts)
    for ax, (key, name, calibrated) in zip(axes[0], shown):
        targets = res.meta[key]
        for i, t in enumerate(targets):
            ax.errorbar(i - 0.32, t["gt_yr"], yerr=t["error_gt_yr"], fmt="s", color=s.BLACK,
                        ms=4, capsize=2)
            for j, host in enumerate(res.hosts):
                x = i - 0.12 + j * 0.44 / max(n - 1, 1)
                for run, marker in (("control", "o"), ("ocean", "D")):
                    v = period_values(res, host, run, key, t["period"])
                    # standard error of the period mean, from the year-to-year spread
                    ax.errorbar(x + (0.06 if run == "ocean" else -0.06), v.mean(),
                                yerr=v.std() / np.sqrt(v.count()),
                                fmt=marker, color=res.color(host), ms=4, capsize=1.5,
                                elinewidth=0.7, mfc="none" if run == "control" else res.color(host))
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([period_label(t["period"], calibrated) for t in targets])
        ax.set_xlim(-0.6, len(targets) - 0.4)
        ax.set_ylabel(f"{name}, Gt yr$^{{-1}}$")
        ax.set_title(f"{name.capitalize()}", loc="left")
        low, high = ax.get_ylim()
        ax.set_ylim(min(low, 0), max(high, 0))
    legend(res, axes[0, -1], [("control, ±1 s.e.", dict(fmt="o", mfc="none", yerr=[])),
                              ("ocean, ±1 s.e.", dict(fmt="D", yerr=[]))],
           "observed, with error", yerr=[], fmt="s", ms=4)
    res.stamp(fig)
    return s.save(fig, "example-calibration", out)


def figure_scatter(res, out):
    shown = observables(res)
    fig, axes = s.plt.subplots(1, len(shown), figsize=(s.FULL, 3.1), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.76, bottom=0.15, top=0.88, wspace=0.55)
    names = period_names(res)
    used = []
    slots = [(host, run) for host in res.hosts for run in RUNS]
    for ax, (key, name, calibrated) in zip(axes[0], shown):
        points = []
        for t in res.meta[key]:
            period = names.get(tuple(t["period"]))
            if period in PERIOD_MARKER:
                used.append(period)
                points += [(t, period, host, run, period_mean(res, host, run, key, t["period"]))
                           for host, run in slots]
        values = [v for t, *_, y in points
                  for v in (y, t["gt_yr"] - t["error_gt_yr"], t["gt_yr"] + t["error_gt_yr"])]
        low, high = np.nanmin(values), np.nanmax(values)
        pad = 0.08 * (high - low)
        lim = (low - pad, high + pad)
        step = 0.03 * (lim[1] - lim[0])
        for t in res.meta[key]:
            if names.get(tuple(t["period"])) in PERIOD_MARKER:
                ax.errorbar(t["gt_yr"], t["gt_yr"], xerr=t["error_gt_yr"], fmt="none",
                            color=s.BLACK, capsize=2, lw=1.0, zorder=1)
        # the modelled values of a period spread sideways about the observed value
        for t, period, host, run, y in points:
            dx = (slots.index((host, run)) - (len(slots) - 1) / 2) * step
            ax.plot(t["gt_yr"] + dx, y, PERIOD_MARKER[period], color=res.color(host), ms=4,
                    mfc="none" if run == "control" else res.color(host), zorder=2)
        ax.plot(lim, lim, color=s.GREY, lw=0.6, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("observed, Gt yr$^{-1}$")
        ax.set_ylabel("modelled, Gt yr$^{-1}$")
        on = f", calibrated on {calibrated[0]}-{calibrated[1]}" if calibrated else ""
        ax.set_title(name.capitalize() + on, loc="left")
    years = {v: f"{p[0]}-{p[1]}" for p, v in names.items()}
    markers = [(years[p], dict(marker=PERIOD_MARKER[p])) for p in PERIOD_MARKER if p in used]
    markers += [("control", dict(marker="o", mfc="none")), ("ocean", dict(marker="o"))]
    legend(res, axes[0, -1], markers, "observed, with error", xerr=[], fmt="none")
    res.stamp(fig)
    return s.save(fig, "example-scatter", out)


def figure_historical(res, out):
    names = period_names(res)
    window = [p for p in names if names[p] in ("calibration", "evaluation")]
    quantities = [("frontal_ablation_gt_yr", "frontal ablation,\nGt yr$^{-1}$",
                   "frontal_ablation_targets")]
    if res.meta.get("mass_change_targets"):
        quantities.append(("mass_change", "mass change,\nGt yr$^{-1}$", "mass_change_targets"))
    quantities += [("volume_km3", "volume, km$^3$", None),
                   ("front_water_depth_m", "water depth at\nthe front, m", None)]
    fig, axes = s.plt.subplots(len(quantities), 1, sharex=True,
                               figsize=(s.FULL, 1.45 * len(quantities) + 0.6))
    fig.subplots_adjust(left=0.11, right=0.76, top=0.94, bottom=0.07, hspace=0.15)
    for ax, (_, label, key) in zip(axes, quantities):
        ax.set_ylabel(label)
        for first, _ in window:
            ax.axvline(first, color=s.LIGHT, lw=0.8)
        for t in res.meta.get(key) or []:
            first, last = t["period"]
            if (first, last) not in window:
                continue
            low, high = t["gt_yr"] - t["error_gt_yr"], t["gt_yr"] + t["error_gt_yr"]
            ax.fill_between([first, last + 1], low, high, color=s.LIGHT, lw=0)
            ax.plot([first, last + 1], [t["gt_yr"]] * 2, color=s.BLACK, lw=1.2)
    for first, last in window:
        axes[0].text(first + 0.3, 0.97, names[(first, last)], va="top",
                     transform=axes[0].get_xaxis_transform(), color=s.GREY)
    first_year, last_year = min(p[0] for p in window), max(p[1] for p in window)
    for host in res.hosts:
        for run in RUNS:
            r = res.run(host, run, "historical").set_index("year")
            r["mass_change"] = res.mass_change(host, run)
            r = r.loc[first_year:last_year]
            for ax, (column, _, _) in zip(axes, quantities):
                ax.plot(r.index + 0.5 * (column in FLUXES), r[column], color=res.color(host),
                        **s.RUN_STYLE[run])
    axes[0].set_ylim(0, None)
    axes[-1].set_xlabel("year")
    axes[-1].xaxis.set_major_locator(s.plt.MultipleLocator(5))
    axes[0].set_title("Historical runs: calibration and evaluation", loc="left")
    for host in res.hosts:
        axes[0].plot([], [], color=res.color(host), lw=1.4, label=res.label(host))
    for run in RUNS:
        axes[0].plot([], [], color=s.GREY, label=run, **s.RUN_STYLE[run])
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append((s.Patch(fc=s.LIGHT, lw=0), s.plt.Line2D([], [], color=s.BLACK, lw=1.2)))
    labels.append("observed, with error")
    axes[0].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    res.stamp(fig)
    return s.save(fig, "example-historical", out)


def figure_ocean(res, out):
    names = period_names(res)
    window = [p for p in names if names[p] in ("calibration", "evaluation")]
    first, last = min(p[0] for p in window), max(p[1] for p in window)
    fig, (ax, bx) = s.plt.subplots(1, 2, figsize=(s.FULL, 2.9),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.17, top=0.88, wspace=0.3)
    values = []
    for host in res.hosts:
        c = res.run(host, "control", "historical").set_index("year").loc[first:last]
        o = res.run(host, "ocean", "historical").set_index("year").loc[first:last]
        control = c["frontal_ablation_gt_yr"].where(c["frontal_ablation_gt_yr"] > 0)
        ratio = o["frontal_ablation_gt_yr"] / control
        k_ratio = o["calving_k_yr"] / c["calving_k_yr"]
        color = res.color(host)
        ax.plot(ratio.index + 0.5, ratio, color=color, lw=1.4)
        ax.plot(k_ratio.index + 0.5, k_ratio, color=color, lw=0.8, ls=":")
        bx.plot(k_ratio, ratio, "o", color=color, ms=3.5)
        values += list(ratio.dropna()) + list(k_ratio.dropna())
    for (a, _) in window:
        ax.axvline(a, color=s.LIGHT, lw=0.8)
    ax.axhline(1, color=s.GREY, lw=0.6, ls="--")
    ax.set_xlabel("year")
    ax.set_ylabel("ocean-forced / control")
    ax.xaxis.set_major_locator(s.plt.MultipleLocator(5))
    ax.set_title("Frontal ablation, ocean-forced over control", loc="left")
    lim = (0, max(values) * 1.08)
    bx.plot(lim, lim, color=s.GREY, lw=0.6, ls="--")
    bx.set_xlim(lim)
    bx.set_ylim(lim)
    bx.set_aspect("equal")
    bx.set_xlabel("calving constant, ocean-forced / control")
    bx.set_ylabel("ocean-forced / control")
    bx.set_title("Against the constants", loc="left")
    for host in res.hosts:
        bx.fill_between([], [], color=res.color(host), label=res.label(host))
    bx.plot([], [], color=s.GREY, lw=1.4, label="frontal ablation,\nocean / control")
    bx.plot([], [], color=s.GREY, lw=0.8, ls=":", label="calving constant,\nocean / control")
    bx.legend(loc="upper left", bbox_to_anchor=(1.06, 1.0))
    res.stamp(fig)
    return s.save(fig, "example-ocean", out)


def figure_projection(res, out):
    rows = [("frontal_ablation_gt_yr", f"frontal ablation,\nGt yr$^{{-1}}$, {SMOOTH_YEARS}-year mean"),
            ("volume_km3", "volume, km$^3$"),
            ("front_water_depth_m", "water depth at\nthe front, m")]
    fig, axes = s.plt.subplots(len(rows), len(res.hosts), sharex=True, sharey="row",
                               figsize=(s.FULL, 5.2), squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.08, hspace=0.12,
                        wspace=0.1)
    for j, host in enumerate(res.hosts):
        axes[0, j].set_title(res.label(host), loc="left", color=res.color(host))
        for scenario in ("historical",) + SCENARIOS:
            for run in RUNS:
                r = res.run(host, run, scenario).set_index("year")
                r["frontal_ablation_gt_yr"] = r["frontal_ablation_gt_yr"].rolling(
                    SMOOTH_YEARS, center=True, min_periods=1).mean()
                for ax, (column, _) in zip(axes[:, j], rows):
                    ax.plot(r.index + 0.5 * (column in FLUXES), r[column],
                            color=s.SCENARIO[scenario], **s.RUN_STYLE[run])
    for ax, (_, label) in zip(axes[:, 0], rows):
        ax.set_ylabel(label)
    depth = axes[-1, 0]
    top = depth.get_ylim()[1]
    depth.set_ylim(-0.35 * top, top)
    for ax in axes[-1]:
        ax.axhspan(-0.35 * top, 0, color=s.LIGHT, lw=0, zorder=0)
        ax.set_xlabel("year")
        ax.xaxis.set_major_locator(s.plt.MultipleLocator(25))
    depth.text(0.03, 0.04, "front on land", transform=depth.transAxes, color=s.GREY,
               va="bottom")
    axes[0, 0].set_ylim(0, None)
    axes[1, 0].set_ylim(0, None)
    handles = [s.plt.Line2D([], [], color=s.SCENARIO[k], lw=1.4)
               for k in ("historical",) + SCENARIOS]
    handles += [s.plt.Line2D([], [], color=s.GREY, **s.RUN_STYLE[run]) for run in RUNS]
    labels = [SCENARIO_LABEL[k] for k in ("historical",) + SCENARIOS] + list(RUNS)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 1.0),
               ncol=len(labels))
    res.stamp(fig)
    return s.save(fig, "example-projection", out)


FIGURES = (figure_map, figure_forcing, figure_calibration, figure_scatter,
           figure_historical, figure_ocean, figure_projection)


def main(directory, out=None, names=None):
    """Draw the figures of a results directory; ``names`` picks some, as ``map``."""
    res = Results(directory)
    figures = FIGURES if res.hosts else (figure_map, figure_forcing)
    if names:
        figures = [f for f in figures if f.__name__.removeprefix("figure_") in names]
    return [fig(res, out) for fig in figures]


if __name__ == "__main__":
    for path in main(sys.argv[1], names=sys.argv[2:]):
        print(path)
