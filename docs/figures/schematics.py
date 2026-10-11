"""The schematics of the documentation: how tidewater works.

Run ``python docs/figures/schematics.py`` with the ``plot`` extra installed;
it writes SVG and PNG files to ``docs/img/``.
"""
import sys

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import style as s
from tidewater import laws, physics, timeconv
from tidewater.config import DEFAULTS


def overview():
    fig, ax = s.canvas(s.FULL, 3.2)
    w, h, gap = 0.23, 0.355, 0.0207
    xs = [0.012 + i * (w + gap) for i in range(4)]
    top, low = 0.59, 0.155
    r0, r_max, grow = 15.0, 150.0, 1.5
    lam = r"$\lambda_1 S + \lambda_2 - \lambda_3 z$"
    row1 = [
        ("Ocean product", "potential temperature, salinity\nand sea-ice cover on\n"
         "(time, level, cell); regular,\ncurvilinear or unstructured",
         "forcing.io.normalise", SKY_FILL),
        ("Footprint", f"nearest wet cells around the\nterminus, about 1,000 km$^2$, at\n"
         f"most 64; search radius {r0:g} km,\n"
         f"×{grow:g} per step while ≤ {r_max:g} km",
         "forcing.footprint.reduce_site", SKY_FILL),
        ("Depth range", "a band [top, bottom], for\nexample 0-60 m, averaged\n"
         "uniformly or by depth; a front\nband ends at the front's depth",
         "physics.thermal_forcing_bands", SKY_FILL),
        ("Thermal forcing", "$TF = \\theta - T_f(S, z)$\n"
         f"$T_f$ = {lam}\none monthly series per band",
         "physics.thermal_forcing", SKY_FILL),
    ]
    row2 = [
        ("Bias correction", "optional: one offset per band\nmatches the mean of a\n"
         "reference period to a\nreference product",
         "forcing.bias.bias_correct", SKY_FILL),
        ("Ocean factor", "$F = (\\max(TF, 0)\\,/\\,TF_{ref})^{\\gamma}$\n"
         "$TF_{ref}$: power mean over\na reference period\n"
         f"$\\gamma$ = {DEFAULTS.calving_tf_exponent:g} by default",
         "laws.ocean_factor, .tf_power_mean", ORANGE_FILL),
        ("Frontal ablation", "one of five laws, e.g.\n$Q = k_0\\,F\\,d\\,h\\,w$\n"
         "m$^3$ ice s$^{-1}$, or $F$ alone\nfor a host's own law",
         "laws", ORANGE_FILL),
        ("Host model", "supplies $h$, $d$, $w$ and time;\nkeeps the geometry,\n"
         "the dynamics, the capping\nand the mass removal",
         "hosts.oggm, .pygem, .glogem", GREEN_FILL),
    ]
    for row, y in ((row1, top), (row2, low)):
        for x, (title, body, foot, fill) in zip(xs, row):
            s.box(ax, x, y, w, h, title, body, color=s.GREY, fill=fill, foot=foot)
        for x in xs[:-1]:
            s.arrow(ax, (x + w + 0.004, y + h / 2), (x + w + gap - 0.004, y + h / 2))
    # wrap from the end of the first row to the start of the second
    xe, xb, ym = xs[-1] + w / 2, xs[0] + w / 2, (top + low + h) / 2
    ax.plot([xe, xe, xb], [top - 0.004, ym, ym], color=s.GREY, lw=1.0)
    s.arrow(ax, (xb, ym), (xb, low + h + 0.004))
    # the calibrated constant enters the law
    cx = xs[2] + w / 2
    s.arrow(ax, (cx, 0.055), (cx, low - 0.004), color=s.ORANGE)
    ax.text(cx + 0.008, 0.03, "$k_0$ calibrated on observed frontal ablation "
            "(calibration.static, calibration.dynamic)", fontsize=s.MIN_PT,
            va="center", ha="center")
    for i, (fill, label) in enumerate(((SKY_FILL, "ocean forcing"),
                                        (ORANGE_FILL, "laws"),
                                        (GREEN_FILL, "host model"))):
        x = (0.012, 0.14, 0.215)[i]
        ax.add_patch(s.FancyBboxPatch((x, 0.965), 0.018, 0.022, fc=fill, ec=s.GREY,
                                      lw=0.6, boxstyle="square,pad=0"))
        ax.text(x + 0.024, 0.976, label, fontsize=s.MIN_PT, va="center")
    return s.save(fig, "overview")


def front():
    fig = s.plt.figure(figsize=(s.FULL, 3.3))
    ax = fig.add_axes([0.085, 0.12, 0.585, 0.8])
    sl, d, free = 0.0, 90.0, 25.0
    h = physics.front_thickness(sl + free, d, sl)
    x_ice = np.linspace(-3000, 0, 60)
    surf = sl + free + 0.045 * (-x_ice) ** 0.95
    bed_ice = -d + 0.01 * x_ice
    x_sea = np.linspace(0, 2000, 40)
    bed_sea = -d - 470 * (1 - np.exp(-x_sea / 600))
    ax.fill_between(x_sea, bed_sea, sl, color=s.WATER, lw=0)
    ax.fill_between(x_ice, bed_ice, surf, color=s.ICE, ec=s.GREY, lw=0.8)
    xb = np.r_[x_ice, x_sea]
    zb = np.r_[bed_ice, bed_sea]
    ax.fill_between(xb, -700, zb, color=s.BED, lw=0)
    ax.plot(xb, zb, color=s.BLACK, lw=0.8)
    ax.axhline(sl, color=s.BLUE, lw=0.8, ls=(0, (4, 2)))
    ax.text(1990, sl + 6, "sea level $z_{sl}$", ha="right", va="bottom",
            fontsize=s.MIN_PT, color=s.BLUE)
    bands, _ = DEFAULTS.depth_bands()
    shade = {"shallow_0_60": s.ORANGE, "ismip6": s.PURPLE}
    for name, top, bottom in bands:
        if name == "front":
            continue
        x0 = 880 if name == "shallow_0_60" else 1420
        ax.add_patch(s.plt.Rectangle((x0, sl - bottom), 500, bottom - top,
                                     fc=shade[name], alpha=0.35, ec=shade[name]))
        # a thin band is labelled under itself
        ax.text(x0 + 250, sl - (top + 15 if bottom - top > 150 else bottom + 8),
                f"{name}\n{top:g}-{bottom:g} m", ha="center", va="top", fontsize=s.MIN_PT)
    ax.add_patch(s.plt.Rectangle((40, sl - d), 300, d, fc=s.VERMILION, alpha=0.25,
                                 ec=s.VERMILION))
    ax.text(190, sl - d / 2, "front\n0 to $d$", ha="center", va="center",
            fontsize=s.MIN_PT)

    def span(x, z0, z1, label, side="left", color=s.BLACK, at=None):
        ax.annotate("", (x, z1), (x, z0),
                    arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color=color,
                                    mutation_scale=7, shrinkA=0, shrinkB=0))
        if at:
            ax.annotate(label, (x, (z0 + z1) / 2), at, ha="right", va="center",
                        fontsize=s.BASE_PT, color=color,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=color,
                                        shrinkA=1, shrinkB=2))
            return
        ax.text(x + (-25 if side == "left" else 25), (z0 + z1) / 2, label,
                ha="right" if side == "left" else "left", va="center",
                fontsize=s.BASE_PT, color=color)

    span(-60, sl - d, sl - d + h, "$h$", color=s.BLACK)
    span(-300, sl, sl + free, "$s - z_{sl}$", color=s.GREY, at=(-520, 110))
    span(-300, sl - d, sl, "$d = z_{sl} - b$", color=s.GREY)
    ax.text(-1500, -d - 30, "bed $b$", fontsize=s.MIN_PT, va="top")
    ax.text(-1700, 60, "ice", fontsize=s.BASE_PT, color=s.BLUE)
    ax.text(520, -260, "ocean", fontsize=s.BASE_PT, color=s.BLUE)
    ax.set_xlim(-3000, 2000)
    ax.set_ylim(-600, 180)
    ax.set_xlabel("distance from the calving front, m")
    ax.set_ylabel("elevation, m")
    ax.set_title("Side view (illustrative geometry)", loc="left")

    side = fig.add_axes([0.7, 0.05, 0.29, 0.92])
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.axis("off")
    side.text(0, 0.99, "Plan view of the front", fontsize=s.BASE_PT + 1, va="top")
    side.fill([0.05, 0.42, 0.42, 0.05], [0.62, 0.66, 0.86, 0.9], color=s.ICE,
              ec=s.GREY, lw=0.8)
    side.fill([0.42, 0.95, 0.95, 0.42], [0.6, 0.6, 0.92, 0.92], color=s.WATER, lw=0)
    side.plot([0.42, 0.42], [0.66, 0.86], color=s.VERMILION, lw=2)
    side.annotate("", (0.47, 0.86), (0.47, 0.66),
                  arrowprops=dict(arrowstyle="<|-|>", lw=0.8, mutation_scale=7,
                                  shrinkA=0, shrinkB=0))
    side.text(0.5, 0.76, "$w$, front width", va="center", fontsize=s.BASE_PT)
    side.text(0, 0.5, "Where each quantity enters", fontsize=s.BASE_PT, va="top",
              fontweight="bold")
    side.text(0, 0.43,
              "constant:  $Q = k\\,d\\,h\\,w$\n"
              "tf_power:  $k = k_0\\,F(TF)$\n"
              "$h = \\max(s - z_{sl}, 0) + d$\n"
              "$TF$: mean over the depth range,\n"
              "     e.g. shallow_0_60 (the default)\n"
              "melt laws: $\\dot m$ from $d$, discharge and $TF$\n"
              "$Q = 0$ when $d \\leq 0$ or $h \\leq 0$",
              fontsize=s.MIN_PT, va="top", linespacing=1.6)
    return s.save(fig, "calving-front")


def nested_laws():
    fig, ax = s.canvas(s.FULL, 3.3)
    w, h = 0.235, 0.2
    pos = {"constant": (0.015, 0.40), "tf_power": (0.015, 0.74),
           "larger_of": (0.015, 0.04), "melt_calving": (0.33, 0.40),
           "sea_ice": (0.33, 0.74)}
    eq = {"constant": "$Q = k\\,d\\,h\\,w$",
          "tf_power": "$k = k_0\\,((1 - a) + a\\,F)$\n$F = (\\max(TF, 0)/TF_{ref})^{\\gamma}$",
          "melt_calving": "$Q = w\\,d\\,(k_c\\,h + \\lambda\\,\\dot m)$",
          "sea_ice": "melt term $\\times\\,f_{ow}^{\\,\\delta}$\n"
                     "$f_{ow}$: open-water fraction",
          "larger_of": "$Q = w\\,d\\,\\max(k_c\\,h, \\lambda\\,\\dot m)$"}
    color = {"constant": s.GREY, "tf_power": s.ORANGE, "melt_calving": s.BLUE,
             "sea_ice": s.SKY, "larger_of": s.GREEN}
    for name, (x, y) in pos.items():
        s.box(ax, x, y, w, h, name, eq[name], color=color[name],
              fill=s.PAPER, body_size=s.BASE_PT)
    cx = pos["constant"][0] + w / 2

    def down(a, b, label, dx=0.01):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        s.arrow(ax, (xa + w / 2, ya - 0.004), (xb + w / 2, yb + h + 0.004),
                text=label, text_xy=(xa + w / 2 + dx, (ya + yb + h) / 2),
                va="center")

    down("tf_power", "constant", "$TF = TF_{ref}$, $\\gamma = 0$ or $a = 0$")
    down("sea_ice", "melt_calving", "$\\delta = 0$")
    s.arrow(ax, (cx, pos["larger_of"][1] + h + 0.004),
            (cx, pos["constant"][1] - 0.004), text="$\\lambda = 0$",
            text_xy=(cx + 0.01, 0.32), va="center")
    s.arrow(ax, (pos["melt_calving"][0] - 0.004, 0.5), (cx + w / 2 + 0.004, 0.5),
            text="$\\lambda = 0$", text_xy=(0.29, 0.52), ha="center")
    ax.text(0.33, 0.3,
            "$\\dot m = (A\\,d\\,q^{\\alpha} + B)\\,TF^{\\beta}$\n"
            "submarine melt rate,\nRignot et al. (2016)",
            fontsize=s.MIN_PT, va="top", linespacing=1.5)
    ax.text(0.015, 0.985, "Each arrow: the law at its tail equals the law at its head for the values on it",
            fontsize=s.MIN_PT, va="top", color=s.GREY)

    ax2 = fig.add_axes([0.68, 0.15, 0.3, 0.72])
    tf = np.linspace(-0.5, 4.0, 361)
    tf_ref = 1.5
    for gamma, col in ((0.0, s.GREY), (1.0, s.SKY), (DEFAULTS.calving_tf_exponent,
                                                     s.ORANGE), (2.0, s.VERMILION)):
        ax2.plot(tf, laws.ocean_factor(tf, tf_ref, gamma), color=col, lw=1.2,
                 label=f"$\\gamma$ = {gamma:g}")
    ax2.plot(tf, laws.tf_scaling(tf, tf_ref, DEFAULTS.calving_tf_exponent, 0.5),
             color=s.ORANGE, lw=1.0, ls="--",
             label=f"$\\gamma$ = {DEFAULTS.calving_tf_exponent:g}, $a$ = 0.5")
    ax2.plot([tf_ref, tf_ref], [0, 1], color=s.GREY, lw=0.6, ls=":")
    ax2.axhline(1, color=s.GREY, lw=0.6, ls=":")
    ax2.set_xlabel("thermal forcing $TF$, °C")
    ax2.set_ylabel("$k / k_0$ of tf_power")
    ax2.set_title(f"tf_power at $TF_{{ref}}$ = {tf_ref:g} °C", loc="left")
    ax2.set_ylim(0, 3.2)
    ax2.legend(loc="upper left")
    return s.save(fig, "nested-laws")


def contract():
    fig, ax = s.canvas(s.FULL, 2.6)
    gives = ("ice thickness at the terminus, $h$ (m)\n"
             "water depth at the terminus, $d$ (m)\n"
             "width of the front, $w$ (m)\n"
             "time, in its own model years (float years)\n"
             "glacier id (any inventory)\n"
             "terminus longitude and latitude (degrees)")
    gets = ("frontal ablation flux\n(m$^3$ ice s$^{-1}$, or m$^3$ ice yr$^{-1}$)\n\n"
            "or the dimensionless ocean factor $F$,\nto multiply its own calving constant")
    s.box(ax, 0.01, 0.46, 0.36, 0.52, "The host supplies", gives, color=s.GREEN,
          fill=GREEN_FILL, align="left")
    s.box(ax, 0.45, 0.52, 0.18, 0.4, "tidewater", "a law, built once per\nglacier from its\n"
          "ocean forcing, or a\nfactor table", color=s.ORANGE, fill=ORANGE_FILL)
    s.box(ax, 0.71, 0.46, 0.28, 0.52, "The host receives", gets, color=s.GREEN,
          fill=GREEN_FILL, align="left")
    s.arrow(ax, (0.372, 0.72), (0.446, 0.72), text="each step",
            text_xy=(0.409, 0.76), ha="center")
    s.arrow(ax, (0.634, 0.72), (0.706, 0.72), text="$Q$ or $F$",
            text_xy=(0.67, 0.76), ha="center")
    ax.text(0.01, 0.3, "Stays with the host: capping the flux by the available ice, "
            "removing the mass, moving the front,\nand the accounting of the removed "
            "volume.", fontsize=s.MIN_PT, va="center")
    ax.text(0.01, 0.12, "One front per glacier: a host with several marine fronts calls "
            "the law once per front and adds the fluxes.\nA yearly factor is the mean of "
            "the monthly factors, not the factor of the yearly mean.",
            fontsize=s.MIN_PT, va="center")
    return s.save(fig, "host-contract")


def routes():
    fig, ax = s.canvas(s.FULL, 4.6)
    w, h, gap = 0.315, 0.155, 0.03
    cols = {
        "OGGM": (s.BLUE, [
            ("set up", "cfg.initialize()\ntidewater.hosts.oggm.register()"),
            ("ocean forcing", "tasks.process_ocean_data(gdir, ...)"),
            ("calibrate $k$", "tasks.run_dynamic_calving_k_calibration(\n"
             "    gdir, ref_fa=..., ref_fa_err=...)"),
            ("run", "tasks.run_with_ocean_forcing(\n    gdir, calving_law='tf_power')"),
            ("in the model", "the law is the calving_law of the\n"
             "flowline model, called at every step"),
        ], "branch oggm-tidewater: four generic hooks"),
        "PyGEM": (s.ORANGE, [
            ("laws", "laws_from_forcing(\n    {'1.03622': path}, 'tf_power')"),
            ("run", "with ocean_dynamics(laws):\n    run_simulation.main()"),
            ("route", "hook calving_law_for (branch), or a\n"
             "runtime swap of the two dynamics\nclasses (PyGEM 1.1.0); same numbers"),
            ("in the model", "OGGM dynamics: the law at every step\n"
             "mass redistribution: one constant\nper model year (YearlyCalving)"),
        ], "branch pygem-tidewater: one hook"),
        "GloGEM": (s.GREEN, [
            ("tables, in Python", "glogem.write_ocean_factor_tables(\n"
             "    dir, tf_monthly, tf_ref)"),
            ("files", "ocean_factor_<region>.dat: one factor per\n"
             "glacier and model year; the IDL reader is\nin tidewater/hosts/glogem/idl"),
            ("config.pro", "!PATH = !PATH + ':<idl dir>'\n"
             "calving_factor_function = 'tw_table_factor'\n"
             "calving_factor_dir = '<tables>'"),
            ("calving_model.pro", "GloGEM's own flux, times\n"
             "call_function(calving_factor_function, ...)"),
        ], "branch glogem-tidewater: one hook, default off"),
    }
    for i, (host, (col, steps, branch)) in enumerate(cols.items()):
        x = 0.01 + i * (w + 0.02)
        ax.text(x + w / 2, 0.99, host, ha="center", va="top", fontsize=s.BASE_PT + 2,
                fontweight="bold", color=col)
        for j, (title, body) in enumerate(steps):
            y = 0.77 - j * (h + gap)
            s.box(ax, x, y, w, h, title, body, color=col, align="left")
            if j:
                s.arrow(ax, (x + w / 2, y + h + gap - 0.004), (x + w / 2, y + h + 0.004),
                        color=col)
        ax.text(x + w / 2, 0.945, branch, ha="center", va="center", fontsize=s.MIN_PT,
                color=s.GREY)
    return s.save(fig, "coupling-routes")


def model_years():
    time = pd.date_range("1999-10-01", "2002-09-01", freq="MS")
    phase = 2 * np.pi * (time.month - 3) / 12
    tf = pd.Series(1.6 - 1.1 * np.cos(phase) + 0.25 * (time.year - 2000), index=time)
    tf_ref = laws.tf_power_mean(tf.loc["2000"].to_numpy(), DEFAULTS.calving_tf_exponent)
    mid = time + pd.to_timedelta(time.days_in_month / 2, unit="D")
    factor = laws.ocean_factor(tf.to_numpy(), tf_ref)

    fig, axes = s.plt.subplots(3, 1, figsize=(s.FULL, 3.6), sharex=True,
                               gridspec_kw={"height_ratios": [1.2, 1, 1]})
    fig.subplots_adjust(left=0.08, right=0.71, top=0.93, bottom=0.1, hspace=0.38)
    ax = axes[0]
    ax.plot(mid, tf, "o", ms=3, color=s.BLUE, label="monthly value,\nat mid-month")
    ax.plot(mid, tf, color=s.BLUE, lw=0.8, label="what a law reads\nat model time $t$")
    ax.axhline(tf_ref, color=s.GREY, ls=":", lw=0.8)
    ax.text(1.005, tf_ref, "$TF_{ref}$", transform=ax.get_yaxis_transform(),
            va="center", fontsize=s.MIN_PT, color=s.GREY)
    ax.set_ylabel("$TF$, °C")
    ax.set_title("Monthly series (illustrative); OGGM, and PyGEM with OGGM dynamics, "
                 "read it at time $t$", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.09, 1.05))

    def years(ax, start, label, color, title):
        yearly = timeconv.yearly_ocean_factor(tf, tf_ref, start_month=start,
                                              label=label, incomplete="drop")
        ax.plot(mid, factor, "o", ms=2.5, color=s.GREY, label="monthly factor $F$")
        for year, value in yearly.items():
            y0, y1 = timeconv.model_year_span(year, start, label)
            a = pd.Timestamp(year=y0[0], month=y0[1], day=1)
            b = pd.Timestamp(year=y1[0], month=y1[1], day=1) + pd.offsets.MonthEnd(1)
            ax.plot([a, b], [value, value], color=color, lw=2.2, solid_capstyle="butt")
            ax.axvline(a, color=s.LIGHT, lw=0.8, zorder=0)
            ax.text(a + (b - a) / 2, 0.02, f"model year {year}", transform=
                    ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=s.MIN_PT, color=color,
                    bbox=dict(fc=s.PAPER, ec="none", pad=0.6))
        ax.plot([], [], color=color, lw=2.2, label="mean of the year's\nmonthly factors")
        ax.set_ylabel("$F$")
        # room under the data for the year labels
        top = 1.1 * factor.max()
        ax.set_ylim(-0.3 * top, top)
        ax.set_yticks([0, 1, 2])
        ax.spines["left"].set_bounds(0, top)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(1.09, 1.05))

    from tidewater.hosts.glogem.tables import LABEL as G_LABEL, start_month as g_start
    from tidewater.hosts.pygem.glaciers import LABEL as P_LABEL, START_MONTH as P_START
    years(axes[1], P_START, P_LABEL, s.ORANGE,
          "PyGEM, mass redistribution: calendar years")
    years(axes[2], g_start(5), G_LABEL, s.GREEN,
          "GloGEM: October to September, named by the year it ends (RGI region 05)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    return s.save(fig, "model-years")


SKY_FILL = "#EAF4FB"
ORANGE_FILL = "#FDF2DE"
GREEN_FILL = "#E3F4EE"


FIGURES = {"overview": overview, "calving-front": front, "nested-laws": nested_laws,
           "host-contract": contract, "coupling-routes": routes,
           "model-years": model_years}


def main(names=None):
    for name in names or FIGURES:
        print(FIGURES[name]())


if __name__ == "__main__":
    main(sys.argv[1:])
