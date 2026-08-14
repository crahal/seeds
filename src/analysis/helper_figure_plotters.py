import ast
import re
import os
import math
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas_datareader import data as pdr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker


def _entropy_bits_from_counts(counts: pd.Series) -> float:
    """Shannon entropy in bits for a discrete distribution given as counts."""
    c = counts.astype(float).to_numpy()
    s = c.sum()
    if s <= 0:
        return float("nan")
    p = c / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def plot_llms(fpath_wseeds, fpath_wfixedseeds, shuffled):
    # ---------------------------------------------------------------------
    # Global style
    # ---------------------------------------------------------------------
    mpl.rcParams["font.family"] = "Helvetica"
    mpl.rcParams["axes.unicode_minus"] = False

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    df = pd.read_csv(fpath_wseeds)
    df_fixed = pd.read_csv(fpath_wfixedseeds)

    print("\n===================== SAMPLE SIZE / SEED TRIALS =====================")

    def _describe_samples(dataframe, name):
        total_rows = len(dataframe)
        has_temp = "temperature" in dataframe.columns
        has_seed = "seed" in dataframe.columns
        unique_temps = sorted(dataframe["temperature"].unique()) if has_temp else []
        unique_seeds = dataframe["seed"].nunique() if has_seed else None
        print(
            f"{name}: rows={total_rows:,}"
            f"{' | unique temperatures=' + str(unique_temps) if has_temp else ' | no temperature column'}"
            f"{' | unique seeds=' + str(unique_seeds) if unique_seeds is not None else ' | no seed column'}"
        )
        if has_temp:
            for t, g in dataframe.groupby("temperature"):
                seeds_here = g["seed"].nunique() if has_seed else None
                print(
                    f"  - temp {t}: rows={len(g):,}"
                    f"{' | unique seeds=' + str(seeds_here) if seeds_here is not None else ''}"
                )

    _describe_samples(df, f"Main file '{fpath_wseeds}' (different seeds)")
    _describe_samples(df_fixed, f"Fixed-seed file '{fpath_wfixedseeds}' (includes temp sweeps)")

    # ---------------------------------------------------------------------
    # Restrict fixed-seed to temperature 0.0 if column exists
    # ---------------------------------------------------------------------
    if "temperature" in df_fixed.columns:
        df_fixed0 = df_fixed.loc[df_fixed["temperature"] == 0.0].copy()
        if df_fixed0.empty:
            raise ValueError("df_fixed has 'temperature' but no rows with temperature==0.0.")
    else:
        df_fixed0 = df_fixed  # assume file is already fixed-seed @ 0.0
    _describe_samples(df_fixed0, "Fixed-seed @ temperature 0.0 subset used for leftmost column")

    # ---------------------------------------------------------------------
    # WITHIN fixed-seed @ 0.0: divergence from main category (per panel)
    # ---------------------------------------------------------------------
    panel_cols = [
        "rep_to_rep_label",
        "dem_to_rep_label",
        "rep_to_dem_label",
        "dem_to_dem_label",
    ]

    print("\nWithin fixed-seed @ 0.0: concentration over class labels\n")
    fixed_seed_unique_seeds = df_fixed0["seed"].nunique() if "seed" in df_fixed0.columns else None
    for col in panel_cols:
        if col not in df_fixed0.columns:
            raise ValueError(f"Expected column '{col}' in df_fixed (or df_fixed0).")

        counts = df_fixed0[col].value_counts(dropna=False)
        total = counts.sum()
        if total == 0:
            print(f"{col:>18s} : empty")
            continue

        pmax = float(counts.max() / total)
        main_label = counts.idxmax()
        divergence_from_main = 1.0 - pmax  # mass not on the modal label

        H = _entropy_bits_from_counts(counts)
        perplexity = float(2.0 ** H) if np.isfinite(H) else float("nan")

        print(
            f"{col:>18s} : n_rows={total:,}"
            f"{' | unique seeds=' + str(fixed_seed_unique_seeds) if fixed_seed_unique_seeds is not None else ''}"
            f" | main='{main_label}' | p_max={pmax:.6f} | "
            f"1-p_max={divergence_from_main:.6f} | H={H:.6f} bits | perp={perplexity:.3f}"
        )

    # ---- rows and titles ----
    rows = [
        ["rep_to_rep_label", "dem_to_rep_label"],   # row 0: describing Republicans
        ["rep_to_dem_label", "dem_to_dem_label"],   # row 1: describing Democrats
    ]

    titles = [
        ["Rep → Rep", "Dem → Rep"],
        ["Rep → Dem", "Dem → Dem"],
    ]

    # colours per panel (regal palette)
    colors = [
        "#d53e4f",
        "#d53e4f",
        "#3288bd",
        "#3288bd",
    ]

    # ---------------------------------------------------------------------
    # Per-row vocabularies ordered by frequency (combined)
    # (use df_fixed0 so "fixed seed @ 0.0" is what defines the left column)
    # ---------------------------------------------------------------------
    words_row0 = (
        pd.concat(
            [
                df[["rep_to_rep_label", "dem_to_rep_label"]].melt(value_name="word"),
                df_fixed0[["rep_to_rep_label", "dem_to_rep_label"]].melt(value_name="word"),
            ],
            ignore_index=True,
        )["word"]
        .value_counts()
        .index
        .tolist()
    )

    words_row1 = (
        pd.concat(
            [
                df[["rep_to_dem_label", "dem_to_dem_label"]].melt(value_name="word"),
                df_fixed0[["rep_to_dem_label", "dem_to_dem_label"]].melt(value_name="word"),
            ],
            ignore_index=True,
        )["word"]
        .value_counts()
        .index
        .tolist()
    )

    row_words = [words_row0, words_row1]

    # ---------------------------------------------------------------------
    # Modal-share stability across temperatures (per panel)
    # ---------------------------------------------------------------------
    def _modal_share_stats(label):
        print(f"\nModal-share stability for '{label}' across temperatures:")
        vals = []
        for t, g in df.groupby("temperature"):
            counts = g[label].value_counts(dropna=False)
            total = counts.sum()
            if total == 0:
                continue
            seeds_here = g["seed"].nunique() if "seed" in g.columns else None
            print(
                f"  temp={t}: rows={len(g):,}"
                f"{' | unique seeds=' + str(seeds_here) if seeds_here is not None else ''}"
                f" | label observations={total:,}"
            )
            vals.append(float(counts.max() / total))
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            print(f"{label}: no data for modal-share stats")
            return
        mean = float(np.mean(vals))
        amin = float(np.min(vals))
        amax = float(np.max(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
        z = mean / sd if sd not in (0, float("nan")) else float("nan")
        print(f"{label}: modal-share p_max across temps -> n={vals.size}, min={amin:.4f}, max={amax:.4f}, mean={mean:.4f}, sd={sd:.4f}, z={z:.4f}")

    for col in panel_cols:
        _modal_share_stats(col)

    # ---------------------------------------------------------------------
    # Temperatures + special fixed-seed column
    # ---------------------------------------------------------------------
    temps_main = sorted(df["temperature"].unique())

    if len(temps_main) >= 2:
        spacing = min(b - a for a, b in zip(temps_main, temps_main[1:]))
    else:
        spacing = 0.25

    fixed_temp_x = temps_main[0] - spacing
    temps_all = [fixed_temp_x] + temps_main

    # ---------------------------------------------------------------------
    # Figure + layout
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1],
        width_ratios=[1, 1],
        wspace=0.1,
        hspace=0.20,
    )

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    axes = ((ax11, ax12), (ax21, ax22))
    all_axes = [ax11, ax12, ax21, ax22]

    panel_info = [
        (ax11, rows[0][0], titles[0][0], colors[0], row_words[0]),
        (ax12, rows[0][1], titles[0][1], colors[1], row_words[0]),
        (ax21, rows[1][0], titles[1][0], colors[2], row_words[1]),
        (ax22, rows[1][1], titles[1][1], colors[3], row_words[1]),
    ]

    # ---------------------------------------------------------------------
    # Scatter panels
    # ---------------------------------------------------------------------
    for ax, col, title, color, words in panel_info:
        word_to_y = {w: i for i, w in enumerate(words, start=1)}

        counts_main = (
            df.groupby(["temperature", col])
            .size()
            .reset_index(name="count")
        )

        counts_fixed = (
            df_fixed0.groupby(col)
            .size()
            .reset_index(name="count")
        )
        counts_fixed["temperature"] = fixed_temp_x

        counts = pd.concat([counts_main, counts_fixed], ignore_index=True)
        counts["y_pos"] = counts[col].map(word_to_y)

        ax.set_axisbelow(True)

        max_count = counts["count"].max()
        size_max = 1200.0
        sizes = ((counts["count"] / max_count) ** 1) * size_max

        ax.scatter(
            counts["temperature"],
            counts["y_pos"],
            s=sizes,
            linewidths=1,
            color=color,
            edgecolor="black",
            zorder=3,
            clip_on=False,
        )

        for side in ["left", "bottom", "right", "top"]:
            ax.spines[side].set_position(("outward", 4))

        ax.set_ylim(-0.15, len(words) + 0.5)
        ax.set_xlim(min(temps_all) - 0.1, max(temps_all) + 0.1)

    # ---------------------------------------------------------------------
    # y-ticks
    # ---------------------------------------------------------------------
    ax11.set_yticks(range(1, len(words_row0) + 1))
    ax11.set_yticklabels(words_row0, fontsize=11)
    ax21.set_yticks(range(1, len(words_row1) + 1))
    ax21.set_yticklabels(words_row1, fontsize=11)

    ax12.set_yticks(range(1, len(words_row0) + 1)); ax12.set_yticklabels([])
    ax22.set_yticks(range(1, len(words_row1) + 1)); ax22.set_yticklabels([])

    for ax in [ax11, ax21]:
        ax.tick_params(axis="y", pad=6)

    # ---------------------------------------------------------------------
    # x-axis ticks / labels
    # ---------------------------------------------------------------------
    xticklabels_main = [f"Temperature {t}\n1000 different seeds" for t in temps_main]
    xticklabels_all = ["Temperature 0.0\n1000 same seeds"] + xticklabels_main

    for ax in all_axes:
        ax.set_xticks(temps_all)

    ax11.set_xticklabels([])
    ax12.set_xticklabels([])

    ax21.set_xticklabels(xticklabels_all, fontsize=9)
    ax22.set_xticklabels(xticklabels_all, fontsize=9)

    for ax in [ax21, ax22]:
        ax.tick_params(axis="x", pad=5)
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_verticalalignment("top")
            label.set_horizontalalignment("center")

    # ---------------------------------------------------------------------
    # Titles
    # ---------------------------------------------------------------------
    axes[0][0].set_title("Described by Republicans", fontsize=16, y=1.025)
    axes[0][1].set_title("Described by Democrats", fontsize=16, y=1.025)

    axes[0][1].yaxis.set_label_position("right")
    axes[0][1].set_ylabel("Describing Republicans", fontsize=16, rotation=270, labelpad=20)

    axes[1][1].yaxis.set_label_position("right")
    axes[1][1].set_ylabel("Describing Democrats", fontsize=16, rotation=270, labelpad=20)

    axes[0][0].set_title('a.', loc='left', fontsize=16, y=1.025, fontweight='bold')
    axes[0][1].set_title('b.', loc='left', fontsize=16, y=1.025, fontweight='bold')
    axes[1][0].set_title('c.', loc='left', fontsize=16, y=1.025, fontweight='bold')
    axes[1][1].set_title('d.', loc='left', fontsize=16, y=1.025, fontweight='bold')

    # ---------------------------------------------------------------------
    # Export decorations
    # ---------------------------------------------------------------------
    first_temp = temps_main[0]
    boundary_x = fixed_temp_x + (first_temp - fixed_temp_x) / 2

    for ax in all_axes:
        ax.axvline(
            boundary_x,
            linestyle="--",
            color="black",
            linewidth=1,
            alpha=0.7,
            zorder=1
        )

    figure_path = os.path.join(os.getcwd(), "..", "figures")
    os.makedirs(figure_path, exist_ok=True)

    import seaborn as sns
    sns.despine(ax=axes[0][0], top=True, right=True)
    sns.despine(ax=axes[0][1], top=True, right=True)
    sns.despine(ax=axes[1][0], top=True, right=True)
    sns.despine(ax=axes[1][1], top=True, right=True)

    plt.savefig(os.path.join(figure_path, f"plot_llms_{shuffled}.pdf"), bbox_inches="tight")



def plot_three_simple_examples(figure_path,
                               figsize=(8.5, 17),
                               colors=['#001c54', '#E89818', '#8b0000'],
                               fill_color=(254 / 255, 208 / 255, 126 / 255, 10 / 255),
                               ):
    k_formatter = ticker.FuncFormatter(lambda x, pos: f'{x / 1000:g}k')
    df_sir = pd.read_csv(os.path.join(os.getcwd(),
                                      '..',
                                      'data',
                                      'sir',
                                      'sir_seeds_1dp.csv')
                         )
    # Basic stats for infection rate at timestep 250 (as a fraction of population 1,000)
    timestep_idx = 250
    infection_cols = ['Infected_min', 'Infected_med', 'Infected_max']
    if timestep_idx < len(df_sir):
        infection_counts = df_sir.loc[timestep_idx, infection_cols].astype(float)
        infection_rates = infection_counts / 1000.0
        infection_rate_stats = {
            'min': float(infection_rates.min()),
            'max': float(infection_rates.max()),
            'mean': float(infection_rates.mean()),
            'std': float(infection_rates.std(ddof=0)),
        }
        infection_rate_zscore = ((infection_rates['Infected_med'] - infection_rate_stats['mean'])
                                 / infection_rate_stats['std']
                                 if infection_rate_stats['std'] else np.nan)
        print(
            "Infection rate at timestep 250 (fraction of population): "
            f"min={infection_rate_stats['min']:.4f}, "
            f"max={infection_rate_stats['max']:.4f}, "
            f"mean={infection_rate_stats['mean']:.4f}, "
            f"std={infection_rate_stats['std']:.4f}, "
            f"z-score={infection_rate_zscore:.4f}"
        )
    df_buffon = pd.read_csv(os.path.join(os.getcwd(),
                                         '..',
                                         'data',
                                         'needles',
                                         'results',
                                         'throw100_25000_5000seeds.csv'),
                            names=['Throws', 'Min', '25th_PC',
                                   'Median', '75th_PC', 'Max']
                            )
    # Needle (Buffon's) stats at specific throw counts using the wide sweep file (1000-40000 throws)
    df_buffon_stats = pd.read_csv(os.path.join(os.getcwd(),
                                               '..',
                                               'data',
                                               'needles',
                                               'results',
                                               'throw1000_40000_5000seeds.csv'),
                                  names=['Throws', 'Min', '25th_PC', 'Median', '75th_PC', 'Max']
                                  )

    def _needle_stats(target_throws):
        # Use the closest available row if the exact throw count is absent (file is in steps of 10)
        idx = (df_buffon_stats['Throws'] - target_throws).abs().idxmin()
        row = df_buffon_stats.loc[idx]
        values = row[['Min', 'Median', 'Max']].astype(float)
        stats = {
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
            'std': float(values.std(ddof=0)),
        }
        zscore = ((row['Median'] - stats['mean']) / stats['std']
                  if stats['std'] else np.nan)
        print(
            f"Needle estimate at ~{int(target_throws):,} throws "
            f"(using {int(row['Throws']):,}): "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"z-score={zscore:.4f}"
        )

    for throws_target in (1000, 40000):
        _needle_stats(throws_target)

    df_collisions = pd.read_csv(os.path.join(os.getcwd(),
                                             '..',
                                             'data',
                                             'collisions',
                                             'stats_32bit_rowwise.csv'
                                             )
                                )
    df_collisions_finalrow = pd.read_csv(os.path.join(os.getcwd(),
                                                      '..',
                                                      'data',
                                                      'collisions',
                                                      'stats_32_final_row.csv')
                                         )
    # Hofert collisions stats at ~1,000,000 draws (full final-row sample)
    if not df_collisions_finalrow.empty:
        collisions_vals = df_collisions_finalrow['x'].astype(float)
        coll_stats = {
            'min': float(collisions_vals.min()),
            'max': float(collisions_vals.max()),
            'mean': float(collisions_vals.mean()),
            'std': float(collisions_vals.std(ddof=0)),
        }
        coll_median = float(collisions_vals.median())
        coll_z = ((coll_median - coll_stats['mean']) / coll_stats['std']
                  if coll_stats['std'] else np.nan)
        print(
            "Hofert collisions at 1,000,000 draws: "
            f"min={coll_stats['min']:.4f}, max={coll_stats['max']:.4f}, "
            f"mean={coll_stats['mean']:.4f}, std={coll_stats['std']:.4f}, "
            f"median={coll_median:.4f}, z-score={coll_z:.4f}"
        )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    letter_fontsize = 24
    label_fontsize = 18
    nbins = 14

    #################
    # Figure 1a here#
    #################
    df_sir['Infected_min'].plot(ax=ax1, color=colors[1], linestyle='-')
    df_sir['Infected_med'].plot(ax=ax1, color=colors[0])
    df_sir['Infected_max'].plot(ax=ax1, color=colors[2])
    ax1.fill_between(df_sir.index, df_sir['Infected_min'], df_sir['Infected_max'],
                     color=fill_color)
    ax1.set_xlim(-25, 1450)
    ax1.set_xlabel('Time', fontsize=17)
    ax1.set_ylabel(r'Fraction Infected', fontsize=17)
    legend_elements1 = [
        Line2D([0], [0], color=colors[2], lw=2, linestyle='--',
               label=r'Max'),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='--',
               label=r'Min'),
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'Median'),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Variance')]
    ax1.legend(handles=legend_elements1, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
               )
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 10:.0f}%'))

    #################
    # Figure 1b here#
    #################
    df_buffon = df_buffon.set_index('Throws')
    df_buffon = df_buffon[45:]
    ax2.plot(df_buffon['Min'], color=colors[1], alpha=0.8, linestyle='--')
    ax2.plot(df_buffon['Max'], color=colors[2], alpha=0.8, linestyle='--')
    ax2.set_xlim(0, df_buffon.index[-1] + 500)
    ax2.set_ylim(2.225, 4.5)
    ax2.hlines(math.pi, df_buffon.index[0] + 500, df_buffon.index[-1],
               color=colors[0], linestyle='-')
    ax2.fill_between(df_buffon.index,
                     df_buffon['Min'],
                     df_buffon['Max'],
                     color=fill_color)
    ax2.set_xlabel('Number of Throws', fontsize=17)
    ax2.set_ylabel(r'Estimate of $\mathrm{\pi}$', fontsize=17)
    ax2.tick_params(axis='both', which='major', labelsize=14,
                    width=1, length=8)
    ax2.xaxis.set_major_formatter(k_formatter)
    legend_elements2 = [
        Line2D([0], [0], color=colors[2], linestyle='--',
               label=r'Max', lw=2),
        Line2D([0], [0], color=colors[1], linestyle='--',
               label=r'Min', lw=2),
        Line2D([0], [0], color=colors[0], linestyle='-',
               label=r'$\mathrm{\pi}$', lw=2),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Variance')
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
               )
    #################
    # Figure 1c here#
    #################

    df_collisions['min'].drop_duplicates().plot(color=colors[1], linestyle='--', ax=ax3)
    df_collisions['median'].drop_duplicates().plot(color=colors[0], linestyle='-', ax=ax3)
    df_collisions['max'].drop_duplicates().plot(color=colors[2], linestyle='--', ax=ax3)

    ax3.fill_between(df_collisions.drop_duplicates().index,
                     df_collisions.drop_duplicates()['min'],
                     df_collisions.drop_duplicates()['max'],
                     color=fill_color)

    legend_elements2 = [
        Line2D([0], [0], color=colors[2], lw=2, linestyle='--',
               label=r'Max'),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='--',
               label=r'Min'),
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'Median'),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Variance')]
    ax3.legend(handles=legend_elements2, loc='lower right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1
               )
    ax3.set_xlim(0, 1000000)
    ax3.xaxis.set_major_formatter(k_formatter)
    ax3.set_ylabel('Number of 32-bit collisions', fontsize=17)
    ax3.set_xlabel('Sample size', fontsize=17)

    ax3_inset = ax3.inset_axes([0.035, 0.535, 0.375, 0.35], transform=ax3.transAxes)
    sns.histplot(df_collisions_finalrow['x'],
                 ax=ax3_inset,
                 color=colors[0],
                 bins=nbins,
                 )
    ax3_twin = ax3_inset.twinx()
    sns.kdeplot(df_collisions_finalrow['x'],
                ax=ax3_twin,
                color=colors[1],
                linestyle='-',
                linewidth=2
                )
    ax3_inset.annotate(r'$\mu$ = ' + str(np.round(np.mean(df_collisions_finalrow['x']), 1)) + r', $\sigma$ = ' +
                       str(np.round(np.std(df_collisions_finalrow['x']), 1)),
                       xy=(0.5, 1), xytext=(0.5, 1.1),
                       xycoords='axes fraction',
                       fontsize=11, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                       arrowprops=dict(arrowstyle='-[, widthB=5.0, lengthB=1',
                                       lw=1.0)
                       )

    ax3_twin.tick_params(width=1, length=8, axis='both', which='major', labelsize=14)
    ax3.set_xlabel('Number of Draws', fontsize=16)
    ax3.set_ylabel('Count of Collisions', fontsize=16)
    ax3.set_axisbelow(True)
    ax3_twin.set_ylabel('')
    ax3_inset.set_ylabel('')
    ax3_twin.set_xlabel('')
    ax3_inset.set_xlabel('')
    ax3_inset.set_yticks([])
    ax3_twin.set_yticks([])

    ##############
    # aesthetics #
    ##############

    ax1.set_title('a.', loc='left', fontsize=letter_fontsize, y=1.0, fontweight='bold')
    ax2.set_title('b.', loc='left', fontsize=letter_fontsize, y=1.0, fontweight='bold')
    ax3.set_title('c.', loc='left', fontsize=letter_fontsize, y=1.0, fontweight='bold')

    for ax in [ax1, ax2, ax3]:
        ax.grid(which="both", linestyle='--', alpha=0.225)
        ax.set_zorder(3)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=14, width=1, length=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))  # ~5 ticks excluding edges
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))

#    sns.despine(ax=ax1)
#    sns.despine(ax=ax2)
#    sns.despine(ax=ax3)
    sns.despine(ax=ax3_inset, left=True, right=True, top=True)
    sns.despine(ax=ax3_twin, left=True, right=True, top=True)

    # tight_layout struggles with inset axes; use explicit padding instead
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, wspace=0.18, hspace=0.30)
    filename = 'three_simple_examples'
    plt.savefig(os.path.join(figure_path, filename + '.pdf'),
                bbox_inches='tight')
    


def plot_new_scientometrics():
    # Scientometric visualizations for RNG corpora (Panels C, F, G, H only)
    # - RN excluded from Panel C (timelines) and Panel F (citations); included again in Panel G (overlaps).
    # - Axes named ax1..ax4. Bars in ax4 annotated and above grid (moved).
    # - Panel C (ax3) shows cross-corpus overlap as circle markers (white fill, colored edges).
    # - Panel D (ax4) shows RNG-only domain totals (deduplicated across corpora).
    # - Grid lines below all plots.
    # - ax2 uses a polished NOTCHED BOXPLOT style (bxp) with mean diamonds & global mean line.
    # - Saves output to ../figures/scientometrics

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns  # for despine
    from collections import defaultdict
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors

    # -------------------- configuration --------------------
    DATA_DIR = os.path.join(os.getcwd(), '..', 'data', 'openalex_returns')
    OUT_DIR = '../figures'
    os.makedirs(OUT_DIR, exist_ok=True)

    color_list = [
        '#4575b4',  # blue (from Spectral)
        '#E6AC00',  # gold (custom)
        '#91cf60',  # green (from Spectral)
        '#d73027',  # red (from Spectral)
    ]

    SAVE_PNG = True

    CORPORA = {
        'RN': 'openalex_rn_papers.csv',
        'RN+Quantum': 'openalex_rn_and_quantum_papers.csv',
        'RN+Hardware': 'openalex_rn_and_hardware_papers.csv',
        'RN+Pseudo': 'openalex_rn_and_pseudo_papers.csv',
        'RN+Quasi': 'openalex_rn_and_quasi_papers.csv',
    }

    # -------------------- utilities --------------------
    def _load_csv(path):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        return pd.read_csv(path, encoding='utf-8', dtype=str, keep_default_na=False, na_values=[''])

    def _ensure_year_int(df, col='publication_year'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        return df

    def _save(fig, basename):
        fig.tight_layout()
        base = os.path.join(OUT_DIR, basename)
        if SAVE_PNG:
            fig.savefig(base + '.pdf', bbox_inches='tight')

    def _clean_doi(s):
        if isinstance(s, str):
            s = s.strip().lower()
            s = s.replace('https://doi.org/', '').replace('http://doi.org/', '').replace('doi:', '')
            return s
        return None

    def _clean_title(t):
        if isinstance(t, str):
            return ' '.join(t.lower().split())
        return None

    # helper: stats for bxp with a common effective n for notch width
    def bxp_stats(groups, labels, eff_n=None, whis=(5, 95)):
        stats = []
        if eff_n is None:
            eff_n = min(len(np.asarray(g)[~np.isnan(g)]) for g in groups if len(g))
        for lab, g in zip(labels, groups):
            x = np.asarray(g, float)
            x = x[~np.isnan(x)]
            if x.size == 0:
                continue
            q1, q3 = np.percentile(x, [25, 75])
            med = np.median(x)
            whislo, whishi = np.percentile(x, whis)
            iqr = q3 - q1
            half = 1.57 * iqr / np.sqrt(eff_n)  # visible notch width
            stats.append(dict(
                label=f"{lab}\n(n={x.size:,})",
                q1=q1, q3=q3, med=med, whislo=whislo, whishi=whishi,
                cilo=med - half, cihi=med + half, fliers=[]
            ))
        return stats

    # -------------------- load corpus files --------------------
    loaded = {}
    for label, fname in CORPORA.items():
        df = _load_csv(os.path.join(DATA_DIR, fname))
        if df is None:
            continue
        df = _ensure_year_int(df, 'publication_year')
        if 'cited_by_count' in df.columns:
            df['cited_by_count'] = pd.to_numeric(df['cited_by_count'], errors='coerce')
        loaded[label] = df

    # Stable label order for consistent coloring
    label_order = [lab for lab in ['RN', 'RN+Quantum', 'RN+Hardware', 'RN+Pseudo', 'RN+Quasi'] if lab in loaded]

    # -------------------- figure & axes --------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # -------------------- ax1 (Panel a): Stacked bars (RN excluded), years >= 1960 and <= 2024 --------------------
    years = np.arange(1960, 2025)
    stack_data = []
    labels_for_stack = []
    for label in label_order:
        if label == 'RN':
            continue
        df = loaded[label]
        if 'publication_year' in df.columns:
            ts = df.dropna(subset=['publication_year']).groupby('publication_year').size()
            ts.index = ts.index.astype(int)
            ts = ts[(ts.index >= 1960) & (ts.index <= 2024)].sort_index()
            series = pd.Series(0, index=years)
            series.loc[ts.index] = ts.values
            stack_data.append(series.values)
            labels_for_stack.append(label)

    if stack_data:
        x = years
        bottom = np.zeros_like(x, dtype=float)
        for i, (lab, vals) in enumerate(zip(labels_for_stack, stack_data)):
            ax1.bar(x, vals, bottom=bottom, label=lab,
                    width=0.9, color=color_list[i % len(color_list)],
                    edgecolor='k', zorder=3, linewidth=0.5)
            bottom = bottom + vals

    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Works Per Year', fontsize=14)
    ax1.set_title('a.', loc='left', fontsize=23, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True,
               fontsize=12, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1)
    ax1.set_axisbelow(True)
    ax1.grid(which="both", linestyle='--', alpha=0.225, zorder=1)

    # -------------------- ax2 (Panel b): NOTCHED BOXPLOTS of log10 citations per corpus (RN excluded) --------------------
    groups = []
    labels_for_boxes = []
    for i, label in enumerate([l for l in label_order if l != 'RN']):  # exclude RN here
        cites = loaded[label].get('cited_by_count')
        if cites is None:
            continue
        x = pd.to_numeric(cites, errors='coerce')
        x = x[x >= 0].astype(float).to_numpy()
        if x.size:
            groups.append(np.log10(x + 1.0))
            labels_for_boxes.append(label)

    if groups:
        ax2.set_axisbelow(True)
        ax2.grid(which="both", linestyle='--', alpha=0.225, zorder=1)

        palette = [color_list[i % len(color_list)] for i, _ in enumerate(labels_for_boxes)]
        Ns = [np.sum(~np.isnan(g)) for g in groups]
        overall_mean = (float(np.nanmean(np.concatenate(groups)))
                        if any(len(g) for g in groups) else np.nan)

        stats = bxp_stats(groups, labels_for_boxes, eff_n=min(n for n in Ns if n > 0), whis=(5, 95))

        bp = ax2.bxp(
            stats, showfliers=False, shownotches=True, patch_artist=True, zorder=2.0,
            boxprops=dict(linewidth=1.2, color="k"),
            medianprops=dict(linewidth=2.0, color="k"),
            whiskerprops=dict(linewidth=1.0, color="k"),
            capprops=dict(linewidth=1.0, color="k"),
        )
        for box, c in zip(bp["boxes"], palette):
            box.set(facecolor=c, alpha=0.85, edgecolor="k")

        means = [float(np.nanmean(g)) if np.sum(~np.isnan(g)) else np.nan for g in groups]
        ax2.scatter(
            np.arange(1, len(means) + 1), means,
            marker="D", s=52, facecolor="white", edgecolor="k", linewidth=1.2,
            zorder=3.0, label="Mean"
        )

        if not np.isnan(overall_mean):
            ax2.axhline(overall_mean, linestyle="--", linewidth=1.2,
                        color="k", alpha=0.5, zorder=1.5)

        legend_handles = [
            Line2D([0], [0], color="k", lw=2, label="Median"),
            Line2D([0], [0], marker="D", markersize=6, markerfacecolor="white",
                   markeredgecolor="k", lw=0, label="Mean"),
            Line2D([0], [0], color="k", lw=1.2, linestyle="--", alpha=0.5,
                   label="Global mean"),
        ]
        ax2.legend(handles=legend_handles, loc="upper right",
                   frameon=True, facecolor="w", edgecolor=(0,0,0,1), framealpha=1, ncols=3,
                  fontsize=12)

        ax2.set_ylabel('log10(Citations + 1)', fontsize=14)
        ax2.set_title('b.', loc='left', fontsize=23, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No citation data", ha='center')

    # -------------------- ax3 (Panel c): Cross-corpus overlap as circle markers (white fill, colored edges) ----
    corpus_sets = {}
    use_titles = False
    for label in label_order:
        df = loaded[label]
        if 'doi' in df.columns:
            dois = set(filter(None, (_clean_doi(x) for x in df['doi'])))
        else:
            dois = set()
        if len(dois) < max(5, 0.1 * len(df)):
            use_titles = True
        corpus_sets[label] = dois

    if use_titles:
        corpus_sets = {}
        for label in label_order:
            df = loaded[label]
            titles = set()
            if 'display_name' in df.columns:
                for t in df['display_name']:
                    tt = _clean_title(t)
                    if tt:
                        titles.add(tt)
            corpus_sets[label] = titles

    labels_heat = list(corpus_sets.keys())
    nL = len(labels_heat)

    def _jaccard(A, B):
        U = len(A | B)
        return (len(A & B) / U) if U else np.nan

    if nL >= 2 and sum(len(s) for s in corpus_sets.values()) > 0:
        jac = np.zeros((nL, nL), dtype=float)
        for i in range(nL):
            for j in range(nL):
                jac[i, j] = _jaccard(corpus_sets[labels_heat[i]], corpus_sets[labels_heat[j]])

        # mask diagonal; collect valid off-diagonal entries only
        mask_diag = np.eye(nL, dtype=bool).ravel()
        X, Y = np.meshgrid(np.arange(nL), np.arange(nL))
        Xf, Yf = X.ravel()[~mask_diag], Y.ravel()[~mask_diag]
        vals = jac.ravel()[~mask_diag]

        vmax = float(np.nanpercentile(vals[np.isfinite(vals)], 95)) if np.isfinite(vals).any() else 1.0
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        cmap = plt.get_cmap('Spectral_r')

        sizes = 900 * (vals / vmax)
        sizes[~np.isfinite(sizes)] = 0.0

        edge_cols = cmap(norm(vals))
        face_cols = np.tile(np.array([[1, 1, 1, 1]]), (len(vals), 1))

        sc = ax3.scatter(
            Xf, Yf,
            s=sizes,
            facecolors=face_cols,        # truly white fill
            edgecolors=edge_cols,        # edges colored by value
            linewidths=1.8,
            marker='o',
            antialiaseds=False,
            zorder=3
        )

        # light grid at cell boundaries
        for k in range(nL + 1):
            ax3.axhline(k - 0.5, color='k', lw=0.4, alpha=0.6, linestyle='--')
            ax3.axvline(k - 0.5, color='k', lw=0.4, alpha=0.3, linestyle='--')

        # colorbar from a ScalarMappable (decoupled from marker faces)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Jaccard\nOverlap', rotation=0, labelpad=-30, y=1.1)
        cbar.ax.xaxis.set_label_position('top')

        ax3.set_xlim(-0.5, nL - 0.5)
        ax3.set_ylim(nL - 0.5, -0.5)
        ax3.set_xticks(range(nL)); ax3.set_yticks(range(nL))
        ax3.set_xticklabels(labels_heat, rotation=45, ha='right')
        ax3.set_yticklabels(labels_heat)
        ax3.set_title('c.', loc='left', fontsize=23, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, "Insufficient data", ha='center')
    ax3.set_axisbelow(True)

    # -------------------- ax4 (Panel d): RNG-only domain totals (no NaNs; annotated; bars above grid) ----
    domain_to_ids = defaultdict(set)

    for label in label_order:
        df = loaded[label]
        if df is None or 'domain' not in df.columns:
            continue
        dois = df['doi'] if 'doi' in df.columns else pd.Series([None]*len(df))
        titles = df['display_name'] if 'display_name' in df.columns else pd.Series([None]*len(df))
        for dom, doi, title in zip(df['domain'], dois, titles):
            if not isinstance(dom, str) or not dom.strip() or dom.strip().lower() == 'nan':
                continue
            key = _clean_doi(doi)
            if not key:
                key = _clean_title(title)
            if not key:
                continue
            domain_to_ids[dom.strip()].add(key)

    if domain_to_ids:
        dom_names = list(domain_to_ids.keys())
        counts = [len(domain_to_ids[d]) for d in dom_names]
        dc_rng = pd.DataFrame({'domain_name': dom_names, 'count': counts})
        dc_rng = dc_rng[~dc_rng['domain_name'].isna()]

        order = ['Physical Sciences', 'Health Sciences', 'Life Sciences', 'Social Sciences']
        dc_rng['order_idx'] = dc_rng['domain_name'].apply(lambda x: order.index(x) if x in order else len(order))
        dc_rng = dc_rng.sort_values(['order_idx', 'domain_name']).drop(columns='order_idx')

        ax4.set_axisbelow(True)
        ax4.grid(which="both", linestyle='--', alpha=0.225, zorder=1)
        bars = ax4.bar(dc_rng['domain_name'], dc_rng['count'],
                       edgecolor='k', color=color_list[0], zorder=3)
        ax4.set_ylabel('Total RNG works (deduplicated)', fontsize=14)
        ax4.set_title('d.', loc='left', fontsize=23, fontweight='bold')

        for rect, v in zip(bars, dc_rng['count']):
            ax4.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(),
                     f"{int(v):,}", ha='center', va='bottom',
                     fontsize=10, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "No RNG domain data", ha='center')
        ax4.set_axisbelow(True)
        ax4.grid(which="both", linestyle='--', alpha=0.225)

    # Despine select axes
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax4)

    ax2.set_ylim(0, 2.25)
    plt.tight_layout()
    _save(fig, 'rng_sciento_panels_swapped_fix_scatter_fill')



def plot_scientometrics(figure_path, domain_df):
    df_rng, df_hrng, df_qrng, df_prng, df_quarng, df_yr, df_yr_dom, df_dom = load_scientometrics()
    df_yr = make_scientometric_ts(df_rng, df_hrng, df_qrng, df_prng, df_quarng, df_yr, domain_df)

    colors = ['#001c54', '#E89818']
    percent_formatter = FuncFormatter(lambda x, pos: f'{x:.3f}%')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    df_yr[df_yr['year'] >= 1970].set_index('year', inplace=False)[['QRNG_count']].plot(ax=ax1, legend=False,
                                                                                       color=colors[0], alpha=1)
    df_yr[df_yr['year'] >= 1970].set_index('year', inplace=False)[['PRNG_count']].plot(ax=ax2, legend=False,
                                                                                       color=colors[0], alpha=1)
    df_yr[df_yr['year'] >= 1970].set_index('year', inplace=False)[['HRNG_count']].plot(ax=ax3, legend=False,
                                                                                       color=colors[0], alpha=1)
    df_yr[df_yr['year'] >= 1970].set_index('year', inplace=False)[['QUASI_count']].plot(ax=ax4, legend=False,
                                                                                        color=colors[0], alpha=1)

    for ax, title in zip([ax1, ax2, ax3, ax4], ['a.', 'b.', 'c.', 'd.']):
        ax.set_axisbelow(True)
        ax.grid(which="both", linestyle='--', alpha=0.225)
        ax.set_title(title, loc='left', fontsize=21, y=1.025, x=-0.075)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.yaxis.set_major_formatter(percent_formatter)

    ax1.set_ylabel('Percent of All Records', fontsize=15)
    ax3.set_ylabel('Percent of All Records', fontsize=15)
    ax3.set_xlabel('Year', fontsize=15)
    ax4.set_xlabel('Year', fontsize=15)

    # Create insets for each subplot
    inset_ax1 = ax1.inset_axes([0.25, 0.6, 0.225, 0.35])  # Top-left position inset
    domain_df['"Random Numbers" and "Quantum"'].plot(kind='barh', ax=inset_ax1, edgecolor='k', color=colors[1])

    inset_ax2 = ax2.inset_axes([0.25, 0.6, 0.225, 0.35])  # Top-left position inset
    domain_df['"Random Numbers" and "Pseudo"'].plot(kind='barh', ax=inset_ax2, edgecolor='k', color=colors[1])

    inset_ax3 = ax3.inset_axes([0.25, 0.6, 0.225, 0.35])  # Top-left position inset
    domain_df['"Random Numbers" and "Hardware"'].plot(kind='barh', ax=inset_ax3, edgecolor='k', color=colors[1])

    inset_ax4 = ax4.inset_axes([0.25, 0.6, 0.225, 0.35])  # Top-left position inset
    domain_df['"Random Numbers" and "Quasi"'].plot(kind='barh', ax=inset_ax4, edgecolor='k', color=colors[1])

    # Remove x-axis labels and y-axis ticks from insets
    for inset_ax in [inset_ax1, inset_ax2, inset_ax3, inset_ax4]:
        inset_ax.set_ylabel('')

        # Find the maximum width of the bars to adjust the x-axis limits
        max_width = max([p.get_width() for p in inset_ax.patches])
        inset_ax.set_xlim(0, max_width * 1.2)  # Extend x-axis limit slightly beyond the max bar width

        inset_ax.set_xticks([])
        inset_ax.set_xticklabels([])
        # Annotate bars with their value (horizontal annotations inside the bars)
        for p in inset_ax.patches:
            width = p.get_width()
            inset_ax.annotate(f'{width:.4f}%',
                              (width + 0.00025, p.get_y() + p.get_height() / 2),
                              # Place text slightly to the right of the bar
                              ha='left', va='center', fontsize=10, color='k')

    ax1.text(0.95, 0.05, '"Random Numbers"\nand "Quantum"    ',  # Example annotation text
             transform=ax1.transAxes, fontsize=10, color='black',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    ax2.text(0.95, 0.05, '"Random Numbers"\nand "Pseudo"     ',  # Example annotation text
             transform=ax2.transAxes, fontsize=10, color='black',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    ax3.text(0.95, 0.05, '"Random Numbers"\nand "Hardware"   ',  # Example annotation text
             transform=ax3.transAxes, fontsize=10, color='black',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    ax4.text(0.95, 0.05, '"Random Numbers"\nand "Quasi"       ',  # Example annotation text
             transform=ax4.transAxes, fontsize=10, color='black',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    plt.tight_layout()
    sns.despine()
    sns.despine(ax=inset_ax1, left=False, right=True, top=True, bottom=True)
    sns.despine(ax=inset_ax2, left=False, right=True, top=True, bottom=True)
    sns.despine(ax=inset_ax3, left=False, right=True, top=True, bottom=True)
    sns.despine(ax=inset_ax4, left=False, right=True, top=True, bottom=True)
    plt.savefig(os.path.join(figure_path, 'scientometrics_over_time.pdf'), bbox_inches='tight')


def plot_predictions(first_wave_10k_stratified_list,
                     figure_path,
                     figsize,
                     colors = ['#001c54', '#E89818']
                     ):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0:3, 0:1])
    ax2 = fig.add_subplot(gs[0:1, 1:2])
    ax3 = fig.add_subplot(gs[1:2, 1:2])
    ax4 = fig.add_subplot(gs[2:3, 1:2])
    ax5 = fig.add_subplot(gs[3:4, 0:1])
    ax6 = fig.add_subplot(gs[4:5, 0:1])
    ax7 = fig.add_subplot(gs[5:6, 0:1])
    ax8 = fig.add_subplot(gs[3:6, 1:2])
    housing = pd.read_csv(os.path.join(os.getcwd(),
                                       '..',
                                       'data',
                                       'housing',
                                       'results',
                                       'housing_outputs_ols.csv'),
                          index_col=0)
    housing = housing.reset_index()

    titanic = pd.read_csv(os.path.join(os.getcwd(),
                                       '..',
                                       'data',
                                       'titanic',
                                       'results',
                                       'titanic_outputs_logistic.csv'))
    mnist = pd.read_csv(os.path.join(os.path.join(os.getcwd(),
                                                  '..',
                                                  'data',
                                                  'MNIST',
                                                  'results',
                                                  'mnist_results.csv'))
                        )
    print('Covid min: ', np.min(first_wave_10k_stratified_list))
    print('Covid max: ', np.max(first_wave_10k_stratified_list))
    print('Covid mean: ', np.mean(first_wave_10k_stratified_list))
    print('Housing min: ', housing['R2'].min())
    print('Housing max: ', housing['R2'].max())
    print('Housing mean: ', housing['R2'].mean())
    print('Titanic min: ', titanic['IMV'].min())
    print('Titanic max: ', titanic['IMV'].max())
    print('Titanic mean: ', titanic['IMV'].mean())
    print('MNIST min: ', mnist['correct'].min())
    print('MNIST max: ', mnist['correct'].max())
    print('MNIST mean: ', mnist['correct'].mean())
    nbins = 24
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    csfont = {'fontname': 'Helvetica'}
    sns.histplot(first_wave_10k_stratified_list, edgecolor='k',
                 color=colors[0], alpha=1, stat='density',
                 ax=ax1, bins=nbins)
    sns.kdeplot(first_wave_10k_stratified_list,
                color=colors[1],
                ax=ax1,
                common_norm=True,
                linewidth=2
                )

    sns.histplot(housing['R2'],
                 edgecolor='k',
                 color=colors[0],
                 alpha=1,
                 stat='density',
                 ax=ax2,
                 bins=nbins
                 )

    modelling_seed_variance = housing.groupby('Folding_Seed')['R2'].std().reset_index()
    modelling_seed_variance.columns = ['Folding_Seed', 'R2_variance']
    sns.histplot(modelling_seed_variance['R2_variance'], edgecolor='k',
                 color=colors[1], alpha=1, stat='density',
                 ax=ax3, bins=nbins)

    if 'Modeling_Seed' in housing.columns:
        folding_seed_variance = housing.groupby('Modeling_Seed')['R2'].std().reset_index()
        folding_seed_variance.columns = ['Modeling_Seed', 'R2_variance']
    else:
        folding_seed_variance = modelling_seed_variance.rename(columns={'Folding_Seed': 'Modeling_Seed'})
    sns.histplot(np.round(folding_seed_variance['R2_variance'], 5), edgecolor='k',
                 color=colors[1], alpha=1, stat='density',
                 ax=ax4, bins=nbins)

    sns.histplot(titanic['IMV'], edgecolor='k',
                 color=colors[0], alpha=1, stat='density',
                 ax=ax5, bins=nbins
                 )

    modelling_seed_variance = titanic.groupby('Folding_Seed')['IMV'].std().reset_index()
    modelling_seed_variance.columns = ['Folding_Seed', 'IMV_variance']
    sns.histplot(modelling_seed_variance['IMV_variance'], edgecolor='k',
                 color=colors[1], alpha=1, stat='density',
                 ax=ax6, bins=nbins)

    if 'Modeling_Seed' in titanic.columns:
        folding_seed_variance = titanic.groupby('Modeling_Seed')['IMV'].std().reset_index()
        folding_seed_variance.columns = ['Modeling_Seed', 'IMV_variance']
    else:
        folding_seed_variance = modelling_seed_variance.rename(columns={'Folding_Seed': 'Modeling_Seed', 'IMV_variance': 'IMV_variance'})
    sns.histplot(folding_seed_variance['IMV_variance'], edgecolor='k',
                 color=colors[1], alpha=1, stat='density',
                 ax=ax7, bins=nbins)

    sns.histplot(mnist['correct'],
                 edgecolor='k',
                 color=colors[0],
                 alpha=1,
                 stat='density',
                 ax=ax8,
                 bins=nbins
                 )
    sns.kdeplot(mnist['correct'],
                color=colors[1],
                ax=ax8,
                common_norm=True,
                linewidth=2
                )

    for ax in [ax2, ax3, ax4, ax8]:
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

    ax1.set_ylabel('Density', fontsize=15)
    ax2.set_ylabel('')
    ax3.set_ylabel('')
    ax4.set_ylabel('')
    ax5.set_ylabel('Density', fontsize=15)
    ax6.set_ylabel('Density', fontsize=15)
    ax7.set_ylabel('Density', fontsize=15)
    ax8.set_ylabel('')

    ax1.set_xlabel('ROC-AUC', fontsize=15)
    ax2.set_xlabel(r'R$^2$', fontsize=15)
    ax3.set_xlabel(r'R$^2$: Modelling ($\sigma$)', fontsize=15)
    ax4.set_xlabel(r'R$^2$: Folding ($\sigma$)', fontsize=15)
    ax5.set_xlabel('IMV', fontsize=14)
    ax6.set_xlabel(r'IMV: Modelling ($\sigma$)', fontsize=15)
    ax7.set_xlabel(r'IMV: Folding ($\sigma$)', fontsize=15)

    ax8.set_xlabel('Accuracy', fontsize=16)
    legend_elements1 = [Patch(facecolor=colors[0], edgecolor='k',
                              label=r'Bins', alpha=1),
                        Line2D([0], [0], color=colors[1], lw=1.5, linestyle='-',
                               label=r'KDE', alpha=1), ]
    ax1.legend(handles=legend_elements1, loc='center left', frameon=True,
               fontsize=label_fontsize - 2, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1)
               )
    ax8.legend(handles=legend_elements1, loc='center left', frameon=True,
               fontsize=label_fontsize - 2, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1)
               )

    mean = round(np.nanmean(first_wave_10k_stratified_list), 3)
    var = round(np.nanstd(first_wave_10k_stratified_list), 3)
    ax1.annotate(r'E(AUC) = ' + str(mean) + r', $\sigma$(AUC) = ' + str(var),
                 xy=(0.5, 0.875), xytext=(0.5, 0.925), xycoords='axes fraction',
                 fontsize=13, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                 arrowprops=dict(arrowstyle='-[, widthB=12.5, lengthB=1',
                                 lw=1.5)
                 )

    mean = round(np.nanmean(mnist['correct']), 3)
    var = round(np.nanstd(mnist['correct']), 3)
    ax8.annotate(r'E(acc) = ' + str(mean) + r', $\sigma$(acc) = ' + str(var),
                 xy=(0.5, 0.875), xytext=(0.5, 0.925), xycoords='axes fraction',
                 fontsize=13, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                 arrowprops=dict(arrowstyle='-[, widthB=12.5, lengthB=1',
                                 lw=1.5)
                 )

    for ax, title in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8],
                         ['a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.']):
        ax.set_axisbelow(True)
        ax.grid(which="both", linestyle='--', alpha=0.225)
        ax.set_title(title, loc='left', fontsize=21, y=1.025, x=-0.075)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] / 7)

    ax1.axvline(x=0.76, ymin=0, ymax=0.82, color='red', linestyle='--')
    ymin, ymax = ax1.get_ylim()
    annotation_y = ymin + (ymax - ymin) * 0.5  # 70% up the y-axis
    ax1.annotate(' Original Result:\nROC-AUC=0.76\n (0.74-0.78)',
                 xy=(0.76, annotation_y),  # Position the arrow at 9625, 70% of y-axis range
                 xytext=(0.8, annotation_y),  # Position the text slightly to the left
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1.5)
                 )

    ax8.axvline(x=9625, ymin=0, ymax=0.82, color='red', linestyle='--')
    ymin, ymax = ax8.get_ylim()
    annotation_y = ymin + (ymax - ymin) * 0.6  # 70% up the y-axis
    ax8.annotate('   Seed 42:\nAccuracy=9625',
                 xy=(9625, annotation_y),  # Position the arrow at 9625, 70% of y-axis range
                 xytext=(9300, annotation_y),  # Position the text slightly to the left
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1.5)
                 )

    ax8.axvline(x=9690, ymin=0, ymax=0.82, color='red', linestyle='--')
    ymin, ymax = ax8.get_ylim()
    annotation_y = ymin + (ymax - ymin) * 0.35  # 70% up the y-axis
    ax8.annotate('   Seed 123:\nAccuracy=9690',
                 xy=(9690, annotation_y),  # Position the arrow at 9625, 70% of y-axis range
                 xytext=(9300, annotation_y),  # Position the text slightly to the left
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1.5)
                 )
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    ax7.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    for ax in [ax1, ax5, ax6, ax7]:
        sns.despine(ax=ax)
    for ax in [ax2, ax3, ax4, ax8]:
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=1.25)
    plt.savefig(os.path.join(figure_path, 'prediction_seeds.pdf'), bbox_inches='tight')


def plot_predictions_three_panel(first_wave_10k_stratified_list,
                                 figure_path,
                                 figsize=(16, 5),
                                 colors=['#001c54', '#E89818']):
    """
    Condensed version of plot_predictions showing only panels a, b, and e
    in a 1x3 layout. Adds KDEs to all three panels and uses a single legend
    on the leftmost axis. Braces above the second and third axes highlight
    their summary statistics.
    """
    mpl.rcParams['font.family'] = 'Helvetica'
    nbins = 24

    housing = pd.read_csv(os.path.join(os.getcwd(),
                                       '..',
                                       'data',
                                       'housing',
                                       'results',
                                       'housing_outputs_ols.csv'),
                          index_col=0).reset_index()

    titanic = pd.read_csv(os.path.join(os.getcwd(),
                                       '..',
                                       'data',
                                       'titanic',
                                       'results',
                                       'titanic_outputs_logistic.csv'))

    def _load_seed_metrics(path, seed):
        metrics = {}
        p = Path(path)
        if not p.exists():
            return metrics
        for line in p.read_text().splitlines():
            if f"seed={seed}" in line:
                for part in line.split(','):
                    if '=' in part:
                        k, v = part.split('=', 1)
                        k = k.strip()
                        v = v.strip()
                        try:
                            metrics[k] = float(v)
                        except ValueError:
                            metrics[k] = v
        return metrics

    housing_seed_metrics = _load_seed_metrics(Path(os.getcwd()) / '..' / 'data' / 'housing' / 'accuracy_seeds.txt', 42)
    titanic_seed_metrics_42 = _load_seed_metrics(Path(os.getcwd()) / '..' / 'data' / 'titanic' / 'accuracy_seeds.txt', 42)
    titanic_seed_metrics_123 = _load_seed_metrics(Path(os.getcwd()) / '..' / 'data' / 'titanic' / 'accuracy_seeds.txt', 123)

    def _print_stats(name, arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n == 0:
            print(f"{name}: empty")
            return
        mean = float(np.mean(arr))
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
        # Effect-size style z (mean over SD) to avoid huge t-stats when n is large
        z_score = mean / std if std not in (0, float("nan")) else float("nan")
        print(f"{name}: n={n}, mean={mean:.4f}, min={amin:.4f}, max={amax:.4f}, sd={std:.4f}, z={z_score:.4f}")

    # Flattened folds stats
    _print_stats("COVID ROC-AUC (all folds)", first_wave_10k_stratified_list)
    _print_stats("Housing R2 (OLS)", housing['R2'])
    _print_stats("Titanic IMV (Logistic Regression)", titanic['IMV'])

    # Row-wise (per-seed) averages across the 5 folds, then summary stats
    if isinstance(first_wave_10k_stratified_list, (list, np.ndarray)):
        arr_raw = np.asarray(first_wave_10k_stratified_list, dtype=float)
        fold_count = arr_raw.shape[1] if arr_raw.ndim == 2 else 5
        row_means = None

        if arr_raw.ndim == 2:
            row_means = np.nanmean(arr_raw, axis=1)
        elif arr_raw.ndim == 1 and arr_raw.size % fold_count == 0:
            row_means = np.nanmean(arr_raw.reshape(-1, fold_count), axis=1)

        if row_means is not None and row_means.size:
            _print_stats("COVID ROC-AUC (row means across folds)", row_means)
        else:
            print(
                f"COVID ROC-AUC (row means): array size {arr_raw.size} "
                f"not divisible by {fold_count}; skipping row means."
            )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax1, ax2, ax3 = axes

    def _add_brace_with_text(ax, text):
        """
        Match the compact bracket-and-label style used in plot_predictions.
        """
        ax.annotate(
            text,
            xy=(0.5, 0.95), xytext=(0.5, 1.015),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=13, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='black', lw=1.0),
            arrowprops=dict(arrowstyle='-[, widthB=9.5, lengthB=1', lw=1.5),
        )

    # Panel a: COVID ROC-AUC
    sns.histplot(first_wave_10k_stratified_list, edgecolor='k',
                 color=colors[0], alpha=1, stat='density',
                 ax=ax1, bins=nbins)
    sns.kdeplot(first_wave_10k_stratified_list,
                color=colors[1],
                ax=ax1,
                common_norm=True,
                linewidth=2)
    ax1.set_xlabel('ROC-AUC', fontsize=13)
    ax1.set_ylabel('Density', fontsize=13)
    ax1.set_title('a.', loc='left', fontsize=18, y=1.02, x=-0.05, fontweight='bold')

    # Panel b: Housing R^2
    sns.histplot(housing['R2'],
                 edgecolor='k',
                 color=colors[0],
                 alpha=1,
                 stat='density',
                 ax=ax2,
                 bins=nbins)
    sns.kdeplot(housing['R2'],
                color=colors[1],
                ax=ax2,
                common_norm=True,
                linewidth=2)
    ax2.set_xlabel(r'R$^2$', fontsize=13)
    ax2.set_ylabel('Density', fontsize=13)
    ax2.set_title('b.', loc='left', fontsize=18, y=1.02, x=-0.05, fontweight='bold')

    # Panel e: Titanic IMV
    sns.histplot(titanic['IMV'],
                 edgecolor='k',
                 color=colors[0],
                 alpha=1,
                 stat='density',
                 ax=ax3,
                 bins=nbins)
    sns.kdeplot(titanic['IMV'],
                color=colors[1],
                ax=ax3,
                common_norm=True,
                linewidth=2)
    ax3.set_xlabel('IMV', fontsize=13)
    ax3.set_ylabel('Density', fontsize=13)
    ax3.set_title('c.', loc='left', fontsize=18, y=1.02, x=-0.05, fontweight='bold')

    # One legend on the left of panel a
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='k', label='Bins', alpha=1),
        Line2D([0], [0], color=colors[1], lw=1.75, linestyle='-', label='KDE', alpha=1),
    ]
    ax1.legend(handles=legend_elements, loc='center left', frameon=True,
               fontsize=11, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1))

    # Braces above panels with summary stats
    auc_mean = np.nanmean(first_wave_10k_stratified_list)
    auc_std = np.nanstd(first_wave_10k_stratified_list)
    housing_mean = np.nanmean(housing['R2'])
    housing_std = np.nanstd(housing['R2'])
    titanic_mean = np.nanmean(titanic['IMV'])
    titanic_std = np.nanstd(titanic['IMV'])

    _add_brace_with_text(ax1, f"E(AUC) = {auc_mean:.3f}, σ(AUC) = {auc_std:.3f}")
    _add_brace_with_text(ax2, f"E(R²) = {housing_mean:.3f}, σ(R²) = {housing_std:.3f}")
    _add_brace_with_text(ax3, f"E(IMV) = {titanic_mean:.3f}, σ(IMV) = {titanic_std:.3f}")

    # Original-result annotation for panel a (mirroring plot_predictions)
    # Extend vline to the brace (assumes brace near top of axes)
    ax1.axvline(x=0.76, ymin=0, ymax=0.95, color='red', linestyle='--')
    ymin, ymax = ax1.get_ylim()
    annotation_y = ymin + (ymax - ymin) * 0.54
    x0, x1 = ax1.get_xlim()
    span = x1 - x0
    text_x = x1 - 0.12 * span  # slight additional move right
    ax1.annotate(' Original Result:\nROC-AUC=0.76\n (0.74-0.78)',
                 xy=(0.76, annotation_y),
                 xytext=(text_x, annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1.5)
                 )

    # Housing annotation for seed 42 (panel b)
    if housing_seed_metrics:
        y_min, y_max = ax2.get_ylim()
        ann_y = y_min + (y_max - y_min) * 0.55
        hx0, hx1 = ax2.get_xlim()
        h_span = hx1 - hx0
        ols_r2 = housing_seed_metrics.get('ols_R2') or housing_seed_metrics.get('logistic_accuracy')
        try:
            arrow_x = float(ols_r2)
        except Exception:
            arrow_x = housing_mean
            print("Warning: Seed 42 OLS R² missing; using overall mean.")
        text_x = hx1 - 0.12 * h_span  # slight additional move right
        ols_str = f"OLS R²={arrow_x:.4f}" if isinstance(arrow_x, float) else f"OLS R²={ols_r2}"
        ax2.axvline(x=arrow_x, ymin=0, ymax=0.95, color='red', linestyle='--')
        ax2.annotate(f"Seed 42:\n{ols_str}",
                     xy=(arrow_x, ann_y),
                     xytext=(text_x, ann_y),
                     ha='center',
                     va='center',
                     fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle="arc3,rad=0",
                                     color='black',
                                     mutation_scale=20,
                                     lw=1.5)
                     )

    # Titanic annotation using per-seed IMV (panel c) — only Seed 123
    if titanic_seed_metrics_123:
        y_min, y_max = ax3.get_ylim()
        ann_y = y_min + (y_max - y_min) * 0.55
        tx0, tx1 = ax3.get_xlim()
        t_span = tx1 - tx0
        imv_123_raw = titanic_seed_metrics_123.get('lr_imv')
        imv_123_val = float(imv_123_raw) if imv_123_raw not in (None, "") else float("nan")
        if not np.isfinite(imv_123_val):
            print("Warning: Seed 123 IMV missing; annotation skipped.")
        else:
            text_x = tx1 - 0.12 * t_span  # slight additional move right
            ax3.axvline(x=imv_123_val, ymin=0, ymax=0.95, color='red', linestyle='--')
            ax3.annotate(f"Seed 123:\nIMV={imv_123_val:.4f}",
                         xy=(imv_123_val, ann_y),
                         xytext=(text_x, ann_y),
                         ha='center',
                         va='center',
                         fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                         arrowprops=dict(arrowstyle='->',
                                         connectionstyle="arc3,rad=0",
                                         color='black',
                                         mutation_scale=20,
                                         lw=1.5)
                         )

    # modest x-padding (10%) to give braces/annotations breathing room
    for ax in axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xpad = 0.05 * (x1 - x0)
        ypad = 0.10 * (y1 - y0)
        ax.set_xlim(x0 - xpad, x1 + xpad)
        ax.set_ylim(y0, y1 + ypad)

    for ax in axes:
        ax.grid(which="both", linestyle='--', alpha=0.225)
        ax.tick_params(axis='both', which='major', labelsize=11)
        sns.despine(ax=ax)

    plt.tight_layout()
    filename = 'prediction_seeds_three_panel'
    plt.savefig(os.path.join(figure_path, f'{filename}.pdf'), bbox_inches='tight')


def plot_housing_titanic_seed_sigma(figure_path,
                                    figsize=(14, 10),
                                    colors=None,
                                    bins=24,
                                    metric_housing="R2",
                                    metric_titanic="IMV",
                                    return_fig=False):
    """
    Two-column, three-row figure showing overall metric and the
    spread (std) across modeling and folding seeds for housing (RF)
    and titanic (RF + SGD). Styling mirrors plot_predictions_three_panel,
    with KDE overlays, braces, and seed-123 annotation.
    """
    if colors is None:
        colors = ['#486cb0', '#fed07e']
    main_color = colors[0]
    sigma_color = colors[1] if len(colors) > 1 else colors[0]
    mpl.rcParams['font.family'] = 'Helvetica'

    def _to_num(series):
        return pd.to_numeric(series, errors="coerce")

    def _group_within_std(df, group_cols, metric):
        """Std within each group (requires >=2 values per group)."""
        if not set(group_cols).issubset(df.columns):
            return np.asarray([], dtype=float)
        vals = []
        for _, g in df.groupby(group_cols):
            s = _to_num(g[metric]).dropna()
            if s.size >= 2:
                vals.append(float(s.std(ddof=1)))
        return np.asarray(vals, dtype=float)

    def _stats(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return {"n": 0, "min": float("nan"), "max": float("nan"),
                    "mean": float("nan"), "std": float("nan"), "z": float("nan")}
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1) if arr.size > 1 else np.nan)
        z = mean / std if std not in (0, float("nan")) else float("nan")
        return {"n": int(arr.size), "min": float(np.min(arr)), "max": float(np.max(arr)),
                "mean": mean, "std": std, "z": z}

    def _set_xlim(ax, data, pad_frac=0.03):
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        if data.size == 0:
            return
        dmin, dmax = float(np.min(data)), float(np.max(data))
        span = dmax - dmin
        pad = span * pad_frac if span > 0 else 0.001
        ax.set_xlim(dmin - pad, dmax + pad)

    # Load data
    housing_rf = pd.read_csv(Path(os.getcwd()) / ".." / "data" / "housing" / "results" / "housing_outputs_rf.csv")
    titanic_sgd = pd.read_csv(Path(os.getcwd()) / ".." / "data" / "titanic" / "results" / "titanic_outputs_sgd.csv")

    # Only SGD for titanic
    titanic_all = titanic_sgd.assign(Model="SGD")

    # Distributions
    housing_vals = _to_num(housing_rf[metric_housing]).dropna()
    housing_model_std = _group_within_std(housing_rf, ["Modeling_Seed"], metric_housing)
    housing_fold_std = _group_within_std(housing_rf, ["Folding_Seed"], metric_housing)

    titanic_vals = _to_num(titanic_all[metric_titanic]).dropna()
    # For SGD-only data, group by single seed columns
    titanic_model_std = _group_within_std(titanic_all, ["Modeling_Seed"], metric_titanic)
    titanic_fold_std = _group_within_std(titanic_all, ["Folding_Seed"], metric_titanic)

    housing_stats = _stats(housing_vals)
    housing_model_stats = _stats(housing_model_std)
    housing_fold_stats = _stats(housing_fold_std)
    titanic_stats = _stats(titanic_vals)
    titanic_model_stats = _stats(titanic_model_std)
    titanic_fold_stats = _stats(titanic_fold_std)

    print("Housing RF R2 (all folds):", housing_stats)
    print("Housing RF modeling σ:", housing_model_stats)
    print("Housing RF folding σ:", housing_fold_stats)
    print("Titanic SGD IMV (all folds):", titanic_stats)
    print("Titanic modeling σ:", titanic_model_stats)
    print("Titanic folding σ:", titanic_fold_stats)

    # Scale requested figsize by row ratios so height reflects layout even if caller passes a small height
    row_ratios = [2.75, 1.4, 1.4]  # first row taller than others
    if figsize is None:
        fig_w, fig_h = 14, 10 * (sum(row_ratios) / 3.0)
    else:
        fig_w, base_h = figsize
        fig_h = base_h * (sum(row_ratios) / 3.0)

    fig, axes = plt.subplots(
        3, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": row_ratios},
    )
    (ax_h_main, ax_t_main), (ax_h_model, ax_t_model), (ax_h_fold, ax_t_fold) = axes

    def _add_brace_with_text(ax, text, xfrac=0.5):
        ax.annotate(
            text,
            xy=(xfrac, 0.95), xytext=(xfrac, 1.015),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=13, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='black', lw=1.0),
            arrowprops=dict(arrowstyle='-[, widthB=9.5, lengthB=1', lw=1.5),
        )

    def _pad_ylim(ax, frac=0.12):
        y0, y1 = ax.get_ylim()
        span = y1 - y0
        ax.set_ylim(y0, y1 + span * frac)

    def _plot_hist_kde(ax, data, hist_color, kde_color, bins_local=None, kde_linewidth=1.5):
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        if data.size == 0:
            return
        sns.histplot(data, color=hist_color, edgecolor="k", bins=bins_local or bins,
                     alpha=0.9, stat="density", ax=ax)
        # Only draw KDE when variance exists
        if data.size > 1 and np.nanstd(data) > 0:
            sns.kdeplot(data, color=kde_color, ax=ax, linewidth=kde_linewidth)
        else:
            print(f"Warning: zero-variance or singleton data for KDE on axis titled '{ax.get_title()}'; skipping KDE.")

    # Housing main metric
    _plot_hist_kde(ax_h_main, housing_vals, hist_color=main_color, kde_color=sigma_color, kde_linewidth=1.75)
    ax_h_main.set_xlabel(r'R$^2$', fontsize=13)
    ax_h_main.set_ylabel('Density', fontsize=13)
    ax_h_main.set_title('a.', loc='left', fontsize=18, y=1.02, x=-0.05, fontweight='bold')
    hx0, hx1 = ax_h_main.get_xlim()
    hspan = hx1 - hx0 if hx1 != hx0 else 1.0
    hfrac = min(max((housing_stats['mean'] - hx0) / hspan, 0.05), 0.95)
    _add_brace_with_text(
        ax_h_main,
        f"E(R²) = {housing_stats['mean']:.3f}, σ = {housing_stats['std']:.3f}",
        xfrac=hfrac
    )
    _set_xlim(ax_h_main, housing_vals)
    _pad_ylim(ax_h_main, frac=0.18)  # extra space for brace/annotation
    legend_elements = [
        Patch(facecolor=main_color, edgecolor='k', label='Bins', alpha=0.9),
        Line2D([0], [0], color=sigma_color, lw=1.75, linestyle='-', label='KDE', alpha=1),
    ]
    ax_h_main.legend(handles=legend_elements, loc='center left', frameon=True,
                     fontsize=11, framealpha=1, facecolor='w',
                     edgecolor=(0, 0, 0, 1))

    # Housing modeling sigma
    _plot_hist_kde(ax_h_model, housing_model_std, hist_color=sigma_color, kde_color=main_color, kde_linewidth=1.5)
    ax_h_model.set_xlabel(r'R$^2$: Modelling (σ)', fontsize=13)
    ax_h_model.set_ylabel('Density', fontsize=13)
    ax_h_model.set_title('b.', loc='left', fontsize=18, y=1.02, x=-0.05, fontweight='bold')
    _set_xlim(ax_h_model, housing_model_std)

    # Housing folding sigma
    _plot_hist_kde(ax_h_fold, housing_fold_std, hist_color=sigma_color, kde_color=main_color, kde_linewidth=1.5)
    ax_h_fold.set_xlabel(r'R$^2$: Folding (σ)', fontsize=13)
    ax_h_fold.set_ylabel('Density', fontsize=13)
    ax_h_fold.set_title('c.', loc='left', fontsize=18, y=1.02, x=-0.05, fontweight='bold')
    _set_xlim(ax_h_fold, housing_fold_std)

    # Titanic main metric (IMV)
    _plot_hist_kde(ax_t_main, titanic_vals, hist_color=main_color, kde_color=sigma_color, kde_linewidth=1.75)
    ax_t_main.set_xlabel('IMV', fontsize=13)
    ax_t_main.set_ylabel('Density', fontsize=13)
    ax_t_main.set_title('d.', loc='left', fontsize=18, y=1.02, x=-0.05, fontweight='bold')
    tx0, tx1 = ax_t_main.get_xlim()
    tspan = tx1 - tx0 if tx1 != tx0 else 1.0
    tfrac = min(max((titanic_stats['mean'] - tx0) / tspan, 0.05), 0.95)
    _add_brace_with_text(
        ax_t_main,
        f"E(IMV) = {titanic_stats['mean']:.3f}, σ = {titanic_stats['std']:.3f}",
        xfrac=tfrac
    )
    _set_xlim(ax_t_main, titanic_vals)
    _pad_ylim(ax_t_main, frac=0.18)  # extra space for brace/annotation

    # Seed=123 vertical line on titanic main panel (prefer Modeling_Seed==123 else Folding_Seed==123)
    seed123_rows = titanic_all[(titanic_all.get("Modeling_Seed") == 123) | (titanic_all.get("Folding_Seed") == 123)]
    seed123_vals = _to_num(seed123_rows[metric_titanic]).dropna()
    if not seed123_vals.empty:
        seed123_val = float(seed123_vals.iloc[0])
        ax_t_main.axvline(x=seed123_val, ymin=0, ymax=0.95, color='red', linestyle='--')
        y_min, y_max = ax_t_main.get_ylim()
        ann_y = y_min + (y_max - y_min) * 0.55
        x0, x1 = ax_t_main.get_xlim()
        span = x1 - x0
        text_x = x1 - 0.22 * span  # shift left ~10% more
        ax_t_main.annotate(f"Seed 123:\nIMV={seed123_val:.4f}",
                           xy=(seed123_val, ann_y),
                           xytext=(text_x, ann_y),
                           ha='center',
                           va='center',
                           fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                           arrowprops=dict(arrowstyle='->',
                                           connectionstyle="arc3,rad=0",
                                           color='black',
                                           mutation_scale=20,
                                           lw=1.5)
                           )

    # Seed-specific annotations from accuracy files
    try:
        housing_acc_path = Path(os.getcwd()) / ".." / "data" / "housing" / "accuracy_seeds.txt"
        titanic_acc_path = Path(os.getcwd()) / ".." / "data" / "titanic" / "accuracy_seeds.txt"
        seed123_rf_r2 = None
        seed123_sgd_imv = None
        if housing_acc_path.exists():
            for line in housing_acc_path.read_text().splitlines():
                if "seed=123" in line:
                    m = re.search(r"rf_R2=([0-9.]+)", line)
                    if m:
                        seed123_rf_r2 = float(m.group(1))
                        break
        if titanic_acc_path.exists():
            for line in titanic_acc_path.read_text().splitlines():
                if "seed=123" in line:
                    m = re.search(r"sgd_imv=([-0-9.]+)", line)
                    if m:
                        seed123_sgd_imv = float(m.group(1))
                        break

        if seed123_rf_r2 is not None:
            ax_h_main.axvline(x=seed123_rf_r2, ymin=0, ymax=0.95, color='red', linestyle='--')
            y_min, y_max = ax_h_main.get_ylim()
            ann_y = y_min + (y_max - y_min) * 0.65
            x0, x1 = ax_h_main.get_xlim()
            span = x1 - x0
            text_x = x1 - 0.22 * span  # shift left ~10% more
            ax_h_main.annotate(f"Seed 123:\nRF R²={seed123_rf_r2:.4f}",
                               xy=(seed123_rf_r2, ann_y),
                               xytext=(text_x, ann_y),
                               ha='center',
                               va='center',
                               fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                               arrowprops=dict(arrowstyle='->',
                                               connectionstyle="arc3,rad=0",
                                               color='black',
                                               mutation_scale=20,
                                               lw=1.5)
                               )

        if seed123_sgd_imv is not None:
            ax_t_main.axvline(x=seed123_sgd_imv, ymin=0, ymax=0.95, color='red', linestyle='--')
            y_min, y_max = ax_t_main.get_ylim()
            ann_y = y_min + (y_max - y_min) * 0.65
            x0, x1 = ax_t_main.get_xlim()
            span = x1 - x0
            text_x = x1 - 0.22 * span  # shift left ~10% more
            ax_t_main.annotate(f"Seed 123:\nSGD IMV={seed123_sgd_imv:.4f}",
                               xy=(seed123_sgd_imv, ann_y),
                               xytext=(text_x, ann_y),
                               ha='center',
                               va='center',
                               fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                               arrowprops=dict(arrowstyle='->',
                                               connectionstyle="arc3,rad=0",
                                               color='black',
                                               mutation_scale=20,
                                               lw=1.5)
                               )
    except Exception as e:
        print(f"Warning: seed annotations failed: {e}")

    # Titanic modeling sigma
    if titanic_model_std.size:
        _plot_hist_kde(ax_t_model, titanic_model_std, hist_color=sigma_color, kde_color=main_color, kde_linewidth=1.5)
        _set_xlim(ax_t_model, titanic_model_std)
    else:
        ax_t_model.text(0.5, 0.5, "No within-seed variance\n(only one obs per modelling seed)",
                        transform=ax_t_model.transAxes, ha='center', va='center', fontsize=11)
    ax_t_model.set_xlabel('IMV: Modelling (σ)', fontsize=13)
    ax_t_model.set_ylabel('Density', fontsize=13)
    ax_t_model.set_title('e.', loc='left', fontsize=18, y=1.02, x=-0.05, fontweight='bold')

    # Titanic folding sigma
    if titanic_fold_std.size:
        _plot_hist_kde(ax_t_fold, titanic_fold_std, hist_color=sigma_color, kde_color=main_color, kde_linewidth=1.5)
        _set_xlim(ax_t_fold, titanic_fold_std)
    else:
        ax_t_fold.text(0.5, 0.5, "No within-seed variance\n(only one obs per folding seed)",
                       transform=ax_t_fold.transAxes, ha='center', va='center', fontsize=11)
    ax_t_fold.set_xlabel('IMV: Folding (σ)', fontsize=13)
    ax_t_fold.set_ylabel('Density', fontsize=13)
    ax_t_fold.set_title('f.', loc='left', fontsize=18, y=1.02, x=-0.05, fontweight='bold')

    # Legends on second/third row, second column (center right)
    legend_elements_sigma = [
        Patch(facecolor=sigma_color, edgecolor='k', label='Bins', alpha=0.9),
        Line2D([0], [0], color=main_color, lw=1.5, linestyle='-', label='KDE', alpha=1),
    ]
    ax_t_model.legend(handles=legend_elements_sigma, loc='center right', frameon=True,
                      fontsize=11, framealpha=1, facecolor='w',
                      edgecolor=(0, 0, 0, 1))
    ax_t_fold.legend(handles=legend_elements_sigma, loc='center right', frameon=True,
                     fontsize=11, framealpha=1, facecolor='w',
                     edgecolor=(0, 0, 0, 1))

    for ax in fig.axes:
        ax.grid(which="both", linestyle='--', alpha=0.225)
        ax.tick_params(axis='both', which='major', labelsize=11)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, 'housing_titanic_seed_sigma.pdf'), bbox_inches='tight')
    return (fig, axes) if return_fig else None

def download_and_resample(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data.resample('D').ffill().dropna()  # Forward fill to handle any missing days
    return data


def plot_further_examples():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    colors = ['#001c54', '#E89818', '#8b0000']
    usuk_data = download_and_resample('USDGBP=X', start="2022-10-01", end="2024-06-30")
    rw_usuk_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_usuk.zip')
    random_walks_usuk = pd.read_csv(rw_usuk_path, header=None, compression='zip')

    def adjust_index(data, rw_data):
        end_date = data.index[-1]
        start_date = end_date + pd.DateOffset(1)  # Start the random walk data the day after the end_date
        new_index = pd.date_range(start=start_date, periods=len(rw_data), freq='D')
        rw_data.index = new_index
        return rw_data

    random_walks_usuk = adjust_index(usuk_data, random_walks_usuk)
    colors = ['#001c54', '#E89818', '#8b0000']
    fill_color = (255 / 255, 223 / 255, 0 / 255, 6 / 255)
    usuk_data['Close'].plot(ax=ax1, color=colors[0])
    random_walks_usuk.min(axis=1).plot(ax=ax1, color=colors[1], alpha=0.8, linestyle='--')
    random_walks_usuk.median(axis=1).plot(ax=ax1, color='k', alpha=0.8, linestyle='--')
    random_walks_usuk.max(axis=1).plot(ax=ax1, color=colors[2], linestyle='--')
    random_walks_usuk.quantile(0.05, axis=1).plot(ax=ax1, color='k', linestyle='--', alpha=0.5, linewidth=0.75)
    random_walks_usuk.quantile(0.95, axis=1).plot(ax=ax1, color='k', linestyle='--', alpha=0.5, linewidth=0.75)

    legend_elements = [
        Line2D([0], [0], color=colors[2], linestyle='--',
               label=r'Max', lw=2),
        Line2D([0], [0], color=colors[1], linestyle='--',
               label=r'Min', lw=2),
        Line2D([0], [0], color=colors[0], linestyle='-',
               label=r'Insample', lw=2),
        Line2D([0], [0], color='k', linestyle='--',
               label=r'Median', lw=2),
        Line2D([0], [0], color='k', linestyle='--', alpha=0.5, linewidth=0.75,
               label=r'95th Percentile', lw=2),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Range')
    ]
    ax1.legend(handles=legend_elements, loc='lower left', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
               )
    ax1.set_xlabel('')

    ax1.grid(which="major", linestyle='--', alpha=0.225)
    ax1.set_title('a.', loc='left', fontsize=22, y=1.01)
    ax2.set_title('b.', loc='left', fontsize=22, y=1.01)

    ax1.fill_between(random_walks_usuk.index,
                     random_walks_usuk.min(axis=1),
                     random_walks_usuk.max(axis=1),
                     color=fill_color
                     )
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'${y / 1:.2f}'))
    ax1.set_ylabel('US ($) / GBP (£)', fontsize=14)
    inset_ax = inset_axes(ax1, width="40%", height="25%", loc='upper left', borderpad=2)
    sns.histplot(random_walks_usuk.iloc[-1], ax=inset_ax,
                 color=colors[0], bins=12,
                 legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax.set_xlabel('US ($) / GBP (£)')
    inset_ax.set_ylabel('Frequency')
    inset_ax.set_axisbelow(True)
    inset_ax.yaxis.set_label_position("right")
    inset_ax.yaxis.tick_right()
    inset_ax.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    print('The minimum USUK RW forecast is:', random_walks_usuk.min(axis=1).iloc[-1])
    print('The maximum USUK RW forecast is:', random_walks_usuk.max(axis=1).iloc[-1])
    print('The median RW forecast is:', random_walks_usuk.median(axis=1).iloc[-1])

    df = pd.read_csv(os.path.join(os.getcwd(),
                                  '..',
                                  'data',
                                  'schelling',
                                  'schelling_df.csv'),
                     index_col=0)
    df1 = pd.read_csv(os.path.join(os.getcwd(),
                                   '..',
                                   'data',
                                   'schelling',
                                   'schelling_summary.csv'),
                      index_col=0
                      )
    df = df[df['Step'] != 'Convergence']
    df['Step'] = df['Step'].astype(int)
    df['Happy Count'] = df['Happy Count'].astype(float)
    df['Happy Count Adjusted'] = df.groupby('Step')['Happy Count'].transform(lambda x: x - x.mean())
    df['Step'] = df['Step'].astype(int)
    df['Happy Count Adjusted'] = df['Happy Count Adjusted'].astype(float)
    filtered_df = df[df['Step'] < 25]
    sns.boxplot(x='Step',
                y='Happy Count Adjusted',
                data=filtered_df,
                legend=True,
                linewidth=1,
                linecolor=colors[0],
                color=colors[1],
                ax=ax2,
                )
    min_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].min()
    max_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].max()
    ax2.plot(min_happy_count.index,
             min_happy_count.values,
             label='Min',
             color=colors[1],
             marker='o',
             markerfacecolor='w',
             linestyle='--')
    ax2.plot(max_happy_count.index,
             max_happy_count.values,
             label='Max',
             color=colors[2],
             marker='o',
             markerfacecolor='w',
             linestyle='--')
    ax2.set_xlabel('Step', fontsize=13)
    ax2.set_ylabel('Mean Adjusted Happy Count', fontsize=13)
    ax2.legend()
    ax2.set_xticks([0, 4, 9, 14, 19, 24, 29, 33])
    ax2.set_axisbelow(True)
    ax2.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax2 = inset_axes(ax2, width="40%", height="25%", loc='lower right', borderpad=2)
    sns.histplot(df1['Total Steps to Converge'], ax=inset_ax2,
                 color=colors[0], bins=12, legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax2.xaxis.set_label_position('top')
    inset_ax2.xaxis.tick_top()
    inset_ax2.set_xlabel('Total Steps')
    inset_ax2.set_ylabel('Frequency')
    inset_ax2.set_xlim(df1['Total Steps to Converge'].min() - 2,
                       df1['Total Steps to Converge'].max())
    inset_ax2.set_axisbelow(True)
    inset_ax2.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    ax2.tick_params(width=0.75, length=6.5, axis='both', which='major', labelsize=11)
    legend_elements2 = [
        Line2D([0], [0], color=colors[2],
               lw=2, linestyle='--', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Max', alpha=1),
        Line2D([0], [0], color=colors[1], lw=2,
               linestyle='-', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Min', alpha=1),
    ]
    ax2.legend(handles=legend_elements2, loc='upper right',
               frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1)
    print(df1['Total Steps to Converge'].min(), df1['Total Steps to Converge'].max())
    fig.subplots_adjust(wspace=0.25)
    filename = 'rw_and_schelling'
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    plt.savefig(os.path.join(os.getcwd(), '..', 'figures', filename + '.pdf'),
                bbox_inches='tight')
#    plt.savefig(os.path.join(os.getcwd(), '..', 'figures', filename + '.svg'),
#                bbox_inches='tight')
#    plt.savefig(os.path.join(os.getcwd(), '..', 'figures', filename + '.png'),
#                bbox_inches='tight', dpi=800)


def plot_four_rws(figsize,
                  colors = ['#001c54', '#E89818', '#8b0000'],
                  fill_color = (255 / 255, 223 / 255, 0 / 255, 5 / 255)):
    def download_and_resample(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        data = data.resample('D').ffill().dropna()  # Forward fill to handle any missing days
        return data

    usuk_data = download_and_resample('USDGBP=X', start="2022-10-01", end="2024-06-30")
    rw_usuk_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_usuk.zip')
    random_walks_usuk = pd.read_csv(rw_usuk_path, header=None, compression='zip')

    btc_data = download_and_resample('BTC-USD', start="2022-10-01", end="2024-06-30")
    rw_btc_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_btc.zip')
    random_walks_btc = pd.read_csv(rw_btc_path, header=None, compression='zip')

    nasdaq_data = download_and_resample('^IXIC', start="2022-10-01", end="2024-06-30")
    rw_nasdaq_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_nasdaq.zip')
    random_walks_nasdaq = pd.read_csv(rw_nasdaq_path, header=None, compression='zip')

    nvidia_data = download_and_resample('NVDA', start="2022-10-01", end="2024-06-30")
    rw_nvidia_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_nvidia.zip')
    random_walks_nvidia = pd.read_csv(rw_nvidia_path, header=None, compression='zip')

    def adjust_index(data, rw_data):
        end_date = data.index[-1]
        start_date = end_date + pd.DateOffset(1)  # Start the random walk data the day after the end_date
        new_index = pd.date_range(start=start_date, periods=len(rw_data), freq='D')
        rw_data.index = new_index
        return rw_data

    random_walks_usuk = adjust_index(usuk_data, random_walks_usuk)
    random_walks_btc = adjust_index(btc_data, random_walks_btc)
    random_walks_nasdaq = adjust_index(nasdaq_data, random_walks_nasdaq)
    random_walks_nvidia = adjust_index(nvidia_data, random_walks_nvidia)

    def _print_stats(label, arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            print(f"{label}: empty")
            return
        mean = float(np.mean(arr))
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        z = mean / sd if sd not in (0, float("nan")) else float("nan")
        print(f"{label}: mean={mean:.4f}, min={amin:.4f}, max={amax:.4f}, sd={sd:.4f}, z={z:.4f}")

    # Final-day stats
    _print_stats("[plot_three_supplementary_rws] BTC RW final-day distribution", random_walks_btc.iloc[-1].values)
    _print_stats("[plot_three_supplementary_rws] NASDAQ RW final-day distribution", random_walks_nasdaq.iloc[-1].values)
    _print_stats("[plot_three_supplementary_rws] NVDA RW final-day distribution", random_walks_nvidia.iloc[-1].values)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize
                                                 )

    usuk_data['Close'].plot(ax=ax1, color=colors[0])
    random_walks_usuk.min(axis=1).plot(ax=ax1, color=colors[1], alpha=1, linestyle='--')
    random_walks_usuk.median(axis=1).plot(ax=ax1, color='k', alpha=1, linestyle='--')
    random_walks_usuk.max(axis=1).plot(ax=ax1, color=colors[2], linestyle='--')
    random_walks_usuk.quantile(0.05, axis=1).plot(ax=ax1, color='k', linestyle='--', alpha=1, linewidth=0.75)
    random_walks_usuk.quantile(0.95, axis=1).plot(ax=ax1, color='k', linestyle='--', alpha=1, linewidth=0.75)

    btc_data['Close'].plot(ax=ax2, color=colors[0])
    random_walks_btc.min(axis=1).plot(ax=ax2, color=colors[1], alpha=1, linestyle='--')
    random_walks_btc.median(axis=1).plot(ax=ax2, color='k', alpha=1, linestyle='--')
    random_walks_btc.max(axis=1).plot(ax=ax2, color=colors[2], linestyle='--')
    random_walks_btc.quantile(0.05, axis=1).plot(ax=ax2, color='k', linestyle='--', alpha=1, linewidth=0.75)
    random_walks_btc.quantile(0.95, axis=1).plot(ax=ax2, color='k', linestyle='--', alpha=1, linewidth=0.75)

    nasdaq_data['Close'].plot(ax=ax3, color=colors[0])
    random_walks_nasdaq.min(axis=1).plot(ax=ax3, color=colors[1], alpha=1, linestyle='--')
    random_walks_nasdaq.median(axis=1).plot(ax=ax3, color='k', alpha=1, linestyle='--')
    random_walks_nasdaq.max(axis=1).plot(ax=ax3, color=colors[2], linestyle='--')
    random_walks_nasdaq.quantile(0.05, axis=1).plot(ax=ax3, color='k', linestyle='--', alpha=1, linewidth=0.75)
    random_walks_nasdaq.quantile(0.95, axis=1).plot(ax=ax3, color='k', linestyle='--', alpha=1, linewidth=0.75)

    nvidia_data['Close'].plot(ax=ax4, color=colors[0])
    random_walks_nvidia.min(axis=1).plot(ax=ax4, color=colors[1], alpha=1, linestyle='-')
    random_walks_nvidia.median(axis=1).plot(ax=ax4, color='k', alpha=1, linestyle='--')
    random_walks_nvidia.max(axis=1).plot(ax=ax4, color=colors[2], linestyle='--')
    random_walks_nvidia.quantile(0.05, axis=1).plot(ax=ax4, color='k', linestyle='--', alpha=1, linewidth=0.75)
    random_walks_nvidia.quantile(0.95, axis=1).plot(ax=ax4, color='k', linestyle='--', alpha=1, linewidth=0.75)

    legend_elements = [
        Line2D([0], [0], color=colors[2], linestyle='--',
               label=r'Max', lw=2),
        Line2D([0], [0], color=colors[1], linestyle='--',
               label=r'Min', lw=2),
        Line2D([0], [0], color=colors[0], linestyle='-',
               label=r'Insample', lw=2),
        Line2D([0], [0], color='k', linestyle='--',
               label=r'Median', lw=2),
        Line2D([0], [0], color='k', linestyle='--', alpha=1, linewidth=0.75,
               label=r'95th Percentile', lw=2),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Range')
    ]
    ax1.legend(handles=legend_elements, loc='lower left', frameon=True,
               fontsize=11.25, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=3
               )

    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax4.set_xlabel('')
    ax1.grid(which="major", linestyle='--', alpha=0.225)
    ax2.grid(which="major", linestyle='--', alpha=0.225)
    ax3.grid(which="major", linestyle='--', alpha=0.225)
    ax4.grid(which="major", linestyle='--', alpha=0.225)
    ax1.set_title('a.', loc='left', fontsize=22, y=1.035)
    ax2.set_title('b.', loc='left', fontsize=22, y=1.035)
    ax3.set_title('c.', loc='left', fontsize=22, y=1.035)
    ax4.set_title('d.', loc='left', fontsize=22, y=1.035)

    ax1.fill_between(random_walks_usuk.index,
                     random_walks_usuk.min(axis=1),
                     random_walks_usuk.max(axis=1),
                     color=fill_color
                     )
    ax2.fill_between(random_walks_btc.index,
                     random_walks_btc.min(axis=1),
                     random_walks_btc.max(axis=1),
                     color=fill_color)
    ax3.fill_between(random_walks_nasdaq.index,
                     random_walks_nasdaq.min(axis=1),
                     random_walks_nasdaq.max(axis=1),
                     color=fill_color)
    ax4.fill_between(random_walks_nvidia.index,
                     random_walks_nvidia.min(axis=1),
                     random_walks_nvidia.max(axis=1),
                     color=fill_color)

    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'${y / 1000:.0f}k'))
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1:.0f}'))
    ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'${y / 1:.0f}'))
    ax1.set_ylabel('US ($) /UK (£) Exchange Rate', fontsize=14)
    ax2.set_ylabel('Bitcoin Price', fontsize=14)
    ax3.set_ylabel('NASDAQ Composite', fontsize=14)
    ax4.set_ylabel('NVidia Share Price', fontsize=14)

    inset_ax = inset_axes(ax1, width="40%", height="25%", loc='upper left', borderpad=2.5)
    sns.histplot(random_walks_usuk.iloc[-1], ax=inset_ax,
                 color=colors[0], bins=12,
                 legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax.set_xlabel('US ($) / GBP (£)')
    inset_ax.set_ylabel('Frequency')
    inset_ax.set_axisbelow(True)
    inset_ax.yaxis.set_label_position("right")
    inset_ax.yaxis.tick_right()
    inset_ax.spines['left'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
#    inset_ax.grid(which="both", linestyle='--', alpha=0.3)
#    sns.despine(ax=inset_ax, left=True, top=True, right=False, bottom=False)

    inset_ax = inset_axes(ax2, width="40%", height="25%", loc='upper left', borderpad=2.5)
    sns.histplot(random_walks_btc.iloc[-1], ax=inset_ax,
                 color=colors[0], bins=15,
                 legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax.set_xlabel('Bitcoin Price')
    inset_ax.set_ylabel('Frequency')
    inset_ax.set_axisbelow(True)
    inset_ax.yaxis.set_label_position("right")
    inset_ax.yaxis.tick_right()
    inset_ax.spines['left'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
#    inset_ax.grid(which="both", linestyle='--', alpha=0.3)
#    sns.despine(ax=inset_ax, left=True, top=True, right=False, bottom=False)

    inset_ax = inset_axes(ax3, width="40%", height="25%", loc='upper left', borderpad=2.5)
    sns.histplot(random_walks_nasdaq.iloc[-1], ax=inset_ax,
                 color=colors[0], bins=15,
                 legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax.set_xlabel('NASDAQ Composite')
    inset_ax.set_ylabel('Frequency')
    inset_ax.set_axisbelow(True)
    inset_ax.yaxis.set_label_position("right")
    inset_ax.yaxis.tick_right()
    inset_ax.spines['left'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
#    inset_ax.grid(which="both", linestyle='--', alpha=0.3)
#    sns.despine(ax=inset_ax, left=True, top=True, right=False, bottom=False)

    inset_ax = inset_axes(ax4, width="40%", height="25%", loc='upper left', borderpad=2.5)
    sns.histplot(random_walks_nvidia.iloc[-1], ax=inset_ax,
                 color=colors[0], bins=15,
                 legend=False, alpha=0.9,
                 common_norm=False
                 )
    inset_ax.set_xlabel('NVidia Share Price')
    inset_ax.set_ylabel('Frequency')
    inset_ax.set_axisbelow(True)
    inset_ax.yaxis.set_label_position("right")
    inset_ax.yaxis.tick_right()
    inset_ax.spines['left'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
#    inset_ax.grid(which="both", linestyle='--', alpha=0.3)
#    sns.despine(ax=inset_ax, left=True, top=True, right=False, bottom=False)

    sns.despine(ax=ax1, left=False, top=True, right=True, bottom=False)
    sns.despine(ax=ax2, left=False, top=True, right=True, bottom=False)
    sns.despine(ax=ax3, left=False, top=True, right=True, bottom=False)
    sns.despine(ax=ax4, left=False, top=True, right=True, bottom=False)
    plt.tight_layout()
    filename = 'four_rws'
    plt.savefig(os.path.join(os.getcwd(), '..', 'figures', filename + '.pdf'),
                bbox_inches='tight')

    # Final-day descriptive stats (min/mean/max/sd/z) for each random walk
    _print_stats("[plot_three_supplementary_rws] BTC RW final-day", random_walks_btc.iloc[-1].values)
    _print_stats("[plot_three_supplementary_rws] NASDAQ RW final-day", random_walks_nasdaq.iloc[-1].values)
    _print_stats("[plot_three_supplementary_rws] NVDA RW final-day", random_walks_nvidia.iloc[-1].values)

def load_collisions():
    chunk_size = 10000
    path = os.path.join(os.getcwd(),
                        '..',
                        'data',
                        'collisions',
                        'output_list_32_R.csv')
    csv_reader = pd.read_csv(path, chunksize=chunk_size)
    collisions = pd.DataFrame()
    for i, chunk in enumerate(csv_reader):
        temp_df = pd.concat([chunk.min(axis=1),
                             chunk.median(axis=1),
                             chunk.max(axis=1)],
                            axis=1)
        collisions = pd.concat([collisions, temp_df],
                               axis=0)
    final_collisions = chunk.iloc[-1]
    return collisions, final_collisions


def plot_collisions(figure_path):
    chunk_size = 10000
    path = os.path.join(os.getcwd(),
                        '..',
                        'data',
                        'collisions',
                        'output_list_32_R.csv')
    csv_reader = pd.read_csv(path, chunksize=chunk_size)
    df = pd.DataFrame()
    for i, chunk in enumerate(csv_reader):
        temp_df = pd.concat([chunk.min(axis=1),
                             chunk.median(axis=1),
                             chunk.max(axis=1)],
                            axis=1)
        df = pd.concat([df, temp_df], axis=0)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    colors = ['#001c54', '#E89818', '#8b0000']
    fill_color = (255 / 255, 223 / 255, 0 / 255, 19 / 255)

    df[0].plot(color=colors[1], linestyle='--', ax=ax2)
    df[1].plot(color=colors[0], linestyle='-', ax=ax2)
    df[2].plot(color=colors[2], linestyle='--', ax=ax2)

    ax2.fill_between(df.index, df[0], df[2],
                     color=fill_color)

    ax1.grid(which="both", linestyle='--', alpha=0.225)
    ax2.grid(which="both", linestyle='--', alpha=0.225)

    ax1.tick_params(width=1, length=8, axis='both', which='major', labelsize=14)
    ax2.tick_params(width=1, length=8, axis='both', which='major', labelsize=14)
    ax1.set_title('a.', loc='left', fontsize=24, y=1.0)
    ax2.set_title('b.', loc='left', fontsize=24, y=1.0)

    legend_elements2 = [
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'Min Collisions', alpha=0.7),
        Line2D([0], [0], color=colors[2], lw=2, linestyle='-',
               label=r'Max Collions', alpha=0.8),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='--',
               label=r'Median Collisions', alpha=0.7),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Seed Variance')]
    ax2.legend(handles=legend_elements2, loc='upper left', frameon=True,
               fontsize=13, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
               )
    ax2.hlines(116, df.index[0], df.index[-1],
               color='k', linestyle='--', alpha=.75)
    ax2.set_xlim(0, 1000000)


    ax2.annotate('Expectation',
                 xy=(125000, 118), xytext=(125000, 118),
                 fontsize=13, ha='center', va='bottom',
                )

    final_collisions = chunk.iloc[-1]

    nbins = 25
    sns.histplot(final_collisions,
                 ax=ax1,
                 color=colors[0],
                 bins=nbins,
                 alpha=0.9
                )
    ax1_twin = ax1.twinx()
    sns.kdeplot(final_collisions,
                ax=ax1_twin,
                color=colors[1],
                linestyle='--',
                linewidth=2
               )
    ax1.set_ylim(0, 1500)
    ax1_twin.set_ylim(0, 0.045)

    ax1.annotate(r'$\mu$ = ' + str(np.round(np.mean(final_collisions), 2)) + r', $\sigma$ = ' +
                 str(np.round(np.std(final_collisions), 3)),
                 xy=(0.5, 0.875), xytext=(0.5, 0.925), xycoords='axes fraction',
                 fontsize=13, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                 arrowprops=dict(arrowstyle='-[, widthB=9.0, lengthB=1',
                                lw=1.0))
    ax2.set_xlabel('Sample size', fontsize=16)
    ax2.set_ylabel('Number of 32-bit collisions', fontsize=16)
    ax1_twin.tick_params(width=1, length=8, axis='both', which='major', labelsize=14)
    ax1.set_xlabel('Number of collisions', fontsize=16)
    ax1.set_ylabel('Count of collisions', fontsize=16)
    ax1_twin.set_ylabel('Density of collisions', fontsize=16)
    plt.tight_layout()

    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    filename='collisions'
    plt.savefig(os.path.join(figure_path, filename + '.pdf'),
                bbox_inches='tight')


def plot_four_simple_examples(figure_path,
                              figsize,
                              colors = ['#001c54', '#E89818', '#8b0000'],
                              fill_color = (254, 208, 126, 10/255),
                              ):
    df_sir = pd.read_csv(os.path.join(os.getcwd(),
                                      '..',
                                      'data',
                                      'sir',
                                      'sir_seeds_1dp.csv')
                         )
    df_buffon = pd.read_csv(os.path.join(os.getcwd(),
                                         '..',
                                         'data',
                                         'needles',
                                         'results',
                                         'throw100_25000_5000seeds.csv'),
                            names=['Throws', 'Min', '25th_PC',
                                   'Median', '75th_PC', 'Max']
                            )
    df_collisions = pd.read_csv(os.path.join(os.getcwd(),
                                             '..',
                                             'data',
                                             'collisions',
                                             'stats_32bit_rowwise.csv'
                                             )
                                )
    df_collisions_finalrow = pd.read_csv(os.path.join(os.getcwd(),
                                                      '..',
                                                      'data',
                                                      'collisions',
                                                      'stats_32_final_row.csv')
                                         )
    df_solow = pd.read_csv(os.path.join(os.getcwd(),
                                        '..',
                                        'data',
                                        'solow',
                                        'solow_growth_results.zip'),
                           compression = 'zip'
                           )
    df_solow = df_solow.groupby('Time').agg({
        'Capital Stock': ['min', 'max', 'median'],
        'Labor': ['min', 'max', 'median'],
        'Output': ['min', 'max', 'median'],
        'Savings Rate': ['min', 'max', 'median'],
        'Depreciation Rate': ['min', 'max', 'median'],
        'TFP': ['min', 'max', 'median']
    }).reset_index()

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(8, 2, figure=fig, )

    ax1 = fig.add_subplot(gs[:4, 0])
    ax2 = fig.add_subplot(gs[:4, 1])
    ax3 = fig.add_subplot(gs[4:, 0])
    ax4 = fig.add_subplot(gs[4:6, 1])
    ax5 = fig.add_subplot(gs[6:8, 1])

    # Remove the extra ax4 subplot created in the 3x2 grid layout
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    nbins = 20

    #################
    # Figure 1a here#
    #################
    df_sir['Infected_min'].plot(ax=ax1, color=colors[1], linestyle='-')
    df_sir['Infected_med'].plot(ax=ax1, color=colors[0])
    df_sir['Infected_max'].plot(ax=ax1, color=colors[2])
    ax1.fill_between(df_sir.index, df_sir['Infected_min'], df_sir['Infected_max'],
                     color=fill_color)
    ax1.set_xlim(-25, 1450)
    ax1.set_xlabel('Time', fontsize=16)
    ax1.set_ylabel(r'Fraction Infected', fontsize=16)
    legend_elements1 = [
        Line2D([0], [0], color=colors[2], lw=2, linestyle='--',
               label=r'Max'),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='--',
               label=r'Min'),
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'Median'),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Variance')]
    ax1.legend(handles=legend_elements1, loc='upper right', frameon=True,
               fontsize=11, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
               )
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 10:.0f}%'))

    #################
    # Figure 1b here#
    #################
    df_buffon = df_buffon.set_index('Throws')
    df_buffon = df_buffon[45:]
    ax2.plot(df_buffon['Min'], color=colors[1], alpha=0.8, linestyle='--')
    ax2.plot(df_buffon['Max'], color=colors[2], alpha=0.8, linestyle='--')
    ax2.set_xlim(0, df_buffon.index[-1] + 500)
    ax2.set_ylim(2.225, 4.5)
    ax2.hlines(math.pi, df_buffon.index[0] + 500, df_buffon.index[-1],
               color=colors[0], linestyle='-')
    ax2.fill_between(df_buffon.index,
                     df_buffon['Min'],
                     df_buffon['Max'],
                     color=fill_color)
    ax2.set_xlabel('Number of Throws', fontsize=16)
    ax2.set_ylabel(r'Estimate of $\mathrm{\pi}$', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14,
                    width=1, length=8)
    legend_elements2 = [
        Line2D([0], [0], color=colors[2], linestyle='--',
               label=r'Max', lw=2),
        Line2D([0], [0], color=colors[1], linestyle='--',
               label=r'Min', lw=2),
        Line2D([0], [0], color=colors[0], linestyle='-',
               label=r'$\mathrm{\pi}$', lw=2),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Variance')
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', frameon=True,
               fontsize=11, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
               )
    #################
    # Figure 1c here#
    #################

    df_collisions['min'].drop_duplicates().plot(color=colors[1], linestyle='--', ax=ax3)
    df_collisions['median'].drop_duplicates().plot(color=colors[0], linestyle='-', ax=ax3)
    df_collisions['max'].drop_duplicates().plot(color=colors[2], linestyle='--', ax=ax3)

    ax3.fill_between(df_collisions.drop_duplicates().index,
                     df_collisions.drop_duplicates()['min'],
                     df_collisions.drop_duplicates()['max'],
                     color=fill_color)

    legend_elements2 = [
        Line2D([0], [0], color=colors[2], lw=2, linestyle='--',
               label=r'Max'),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='--',
               label=r'Min'),
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'Median'),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Variance')]
    ax3.legend(handles=legend_elements2, loc='lower right', frameon=True,
               fontsize=11, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
               )
    # ax3.hlines(116, collisions.index[-1]/3, collisions.index[-1],
    #           color='k', linestyle='--', alpha=.75)
    ax3.set_xlim(0, 1000000)
    #    ax3.annotate('Expectation',
    #                 xy=(125000, 118), xytext=(125000, 118),
    #                 fontsize=13, ha='center', va='bottom',
    #                )
    ax3.set_ylabel('Number of 32-bit collisions', fontsize=16)
    ax3.set_xlabel('Sample size', fontsize=16)

    ax3_inset = ax3.inset_axes([0.035, 0.535, 0.375, 0.35], transform=ax3.transAxes)
    sns.histplot(df_collisions_finalrow['x'],
                 ax=ax3_inset,
                 color=colors[0],
                 bins=nbins,
                 alpha=0.9
                 )
    ax3_twin = ax3_inset.twinx()
    sns.kdeplot(df_collisions_finalrow['x'],
                ax=ax3_twin,
                color=colors[1],
                linestyle='--',
                linewidth=2
                )
    #    ax3_inset.set_ylim(0, 1500)
    #    ax3_twin.set_ylim(0, 0.045)
    ax3_inset.annotate(r'$\mu$ = ' + str(np.round(np.mean(df_collisions_finalrow['x']), 1)) + r', $\sigma$ = ' +
                       str(np.round(np.std(df_collisions_finalrow['x']), 1)),
                       xy=(0.5, 1), xytext=(0.5, 1.1),
                       xycoords='axes fraction',
                       fontsize=11, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                       arrowprops=dict(arrowstyle='-[, widthB=5.0, lengthB=1',
                                       lw=1.0)
                       )

    ax3_twin.tick_params(width=1, length=8, axis='both', which='major', labelsize=14)
    ax3.set_xlabel('Number of Draws', fontsize=16)
    ax3.set_ylabel('Count of Collisions', fontsize=16)
    #    ax3_twin.set_ylabel('Density of collisions', fontsize=16)
    ax3.set_axisbelow(True)
    ax3_twin.set_ylabel('')
    ax3_inset.set_ylabel('')
    ax3_twin.set_xlabel('')
    ax3_inset.set_xlabel('')
    ax3_inset.set_yticks([])
    ax3_twin.set_yticks([])

    ##################
    # Figure 1d here #
    ##################

    df_solow.columns = ['_'.join(col).strip() for col in df_solow.columns.values]
    df_solow['Output_min'].plot(ax=ax4, color=colors[1], linestyle='--')
    df_solow['Output_median'].plot(ax=ax4, color=colors[0])
    df_solow['Output_max'].plot(ax=ax4, color=colors[2], linestyle='--')

    df_solow['Capital Stock_min'].plot(ax=ax5, color=colors[1], linestyle='--')
    df_solow['Capital Stock_median'].plot(ax=ax5, color=colors[0])
    df_solow['Capital Stock_max'].plot(ax=ax5, color=colors[2], linestyle='--')

    ax4.fill_between(df_solow.index,
                     df_solow['Output_min'],
                     df_solow['Output_max'],
                     color=fill_color)
    ax5.fill_between(df_solow.index,
                     df_solow['Capital Stock_min'],
                     df_solow['Capital Stock_max'],
                     color=fill_color)

    ax4.legend(handles=legend_elements1, loc='upper left', frameon=True,
               fontsize=11, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
               )
    ax5.legend(handles=legend_elements1, loc='upper left', frameon=True,
               fontsize=11, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
               )
    ax4.set_xlabel('')
    ax5.set_xlabel('Time', fontsize=14)
    ax4.set_ylabel('Output', fontsize=14)
    ax5.set_ylabel('Capital Stock', fontsize=14)
    ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'${y / 1000:.0f}k'))
    ax5.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'${y / 1000:.0f}k'))

    ##############
    # aesthetics #
    ##############

    ax1.set_title('a.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax2.set_title('b.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax3.set_title('c.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax4.set_title('d.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax5.set_title('e.', loc='left', fontsize=letter_fontsize, y=1.0)

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.grid(which="both", linestyle='--', alpha=0.225)
        ax.set_zorder(3)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=14, width=1, length=8)
        ax.tick_params(width=1, length=8, axis='both', which='major', labelsize=14)

    for ax in [ax3_twin, ax3_inset]:
        sns.despine(ax=ax,
                    left=True,
                    right=True,
                    top=True,
                    bottom=False)
    # plt.tight_layout()
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3)
    sns.despine(ax=ax4)
    sns.despine(ax=ax5)
    filename = 'four_simple_examples'
    plt.savefig(os.path.join(figure_path, filename + '.pdf'),
                bbox_inches='tight')


def plot_mvprobit(figure_path,
                  figsize,
                  colors = ['#001c54', (255/255, 223/255, 0/255, 3/255), '#8b0000'],
                  fill_color = (254, 208, 126, 10/255)
                  ):
    df = pd.read_csv(os.path.join(os.getcwd(),
                                  '..',
                                  'data',
                                  'mvprobit',
                                  'results_school_total_draws150_total_seeds1000.csv')
                    )
    new_df = pd.DataFrame(index=df['draws'].unique())
    for draw in df['draws'].unique():
        new_df.at[draw, 'Min'] = df[df['draws']==draw]['rho21'].min()
        new_df.at[draw, 'Max'] = df[df['draws']==draw]['rho21'].max()
        new_df.at[draw, 'Median'] = df[df['draws']==draw]['rho21'].median()

    fig, (ax1) = plt.subplots(1, 1, figsize=figsize)
    mpl.rcParams['font.family'] = 'Helvetica'
    new_df['Median'].plot(ax=ax1, color=colors[0])
    new_df['Max'].plot(ax=ax1, linestyle='--', color=colors[2])
    new_df['Min'].plot(ax=ax1, linestyle='--', color=colors[2])
    ax1.grid(which = "both", linestyle='--', alpha=0.225)
#    ax1.set_title('a.', loc='left', fontsize=18)
    ax1.set_xlabel('Number of Draws', fontsize=14)
    ax1.set_ylabel("Simulated Maximum \n " +
                   r"Likelihood Estimate of $\rho_{21}$", fontsize=14)
    ax1.fill_between(new_df.index, new_df.min(axis=1),
                     new_df.max(axis=1), color=fill_color)
    legend_elements2 = [
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'Median'),
       Line2D([0], [0], color=colors[2], lw=2, linestyle='--',
               label=r'Min/Max'),
       Line2D([0], [0], color='k',linewidth=0.5, linestyle='dashed',
               label=r'ML Estimate'),
        Patch(facecolor=fill_color, edgecolor=(0,0,0,1),
              label=r'Seed Variance')]
    ax1.legend(handles=legend_elements2, loc='upper right', frameon=True,
               fontsize=12, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
              )
    plt.hlines(y=-0.270, xmin=2, xmax=150,
               color='k', linewidth=1, linestyle='dashed', alpha=1)
    ax1.set_xlim(0, 152)
    print(r'Min value of $\rho_{21}$ at 2 draws:',
          df[df['draws'] == 2]['rho21'].min())
    print(r'Max value of $\rho_{21}$ at 2 draws:',
          df[df['draws'] == 2]['rho21'].max())
    print(r'Min value of $\rho_{21}$ at 150 draws:',
          df[df['draws'] == 150]['rho21'].min())
    print(r'Max value of $\rho_{21}$ at 150 draws:',
          df[df['draws'] == 150]['rho21'].max())
    plt.savefig(os.path.join(figure_path, 'mvprobit' + '.pdf'),
                bbox_inches = 'tight')


def plot_two_inference(figure_path):
    results_path = os.path.join(os.getcwd(), '..', 'data', 'mcs',
                                'results', 'merged_files',
                                'merged_csvs.csv')
    df = pd.read_csv(results_path, index_col=False)
    min_series = df.min(axis=1).sort_values().reset_index()[0]
    max_series = df.max(axis=1).sort_values().reset_index()[0]
    med_series = df.median(axis=1).sort_values().reset_index()[0]
    all_in_one_list = list(df.melt().drop('variable',axis=1).rename({'value':'A'},axis=1)['A'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    figure_path = os.path.join(os.getcwd(), '..', 'figures')
    colors = ['#001c54', '#E89818']
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    ax1.plot(min_series.index, min_series, color=colors[1], linestyle='-', alpha=0.8)
    ax1.plot(max_series.index, max_series, color=colors[1], linestyle='-', alpha=0.8)
    ax1.plot(med_series.index, med_series, color=colors[0], linestyle='--', alpha=0.8)
    ax1.set_ylabel(r'Effect Size ($\rm{\hat{\beta}}$)', fontsize=label_fontsize)
    ax1.set_xlabel(r'Specification (n)', fontsize=label_fontsize)
    ax2.set_xlabel(r'Effect Size ($\rm{\hat{\beta}}$)', fontsize=label_fontsize)
    legend_elements1 = [Line2D([0], [0], color=colors[0], lw=1, linestyle='--',
                               label=r'Median', alpha=0.7),
                        Line2D([0], [0], color=colors[1], lw=1, linestyle='-',
                               label=r'Bounds', alpha=0.7), ]
    ax1.legend(handles=legend_elements1, loc='upper left', frameon=True,
              fontsize=label_fontsize - 4, framealpha=1, facecolor='w',
              edgecolor=(0, 0, 0, 1))
    ax1.hlines(y=0, xmin=ax1.get_xlim()[0], xmax=ax1.get_xlim()[1],
               color='k', linewidth=1, linestyle='--', alpha=0.5)
    ax1.fill_between(min_series.index, min_series, y2=max_series,
                     color=colors[1], alpha=0.075)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    inset_ax = inset_axes(ax1,
                          width="41%",
                          height="90%",
                          loc='lower right',
                          bbox_to_anchor=(-0.005, 0.075, 1, 0.3),
                          bbox_transform=ax1.transAxes)
    inset_ax.set_xlabel(r'Effect Size ($\rm{\hat{\beta}}$)',
                        fontsize=label_fontsize - 5, labelpad=-3)
    inset_ax.set_ylabel('Frequency', fontsize=label_fontsize - 5)
    inset_ax.hist(all_in_one_list, bins=50, color=colors[0],
                  alpha=0.6, edgecolor='k')
    ax1.set_title('A.', loc='left', fontsize=letter_fontsize, y=1.0, x=-.05)
    ax2.set_title('B.', loc='left', fontsize=letter_fontsize, y=1.0, x=-.05)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.grid(which = "both", linestyle='--', alpha=0.225)
    ax2.grid(which = "both", linestyle='--', alpha=0.225)

    btc_data = pdr.get_data_yahoo('BTC-USD', start="2021-05-08", end="2023-05-15")
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    colors = ['#001c54', (255/255, 223/255, 0/255, 10/255), '#8b0000']
    size = 366
    rw_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks.csv')
    random_walks = pd.read_csv(rw_path, header=None)
    btc_data = btc_data/1000
    random_walks = random_walks/1000
    index= pd.DataFrame(index=btc_data.index + pd.DateOffset(len(btc_data)+1))[0:size+1].index
    random_walks.index = index
    btc_data['Close'].plot(ax=ax2, color = colors[0])
    random_walks.min(axis=1).plot(ax=ax2, color = colors[2])
    random_walks.max(axis=1).plot(ax=ax2, color = colors[2])
    random_walks.median(axis=1).plot(ax=ax2, color = 'k', linestyle='--', alpha=.5)
    legend_elements2 = [
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'In-sample', alpha=0.7),
       Line2D([0], [0], color=colors[2], lw=2, linestyle='-',
               label=r'Min/Max', alpha=0.7),
        Line2D([0], [0], color='k', lw=2, linestyle='--',
               label=r'Median', alpha=0.7),
        Patch(facecolor=colors[1], edgecolor=(0,0,0,1),
                              label=r'Range')]
    ax2.legend(handles=legend_elements2, loc='upper left', frameon=True,
               fontsize=label_fontsize-4, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
              )
    ylabels = ['${:,.0f}'.format(x) + 'k' for x in ax2.get_yticks()]
    ax2.set_yticklabels(ylabels)
    ax2.set_xlabel('')
    ax2.fill_between(random_walks.index, random_walks.min(axis=1),
                     random_walks.max(axis=1), color=colors[1])
    ax1.set_title('A.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax2.set_title('B.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax2.tick_params(width=1, length=8)
    ax2.set_ylabel(r'Price', fontsize=label_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_axisbelow(True)
    ax2.grid(which="major", linestyle='--', alpha=0.225)
    ax2.set_zorder(3)
    sns.despine()
    plt.tight_layout()
    filename = 'plot_two_inference'
    plt.savefig(os.path.join(figure_path, filename + '.pdf'),
                bbox_inches = 'tight')
    sns.despine()
    plt.savefig(os.path.join(figure_path, 'mcs_and_erhlich_seeds.pdf'), bbox_inches='tight')


def combine_buffon_and_rw(figure_path):
    yf.pdr_override()
    btc_data = pdr.get_data_yahoo('BTC-USD', start="2021-05-08", end="2023-05-15")
    df = pd.read_csv(os.path.join(os.getcwd(),
                                  '..', 'data',
                                  'needles',
                                  'results',
                                  'throw100_25000_5000seeds.csv'),
                    names = ['Throws', 'Min', '25th_PC',
                    'Median', '75th_PC', 'Max'])
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    df = df.set_index('Throws')
    df = df[45:]
    colors = ['#001c54', (255/255, 223/255, 0/255, 10/255), '#8b0000']
    ax1.plot(df['Min'], color=colors[2])
    ax1.plot(df['Max'], color=colors[0])
    ax1.set_xlim(0, df.index[-1]+500)
    ax1.set_ylim(2.225, 4.5)
    ax1.hlines(math.pi, df.index[0]+500, df.index[-1],
               color='k', linestyle='--', alpha=.5)
    ax1.fill_between(df.index, df['Min'], df['Max'],
                     color=colors[1])
    ax1.set_xlabel('Number of Throws', fontsize=label_fontsize)
    ax1.set_ylabel(r'Estimate of $\mathrm{\pi}$', fontsize=label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(width=1, length=8)
    legend_elements1 = [
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'Upper Limit', alpha=0.7),
       Line2D([0], [0], color=colors[2], lw=2, linestyle='-',
               label=r'Lower Limit', alpha=0.7),
        Line2D([0], [0], color='k', lw=2, linestyle='--',
               label=r'$\mathrm{\pi}$', alpha=0.7),
        Patch(facecolor=colors[1], edgecolor=(0,0,0,1),
                              label=r'Range')]
    ax1.legend(handles=legend_elements1, loc='upper right', frameon=True,
               fontsize=label_fontsize-4, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
              )
    ax1.set_axisbelow(True)
    ax1.grid(which = "both", linestyle='--', alpha=0.225)
    size = 366
    rw_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks.csv')
    random_walks = pd.read_csv(rw_path, header=None)
    btc_data = btc_data/1000
    random_walks = random_walks/1000
    index= pd.DataFrame(index=btc_data.index + pd.DateOffset(len(btc_data)+1))[0:size+1].index
    random_walks.index = index
    btc_data['Close'].plot(ax=ax2, color = colors[0])
    random_walks.min(axis=1).plot(ax=ax2, color = colors[2])
    random_walks.max(axis=1).plot(ax=ax2, color = colors[2])
    random_walks.median(axis=1).plot(ax=ax2, color = 'k', linestyle='--', alpha=.5)
    legend_elements2 = [
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'In-sample', alpha=0.7),
       Line2D([0], [0], color=colors[2], lw=2, linestyle='-',
               label=r'Min/Max', alpha=0.7),
        Line2D([0], [0], color='k', lw=2, linestyle='--',
               label=r'Median', alpha=0.7),
        Patch(facecolor=colors[1], edgecolor=(0,0,0,1),
                              label=r'Range')]
    ax2.legend(handles=legend_elements2, loc='upper left', frameon=True,
               fontsize=label_fontsize-4, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
              )
    ylabels = ['${:,.0f}'.format(x) + 'k' for x in ax2.get_yticks()]
    ax2.set_yticklabels(ylabels)
    ax2.set_xlabel('')
    ax2.fill_between(random_walks.index, random_walks.min(axis=1),
                     random_walks.max(axis=1), color=colors[1])
    ax1.set_title('A.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax2.set_title('B.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax2.tick_params(width=1, length=8)
    ax2.set_ylabel(r'Price', fontsize=label_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_axisbelow(True)
    ax2.grid(which="major", linestyle='--', alpha=0.225)
    ax2.set_zorder(3)
    sns.despine()
    plt.tight_layout()
    filename = 'buffon_and_rw'
    plt.savefig(os.path.join(figure_path, filename + '.pdf'),
                bbox_inches = 'tight')

'''
# Old FFC plotting code here
def plot_ffc(df, figure_path):
    def jointplotter(df, outcome, model, counter):
        df1 = df[(df['outcome']==outcome) &
                 (df['account']==model)]#[0:10000]
        title_list = ['a.', 'b.', 'c.', 'd.', 'e.', 'f.']
        title = title_list[counter]
        print(str(outcome) + '. Min beta :' + str(np.round(df1['beta'].min(), 4)) +
              '. Max beta: ' + str(np.round(df1['beta'].max(), 4)) +
              '. Min R2: ' + str(np.round(df1['r2_holdout'].min(), 4)) +
              '. Max R2: ' + str(np.round(df1['r2_holdout'].max(), 4))
              )
        g = sns.jointplot(x=df1['beta'],
                          y=df1['r2_holdout'],
                          kind='hex',
                          marginal_kws=dict(bins=25,
                                            color='w'))
        g.plot_joint(sns.kdeplot, color="r", levels=6)
        g.ax_marg_x.annotate(title, xy=(-0.1, .45), xycoords='axes fraction',
                    ha='left', va='center', fontsize=26)
        if counter in [0, 3]:
            g.ax_joint.set_ylabel(r'Pseudo R$^2$', fontsize=14)
        else:
            g.ax_joint.set_ylabel('')
        if counter in [3,4,5]:
            g.ax_joint.set_xlabel('Lagged Coefficient', fontsize=14)
        else:
            g.ax_joint.set_xlabel('')
        return g

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3)
    figurez = []
    outcomes = ['gpa', 'grit', 'materialHardship',
                'eviction', 'jobTraining', 'layoff']
    for outcome, counter in zip(outcomes, range(0, 6)):
        if counter > 2:
            model = 'logit'
        else:
            model = 'ols'
        figurez.append(jointplotter(df, outcome, model, counter))
        tmp = SeabornFig2Grid(figurez[counter], fig, gs[counter])
    figurez[0] = figurez[0].ax_joint.annotate('GPA', xy=(0.9, 0.05),
                                              xycoords='axes fraction',
                                              ha='left', va='center', fontsize=14)
    figurez[1] = figurez[1].ax_joint.annotate('Grit', xy=(0.878, 0.05),
                                              xycoords='axes fraction',
                                              ha='left', va='center', fontsize=14)
    figurez[2] = figurez[2].ax_joint.annotate('Material Hardship', xy=(0.56, 0.05),
                                              xycoords='axes fraction',
                                              ha='left', va='center', fontsize=14)
    figurez[3] = figurez[3].ax_joint.annotate('Eviction', xy=(0.805, 0.05),
                                              xycoords='axes fraction',
                                              ha='left', va='center', fontsize=14)
    figurez[4] = figurez[4].ax_joint.annotate('Job Training', xy=(0.72, 0.05),
                                              xycoords='axes fraction',
                                              ha='left', va='center', fontsize=14)
    figurez[5] = figurez[5].ax_joint.annotate('Layoff', xy=(0.86, 0.05),
                                              xycoords='axes fraction',
                                              ha='left', va='center', fontsize=14)
    gs.tight_layout(fig)
    gs.update(hspace=0.125, wspace=0.125)
#    plt.subplots_adjust(top=0.9, right=0.9)


    # Interestingly, this is broken with gs.
    plt.savefig(os.path.join(figure_path, 'ffc_seeds.pdf')
    , bbox_inches='tight'
    )
    plt.savefig(os.path.join(figure_path, 'ffc_seeds.svg')
    , bbox_inches='tight'
    )
    plt.savefig(os.path.join(figure_path, 'ffc_seeds.png')
    , bbox_inches='tight',# dpi=900,
    )
    plt.savefig(os.path.join(figure_path, 'ffc_seeds.tiff')
    , bbox_inches='tight',
    #dpi = 600,
    format = "tiff", pil_kwargs = {"compression": "tiff_lzw"}
                )
    #    plt.show()


class SeabornFig2Grid():
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

'''
def load_sympt(filename):
    def try_literal_eval(e):
        try:
            return ast.literal_eval(e)
        except ValueError:
            return [np.nan, np.nan, np.nan, np.nan, np.nan]

    df = pd.read_csv(os.path.join(os.getcwd(), '..', 'data', 'symptomtracker',
                                  filename), index_col=0)
    df['roc_auc'] = df['roc_auc'].apply(try_literal_eval)
    df['roc_auc_mean'] = np.mean(df['roc_auc'].tolist(), axis=1)
    mylist = df['roc_auc'].to_list()
    flat_list = [item for sublist in mylist for item in sublist]
    print(len(flat_list))
    return flat_list


def plot_ffc(ffc, figure_path=None):
    # Define the gridspec layout with adjusted wspace
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.1], hspace=0.25, wspace=0.4)  # Adjusted wspace

    # Create subplots using gridspec
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    colormap = 'Spectral_r'

    outcomes_meta = [
        ('gpa', 'ols', 'GPA'),
        ('grit', 'ols', 'Grit'),
        ('materialHardship', 'ols', 'Material Hardship'),
        ('eviction', 'logit', 'Eviction'),
        ('jobTraining', 'logit', 'Job Training'),
        ('layoff', 'logit', 'Layoff'),
    ]

    def _log_metric_stats(label, series, metric_name):
        series = pd.to_numeric(series, errors='coerce').dropna()
        if series.empty:
            print(f"{label}: no data for {metric_name}")
            return None
        stats = {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean()),
            'std': float(series.std(ddof=0)),
            'median': float(series.median()),
        }
        zscore = ((stats['median'] - stats['mean']) / stats['std']
                  if stats['std'] else np.nan)
        print(
            f"{label} {metric_name}: "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"median={stats['median']:.4f}, z-score={zscore:.4f}"
        )
        return stats

    for outcome, account_type, pretty in outcomes_meta:
        subset = ffc[(ffc['outcome'] == outcome) & (ffc['account'] == account_type)]
        label = f"{pretty} ({account_type})"
        _log_metric_stats(label, subset['beta'], 'beta')
        _log_metric_stats(label, subset['r2_holdout'], 'pseudo-r2')

    gpa = ffc[(ffc['outcome'] == 'gpa') & (ffc['account'] == 'ols')]
    hb2 = ax1.hexbin(gpa['beta'], gpa['r2_holdout'], cmap=colormap, gridsize=25,
                     mincnt=1, linewidths=0, edgecolor='w')

    grit = ffc[(ffc['outcome'] == 'grit') & (ffc['account'] == 'ols')]
    hb3 = ax2.hexbin(grit['beta'], grit['r2_holdout'], cmap=colormap, gridsize=25,
                     mincnt=1, linewidths=0, edgecolor='w')

    materialHardship = ffc[(ffc['outcome'] == 'materialHardship') & (ffc['account'] == 'ols')]
    hb6 = ax3.hexbin(materialHardship['beta'], materialHardship['r2_holdout'], cmap=colormap, gridsize=25,
                     mincnt=1, linewidths=0, edgecolor='w')

    eviction = ffc[(ffc['outcome'] == 'eviction') & (ffc['account'] == 'logit')]
    hb1 = ax4.hexbin(eviction['beta'], eviction['r2_holdout'], cmap=colormap, gridsize=25,
                     mincnt=1, linewidths=0, edgecolor='w')

    jobTraining = ffc[(ffc['outcome'] == 'jobTraining') & (ffc['account'] == 'logit')]
    hb4 = ax5.hexbin(jobTraining['beta'], jobTraining['r2_holdout'], cmap=colormap, gridsize=25,
                     mincnt=1, linewidths=0, edgecolor='w')

    layoff = ffc[(ffc['outcome'] == 'layoff') & (ffc['account'] == 'logit')]
    hb5 = ax6.hexbin(layoff['beta'], layoff['r2_holdout'], cmap=colormap, gridsize=25,
                     mincnt=1, linewidths=0, edgecolor='w')

    # Manually position the colorbar
    cbar_ax = fig.add_axes([0.85, 0.1085, 0.02, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(hb1, cax=cbar_ax, spacing='uniform', extend='max')
    #    cbar.set_label('Counts', fontsize=14)
    cbar.ax.set_title('Count', fontsize=14)

    for ax, title in zip([ax1, ax2, ax3, ax4, ax5, ax6],
                         ['a.', 'b.', 'c.', 'd.', 'e.', 'f.']):
        ax.set_axisbelow(True)
        ax.grid(which="both", linestyle='--', alpha=0.225)
        ax.set_title(title, loc='left', fontsize=18, y=1.025, x=-0.075, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=10)  # Reduced label size

    # Adjust the layout to minimize padding and avoid label overlap

    ax1.set_ylabel('Pseudo-R$^2$ (GPA)', fontsize=12)
    ax2.set_ylabel('Pseudo-R$^2$ (Grit)', fontsize=12)
    ax3.set_ylabel('Pseudo-R$^2$ (Material Hardship)', fontsize=12)
    ax4.set_ylabel('Pseudo-R$^2$ (Eviction)', fontsize=12)
    ax5.set_ylabel('Pseudo-R$^2$ (Job Training)', fontsize=12)
    ax6.set_ylabel('Pseudo-R$^2$ (Layoff)', fontsize=12)

    ax1.set_xlabel(r'$\mathrm{\hat{\beta}}$ GPA (Lagged)', fontsize=12)
    ax2.set_xlabel(r'$\mathrm{\hat{\beta}}$ Grit (Lagged)', fontsize=12)
    ax3.set_xlabel(r'$\mathrm{\hat{\beta}}$ Material Hardship (Lagged)', fontsize=12)
    ax4.set_xlabel(r'$\mathrm{\hat{\beta}}$ Eviction (Lagged)', fontsize=12)
    ax5.set_xlabel(r'$\mathrm{\hat{\beta}}$ Job Training (Lagged)', fontsize=12)
    ax6.set_xlabel(r'$\mathrm{\hat{\beta}}$ Layoff (Lagged)', fontsize=12)

    '''
    From the docker container:

    eviction,logit,0.014352564873968743,1.71382900923502,8544
    eviction,ols,0.01839766283470634,0.191094206453511,8544
    gpa,logit,NaN,NA,8544
    gpa,ols,0.10595694838274616,0.191098639078027,8544
    grit,logit,NaN,NA,8544
    grit,ols,0.014244423233962134,0.0276498416056555,8544
    jobTraining,logit,0.05071272180604092,0.757814349306344,8544
    jobTraining,ols,0.05330811724942219,0.146585227369163,8544
    layoff,logit,0.007744404997622412,0.34444620652094,8544
    layoff,ols,0.007013521417237656,0.057964715568005,8544
    materialHardship,logit,NaN,NA,8544
    materialHardship,ols,0.1738409252064158,0.371661682034748,8544
    '''
    ax1.axvline(0.191098639078027, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax1.axhline(0.10595694838274616, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax2.axvline(0.0276498416056555, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax2.axhline(0.014244423233962134, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax3.axhline(0.1738409252064158, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax3.axvline(0.371661682034748, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax4.axhline(0.014352564873968743, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax4.axvline(1.71382900923502, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax5.axhline(0.05071272180604092, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax5.axvline(0.757814349306344, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax6.axvline(0.34444620652094, linestyle='--', color='k', linewidth=1.1, alpha=0.5)
    ax6.axhline(0.007744404997622412, linestyle='--', color='k', linewidth=1.1, alpha=0.5)

    # 'a.' annotation'

    ax1.annotate('Seed: 08544',
                 xy=(0.191098639078027,
                     0.10595694838274616),
                 xytext=(0.15,
                         0.065),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="k", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=-0.25",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1))

    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space on the right for the colorbar
    plt.savefig(os.path.join(figure_path, 'ffc_seeds.pdf'), bbox_inches='tight')


def covid_plotter(list1, list2, list3, list4, figure_path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    colors = ['#001c54', '#E89818']
    nbins=18
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    csfont = {'fontname': 'Helvetica'}
    sns.distplot(list1, hist_kws={'facecolor': colors[0],
                                  'edgecolor': 'k',
                                  'alpha': 0.7},
                 kde_kws={'color': colors[1]}, ax=ax1, bins=nbins)
    sns.distplot(list2, hist_kws={'facecolor': colors[1],
                                  'edgecolor': 'k',
                                  'alpha': 0.7},
                 kde_kws={'color': colors[0]}, ax=ax2, bins=nbins)
    sns.distplot(list3, hist_kws={'facecolor': colors[0],
                                  'edgecolor': 'k',
                                  'alpha': 0.7},
                 kde_kws={'color': colors[1]}, ax=ax3, bins=nbins)
    sns.distplot(list4, hist_kws={'facecolor': colors[1],
                                  'edgecolor': 'k',
                                  'alpha': 0.7},
                 kde_kws={'color': colors[0]}, ax=ax4, bins=nbins)
    ax1.set_ylabel('Density', fontsize=label_fontsize+2)
    ax3.set_ylabel('Density', fontsize=label_fontsize+2)
    ax3.set_xlabel('ROC-AUC (First Wave)', fontsize=label_fontsize+2)
    ax4.set_xlabel('ROC-AUC (First Year)', fontsize=label_fontsize+2)
    ax1.set_title('a.', loc='left', fontsize=letter_fontsize, y=1.035)
    ax2.set_title('b.', loc='left', fontsize=letter_fontsize, y=1.035)
    ax3.set_title('c.', loc='left', fontsize=letter_fontsize, y=1.035)
    ax4.set_title('d.', loc='left', fontsize=letter_fontsize, y=1.035)

    legend_elements1 = [Patch(facecolor=colors[0], edgecolor='k',
                              label=r'Bins', alpha=0.7),
                        Line2D([0], [0], color=colors[1], lw=1, linestyle='-',
                               label=r'KDE', alpha=0.7), ]
    legend_elements2 = [Patch(facecolor=colors[1], edgecolor='k',
                              label=r'Bins', alpha=0.7),
                        Line2D([0], [0], color=colors[0], lw=1, linestyle='-',
                               label=r'KDE', alpha=0.7), ]
    ax1.legend(handles=legend_elements1, loc='center left', frameon=True,
                   fontsize=label_fontsize-4, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1),
               title='Unstratified', title_fontsize=label_fontsize-5)
    ax2.legend(handles=legend_elements2, loc='center left', frameon=True,
                   fontsize=label_fontsize-4, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1),
               title='Unstratified', title_fontsize=label_fontsize-5)
    ax3.legend(handles=legend_elements1, loc='center left', frameon=True,
                   fontsize=label_fontsize-4, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1),
               title='Stratified', title_fontsize=label_fontsize - 5)
    ax4.legend(handles=legend_elements2, loc='center left', frameon=True,
                   fontsize=label_fontsize-4, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1),
               title='Stratified', title_fontsize=label_fontsize - 5)

    def annotator(input_list, ax):
        mean = np.nanmean(input_list)
        var = np.nanstd(input_list)
        ax.annotate(r'E(ROC-AUC) = ' + str(round(mean, 4)) + r', $\sigma$(ROC-AUC) = ' + str(round(var, 4)),
                    xy=(0.5, 0.85), xytext=(0.5, 0.90), xycoords='axes fraction',
                    fontsize=17, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                    arrowprops=dict(arrowstyle='-[, widthB=10.0, lengthB=1', lw=1.0))

    annotator(list1, ax1)
    annotator(list2, ax2)
    annotator(list3, ax3)
    annotator(list4, ax4)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1]/4)

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, 'covid_seeds.pdf'), bbox_inches='tight')


def mca_plotter(figure_path):
    results_path = os.path.join(os.getcwd(), '..', 'data', 'mcs', 'results', 'csvs')
    df_list = []
    counter = 0
    for file in os.listdir(results_path):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(results_path, file), index_col=False)
            df.rename({'x': str(counter)}, axis=1, inplace=True)
            df_list.append(df[str(counter)])
            counter = counter + 1
        else:
            pass
    df = pd.concat(df_list, axis=1)
    min_series = df.min(axis=1).sort_values().reset_index()[0]
    max_series = df.max(axis=1).sort_values().reset_index()[0]
    med_series = df.median(axis=1).sort_values().reset_index()[0]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    figure_path = os.path.join(os.getcwd(), '..', 'figures')
    colors = ['#001c54', '#E89818']
    nbins = 18
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    csfont = {'fontname': 'Helvetica'}
    ax.plot(min_series.index, min_series, color=colors[1], linestyle='-', alpha=0.8)
    ax.plot(max_series.index, max_series, color=colors[1], linestyle='-', alpha=0.8)
    ax.plot(med_series.index, med_series, color=colors[0], linestyle='--', alpha=0.8)
    ax.set_ylabel(r'Effect Size ($\rm{\hat{\beta}}$)', fontsize=label_fontsize)
    ax.set_xlabel(r'Specification (n)', fontsize=label_fontsize)
    legend_elements1 = [Line2D([0], [0], color=colors[0], lw=1, linestyle='--',
                               label=r'Median', alpha=0.7),
                        Line2D([0], [0], color=colors[1], lw=1, linestyle='-',
                               label=r'Bounds', alpha=0.7), ]
    ax.legend(handles=legend_elements1, loc='upper left', frameon=True,
              fontsize=label_fontsize - 4, framealpha=1, facecolor='w',
              edgecolor=(0, 0, 0, 1),
              )
    plt.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
               color='k', linewidth=1, linestyle='--', alpha=0.5)
    plt.fill_between(min_series.index, min_series, y2=max_series,
                     color=colors[1], alpha=0.075)
    ax.tick_params(axis='both', which='major', labelsize=16)

    inset_ax = inset_axes(ax,
                          width="41%",  # width = 30% of parent_bbox
                          height="90%",  # height : 1 inch
                          loc='lower right',
                          bbox_to_anchor=(-0.005, 0.075, 1, 0.3),
                          bbox_transform=ax.transAxes)
    inset_ax.set_xlabel(r'Effect Size ($\rm{\hat{\beta}}$)',
                        fontsize=label_fontsize - 5, labelpad=-3)
    inset_ax.set_ylabel('Frequency', fontsize=label_fontsize - 5)
    df = pd.concat(df_list, axis=0)
    inset_ax.hist(df, bins=50, color=colors[0],
                  alpha=0.6, edgecolor='k')
    ax.set_title('B.', loc='left', fontsize=letter_fontsize, y=1.0, x=-.05)
    ax.set_axisbelow(True)
    ax.grid(which = "both", linestyle='--', alpha=0.225)
#    inset_ax.set_title('.', loc='left', fontsize=letter_fontsize - 8, y=1.035, x=-0.12)
    sns.despine()
    plt.savefig(os.path.join(figure_path, 'mcs_seeds.pdf'), bbox_inches='tight')


def plot_mcs_single(merged_csv_path, figsize=(8, 8), colors=None, title='a.', ax=None):
    """
    Plot min/median/max of merged MCS specification results as a single square panel.
    """
    colors = colors or ['#001c54', '#E89818']
    df = pd.read_csv(merged_csv_path, index_col=False)
    df = df.apply(pd.to_numeric, errors='coerce')

    min_series = df.min(axis=1).sort_values().reset_index(drop=True)
    max_series = df.max(axis=1).sort_values().reset_index(drop=True)
    med_series = df.median(axis=1).sort_values().reset_index(drop=True)

    context = {'font.family': 'Helvetica', 'text.usetex': False}
    with mpl.rc_context(context):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.plot(min_series.index, min_series, color=colors[1], linestyle='-', alpha=0.8)
        ax.plot(max_series.index, max_series, color=colors[1], linestyle='-', alpha=0.8)
        ax.plot(med_series.index, med_series, color=colors[0], linestyle='--', alpha=0.8)
        ax.fill_between(min_series.index, min_series, max_series, color=colors[1], alpha=0.075)
        ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
                  color='k', linewidth=1, linestyle='--', alpha=0.5)

        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=1, linestyle='--', label='Median', alpha=0.7),
            Line2D([0], [0], color=colors[1], lw=1, linestyle='-', label='Bounds', alpha=0.7),
        ]
        ax.legend(handles=legend_elements, loc='upper left', frameon=True,
                  fontsize=14, framealpha=1, facecolor='w',
                  edgecolor=(0, 0, 0, 1))

        ax.set_ylabel('Effect Size (beta_hat)', fontsize=18)
        ax.set_xlabel('Specification (n)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(which='both', linestyle='--', alpha=0.225)
        ax.set_axisbelow(True)
        ax.set_box_aspect(1)
        ax.set_title(title, loc='left', fontsize=24, fontweight='bold', y=1.0)
        sns.despine()
    return fig, ax


def plot_mcs_pair(merged_csv_path, figsize=(14, 8), colors=None, title_a='a.', title_b='b.'):
    """
    Plot MCS min/median/max (panel a) and distribution of all effects (panel b).
    """
    colors = colors or ['#001c54', '#E89818', '#8b0000']
    fill_color = (254 / 255, 208 / 255, 126 / 255, 10 / 255)
    df = pd.read_csv(merged_csv_path, index_col=False)
    df = df.apply(pd.to_numeric, errors='coerce')

    min_series = df.min(axis=1).sort_values().reset_index(drop=True)
    max_series = df.max(axis=1).sort_values().reset_index(drop=True)
    med_series = df.median(axis=1).sort_values().reset_index(drop=True)
    all_vals = df.stack().dropna().values

    # Summary stats for the distribution used in panel b
    if all_vals.size:
        all_vals = all_vals.astype(float)
        mu = float(np.nanmean(all_vals))
        sigma = float(np.nanstd(all_vals))
        median_val = float(np.nanmedian(all_vals))
        stats = {
            'min': float(np.nanmin(all_vals)),
            'max': float(np.nanmax(all_vals)),
            'mean': mu,
            'std': sigma,
            'median': median_val,
        }
        zscore = ((median_val - mu) / sigma) if sigma else np.nan
        print(
            "MCS effect distribution (panel b): "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"median={stats['median']:.4f}, z-score={zscore:.4f} "
            f"(n={all_vals.size})"
        )

    context = {
        'font.family': 'Helvetica',
        'text.usetex': False,
        'axes.unicode_minus': True
        }
    with mpl.rc_context(context):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        # Panel a
        ax1.plot(min_series.index, min_series, color=colors[1], linestyle='-', alpha=0.8)
        ax1.plot(max_series.index, max_series, color=colors[1], linestyle='-', alpha=0.8)
        ax1.plot(med_series.index, med_series, color=colors[0], linestyle='--', alpha=0.8)
        ax1.fill_between(min_series.index, min_series, max_series, color=fill_color)
        ax1.hlines(y=0, xmin=ax1.get_xlim()[0], xmax=ax1.get_xlim()[1],
                   color='k', linewidth=1, linestyle='--', alpha=0.5)
        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=1, linestyle='--', label='Median', alpha=0.7),
            Line2D([0], [0], color=colors[1], lw=1, linestyle='-', label='Bounds', alpha=0.7),
        ]
        ax1.legend(handles=legend_elements, loc='upper left', frameon=True,
                   fontsize=14, framealpha=1, facecolor='w',
                   edgecolor=(0, 0, 0, 1))
        ax1.set_ylabel(r'Effect Size ($\hat{\beta}$)', fontsize=18)
        ax1.set_xlabel('Specification (n)', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.grid(which='both', linestyle='--', alpha=0.225)
        ax1.set_axisbelow(True)
        ax1.set_box_aspect(1)
        ax1.set_title(title_a, loc='left', fontsize=24, fontweight='bold', y=1.0)

        # Panel b
        hist_color = colors[0]
        kde_color = colors[2]
        ax2.hist(all_vals, bins=50, color=hist_color, alpha=0.6, edgecolor='k', density=True, label='Histogram')
        sns.kdeplot(all_vals, ax=ax2, color=kde_color, lw=1.8, label='KDE')
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_xlabel(r'Effect Size ($\hat{\beta}$)', fontsize=18)
        ax2.set_ylabel('Density', fontsize=18)
        ax2.grid(which='both', linestyle='--', alpha=0.225)
        ax2.set_axisbelow(True)
        ax2.set_box_aspect(1)
        ax2.set_title(title_b, loc='left', fontsize=24, fontweight='bold', y=1.0)
        ax2.legend(loc='center left', frameon=True, fontsize=12, framealpha=1, facecolor='w',
                   edgecolor=(0, 0, 0, 1))

        mu = np.nanmean(all_vals)
        sigma = np.nanstd(all_vals)
        ax2.set_ylim(0, 11)
        y_annot = 10.25
        ax2.annotate(r'$\mu$ = ' + f"{mu:+.3f}" + r', $\sigma$ = ' + f"{sigma:.3f}",
                     xy=(mu, y_annot - 0.3), xytext=(mu, y_annot),
                     xycoords='data',
                     fontsize=14, ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                     arrowprops=dict(arrowstyle='-[, widthB=8, lengthB=1', lw=1.0))

        sns.despine()
        fig.savefig(os.path.join(os.getcwd(), '..', 'figures', 'mcs_pair.pdf'),
                    bbox_inches='tight')
    return fig, (ax1, ax2)


def buffons_plotter(figure_path):
    df = pd.read_csv(os.path.join(os.getcwd(),
                                  '..', 'data',
                                  'needles',
                                  'results',
                                  'throw100_25000_5000seeds.csv'),
                    names = ['Throws', 'Min', '25th_PC',
                    'Median', '75th_PC', 'Max'])
    print('Buffons last row: ', df.iloc[-1])
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
    df = df.set_index('Throws')
    df = df[45:]
    color_fill = '#E89818'
    colors = ['#001c54', '#F7EDD2', '#8b0000']
    ax.plot(df['Min'], color=colors[2])
    ax.plot(df['Max'], color=colors[0])
    ax.set_xlim(0, df.index[-1]+500)
    ax.set_ylim(2.225, 4.5)
    ax.hlines(math.pi, df.index[0]+500, df.index[-1], color='k', linestyle='--', alpha=0.5)
    ax.fill_between(df.index, df['Min'], df['Max'], color=color_fill, alpha=0.075)
    ax.set_xlabel('Number of Throws', fontsize=16)
    ax.set_ylabel(r'Estimate of $\mathrm{\pi}$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title('A.', loc='left', fontsize=16, y=1.025, x=-.025)
    ax.tick_params(width=1, length=8)
    legend_elements1 = [
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'Upper Limit', alpha=0.7),
        Line2D([0], [0], color=colors[2], lw=2, linestyle='-',
               label=r'Lower Limit', alpha=0.7),
        Line2D([0], [0], color='k', lw=2, linestyle='--',
               label=r'$\mathrm{\pi}$', alpha=0.7),
        Patch(facecolor=colors[1], edgecolor=(0,0,0,1),
                              label=r'Range', alpha=1)]
    ax.legend(handles=legend_elements1, loc='upper right', frameon=True,
              fontsize=label_fontsize-4, framealpha=1, facecolor='w',
              edgecolor=(0, 0, 0, 1), ncols=2
             )
    ax.set_axisbelow(True)
    ax.grid(which = "both", linestyle='--', alpha=0.225)
    plt.savefig(os.path.join(figure_path,
                             'buffon_seeds.pdf'),
                bbox_inches='tight')


def make_ffc_just_gpa(ffc, figure_path):
    fig = plt.figure(figsize=(8, 8))
    figurez = []
    outcome = 'gpa'
    model = 'ols'
    df1 = ffc[(ffc['outcome']==outcome) &
              (ffc['account']==model)][0:10000]
    g = sns.jointplot(x=df1['beta'],
                      y=df1['r2_holdout'],
                          kind='hex',
                          marginal_kws=dict(bins=25,
                                            color='w'))
    g.plot_joint(sns.kdeplot, color="r", levels=6)
    g.ax_marg_x.annotate('A.', xy=(-0.1, 1), xycoords='axes fraction', ha='left', va='center', fontsize=24)
    g.ax_joint.set_ylabel(r'Pseudo R$^2$', fontsize=16)
    g.ax_joint.set_xlabel('Lagged Coefficient', fontsize=16)
    g.ax_joint.annotate('GPA', xy=(0.9, 0.05),
                        xycoords='axes fraction',
                        ha='left', va='center', fontsize=14)
    g.ax_joint.tick_params(axis='both', which='major', labelsize=13)
    g.ax_joint.grid(which = "both", linestyle='--', alpha=0.25)
    plt.savefig(os.path.join(figure_path, 'ffc_seeds_just_gpa.pdf'), bbox_inches='tight')
    plt.show()
    print('Beta minimum for GPA: ', df1['beta'].min())
    print('Beta median for GPA: ', df1['beta'].median())
    print('Beta maximum for GPA: ', df1['beta'].max())
    print('R2 minimum for GPA: ', df1['r2_holdout'].min())
    print('R2 median for GPA:', df1['r2_holdout'].median())
    print('R2 median for GPA:', df1['r2_holdout'].max())


def plot_rgms(figure_path):
    df = pd.read_csv(os.path.join(os.getcwd(), '..',
                                  'data', 'rgms',
                                  'rgms.csv'),
                     header=None)
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(8, 8))
    nbins = 15
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    csfont = {'fontname': 'Helvetica'}
    colors = ['#001c54', '#E89818']
    sns.swarmplot(y=df[0], ax=ax1,  color=colors[0])
    sns.swarmplot(y=df[1], ax=ax3, color=colors[1], alpha=0.825)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    sns.despine(ax=ax1)
    sns.despine(ax=ax3)
    ax1.set_xlabel('Pr=0.2', fontsize=label_fontsize)
    ax1.set_ylabel('Average Degree', fontsize=label_fontsize)
    ax3.set_ylabel('', fontsize=label_fontsize)
    ax3.set_xlabel('Pr=0.4', fontsize=label_fontsize)
    ax1.set_title('A.', loc='left', fontsize=letter_fontsize, y=1.035)
    ax3.tick_params(axis='y', colors='k')
    ax1.grid(which = "both", linestyle='--', alpha=0.225)
    ax3.grid(which = "both", linestyle='--', alpha=0.225)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.setp(ax1.collections, alpha=.85)
    plt.setp(ax3.collections, alpha=.85)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, 'rgm_seeds.pdf'),
                bbox_inches='tight')

def plot_topic_jointplot():
    import os
    import math
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    mpl.rcParams['font.family'] = 'Helvetica'

    # nature = pd.read_csv(os.path.join(os.getcwd(),
    #                     '..',
    #                     'data',
    ##                     'bibliometric',
    #                     'meta_data',
    #                     'metadata_nature.csv')
    #                    )

    science = pd.read_csv(os.path.join(os.getcwd(),
                                       '..',
                                       'data',
                                       'bibliometric',
                                       'meta_data',
                                       'metadata_science.csv')
                          )

    # pnas = pd.read_csv(os.path.join(os.getcwd(),
    #                     '..',
    #                     'data',
    #                     'bibliometric',
    #                     'meta_data',
    #                     'metadata_pnas.csv')
    #                    )

    nejm = pd.read_csv(os.path.join(os.getcwd(),
                                    '..',
                                    'data',
                                    'bibliometric',
                                    'meta_data',
                                    'metadata_nejm.csv')
                       )

    class SeabornFig2Grid():
        def __init__(self, seaborngrid, fig, subplot_spec):
            self.fig = fig
            self.sg = seaborngrid
            self.subplot = subplot_spec
            if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
                    isinstance(self.sg, sns.axisgrid.PairGrid):
                self._movegrid()
            elif isinstance(self.sg, sns.axisgrid.JointGrid):
                self._movejointgrid()
            self._finalize()

        def _movegrid(self):
            """ Move PairGrid or Facetgrid """
            self._resize()
            n = self.sg.axes.shape[0]
            m = self.sg.axes.shape[1]
            self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
            for i in range(n):
                for j in range(m):
                    self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

        def _movejointgrid(self):
            """ Move Jointgrid """
            h = self.sg.ax_joint.get_position().height
            h2 = self.sg.ax_marg_x.get_position().height
            r = int(np.round(h / h2))
            self._resize()
            self.subgrid = gridspec.GridSpecFromSubplotSpec(r + 1, r + 1, subplot_spec=self.subplot)

            self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
            self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
            self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

        def _moveaxes(self, ax, gs):
            # https://stackoverflow.com/a/46906599/4124317
            ax.remove()
            ax.figure = self.fig
            self.fig.axes.append(ax)
            self.fig.add_axes(ax)
            ax._subplotspec = gs
            ax.set_position(gs.get_position(self.fig))
            ax.set_subplotspec(gs)

        def _finalize(self):
            plt.close(self.sg.fig)
            self.fig.canvas.mpl_connect("resize_event", self._resize)
            self.fig.canvas.draw()

        def _resize(self, evt=None):
            self.sg.fig.set_size_inches(self.fig.get_size_inches())

    def jointplotter(df, counter):
        title_list = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.']
        title = title_list[counter]
        g = sns.jointplot(x=df['topics_count'],
                          y=df['outliers_count'],
                          kind='hex',
                          marginal_kws=dict(bins=25,
                                            color='w'))
        g.plot_joint(sns.kdeplot, color="r", levels=6)
        g.ax_marg_x.annotate(title, xy=(-0.1, .45), xycoords='axes fraction',
                             ha='left', va='center', fontsize=24)
        return g

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)
    figurez = []

    # figurez.append(jointplotter(nature, 0))
    figurez.append(jointplotter(science, 0))
    # figurez.append(jointplotter(pnas, 2))
    figurez.append(jointplotter(nejm, 1))
    # tmp = SeabornFig2Grid(figurez[0], fig, gs[0])
    tmp = SeabornFig2Grid(figurez[0], fig, gs[0])
    # tmp = SeabornFig2Grid(figurez[2], fig, gs[2])
    tmp = SeabornFig2Grid(figurez[1], fig, gs[1])

    figurez[0] = figurez[0].ax_joint.annotate('Science', xy=(0.9, 0.05),
                                              xycoords='axes fraction',
                                              ha='left', va='center', fontsize=14)
    figurez[1] = figurez[1].ax_joint.annotate('NEJM', xy=(0.878, 0.05),
                                              xycoords='axes fraction',
                                              ha='left', va='center', fontsize=14)
    # figurez[2] = figurez[2].ax_joint.annotate('Material Hardship', xy=(0.56, 0.05),
    #                                          xycoords='axes fraction',
    #                                          ha='left', va='center', fontsize=14)
    # figurez[3] = figurez[3].ax_joint.annotate('Eviction', xy=(0.805, 0.05),
    #                                          xycoords='axes fraction',
    #                                          ha='left', va='center', fontsize=14)
    gs.tight_layout(fig)
    gs.update(hspace=0.1)
    figure_path = os.path.join(os.getcwd(), '..', 'figures')
    plt.savefig(os.path.join(figure_path, 'topic_modelling_seeds_jointplot_2.pdf'), bbox_inches='tight')
    plt.show()


def plot_topics_barplot(figure_path, figsize, colors = ['#001c54', '#E89818']):
    #
    metapath = os.path.join(os.getcwd(),
                            '..',
                            'data',
                            'bibliometric',
                            'meta_data'
                            )
    science = pd.read_csv(os.path.join(metapath,
                                       'metadata_science.csv')
                          )
    pnas = pd.read_csv(os.path.join(metapath,
                                    'metadata_pnas.csv')
                       )
    nejm = pd.read_csv(os.path.join(metapath,
                                    'metadata_nejm.csv')
                       )
    nature = pd.read_csv(os.path.join(metapath,
                                      'metadata_nature.csv'
                                      )
                         )

    shape = pd.read_csv(os.path.join(metapath,
                                     'metadata_shape.csv'
                                     )
                        )

    popstudies = pd.read_csv(os.path.join(metapath,
                                          'metadata_popstudies.csv'
                                          )
                             )
    fig, ((ax1, ax2, ax3),
          (ax4, ax5, ax6)
          ) = plt.subplots(2, 3, figsize=figsize)
    nbins = 25
    sns.histplot(science[science['random_state'] != 77]['topics_count'],
                 ax=ax1,
                 color=colors[0],
                 bins=nbins)
    ax1_twin = ax1.twinx()
    sns.kdeplot(science[science['random_state'] != 77]['topics_count'], ax=ax1_twin, color=colors[1])

    sns.histplot(nejm[nejm['random_state'] != 77]['topics_count'],
                 ax=ax2,
                 color=colors[0],
                 bins=nbins)
    ax2_twin = ax2.twinx()
    sns.kdeplot(nejm[nejm['random_state'] != 77]['topics_count'], ax=ax2_twin, color=colors[1])

    sns.histplot(pnas[pnas['random_state'] != 77]['topics_count'],
                 ax=ax3,
                 color=colors[0],
                 bins=nbins)
    ax3_twin = ax3.twinx()
    sns.kdeplot(pnas[pnas['random_state'] != 77]['topics_count'], ax=ax3_twin, color=colors[1])

    sns.histplot(nature[nature['random_state'] != 77]['topics_count'],
                 ax=ax4,
                 color=colors[0],
                 bins=nbins)
    ax4_twin = ax4.twinx()
    sns.kdeplot(nature[nature['random_state'] != 77]['topics_count'], ax=ax4_twin, color=colors[1])

    sns.histplot(shape[shape['random_state'] != 77]['topics_count'],
                 ax=ax5,
                 color=colors[0],
                 bins=nbins)
    ax5_twin = ax5.twinx()
    sns.kdeplot(shape[shape['random_state'] != 77]['topics_count'],
                ax=ax5_twin, color=colors[1])

    sns.histplot(popstudies[popstudies['random_state'] != 77]['topics_count'],
                 ax=ax6,
                 color=colors[0],
                 bins=nbins)
    ax6_twin = ax6.twinx()
    sns.kdeplot(popstudies[popstudies['random_state'] != 77]['topics_count'],
                ax=ax6_twin, color=colors[1])

    ax1.set_title('a.', loc='left', fontsize=23)
    ax2.set_title('b.', loc='left', fontsize=23)
    ax3.set_title('c.', loc='left', fontsize=22)
    ax4.set_title('d.', loc='left', fontsize=22)
    ax5.set_title('e.', loc='left', fontsize=22)
    ax6.set_title('f.', loc='left', fontsize=22)
    # ax1.set_xlim(0, 400)
    # ax2.set_xlim(0, 160)
    # ax3.set_xlim(0, ax3.get_xlim()[1])
    # ax4.set_xlim(0, ax4.get_xlim()[1])
    ax1.grid(which="both", linestyle='--', alpha=0.225)
    ax2.grid(which="both", linestyle='--', alpha=0.225)
    ax3.grid(which="both", linestyle='--', alpha=0.225)
    ax4.grid(which="both", linestyle='--', alpha=0.225)
    ax5.grid(which="both", linestyle='--', alpha=0.225)
    ax6.grid(which="both", linestyle='--', alpha=0.225)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax4.set_axisbelow(True)
    ax5.set_axisbelow(True)
    ax6.set_axisbelow(True)

    legend_elements1 = [
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1),
              label=r'Histogram'),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='-',
               label=r'Kernel Density', alpha=1)
    ]
    #    ax1.legend(handles=legend_elements1, loc='upper right', frameon=True,
    #               fontsize=10, framealpha=1, facecolor='w',
    #               edgecolor=(0, 0, 0, 1), ncols=1, title='Science'
    #               )
    #    ax2.legend(handles=legend_elements1, loc='upper right', frameon=True,
    #               fontsize=10, framealpha=1, facecolor='w',
    #               edgecolor=(0, 0, 0, 1), ncols=1, title='New England Journal Of Medicine'
    #               )
    ax3.legend(handles=legend_elements1, loc='center right', frameon=True,
               fontsize=12, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1,  # title='PNAS'
               )
    #    ax4.legend(handles=legend_elements1, loc='upper right', frameon=True,
    #               fontsize=10, framealpha=1, facecolor='w',
    #               edgecolor=(0, 0, 0, 1), ncols=1, title='Nature'
    #               )
    #
    #    ax5.legend(handles=legend_elements1, loc='upper right', frameon=True,
    #               fontsize=10, framealpha=1, facecolor='w',
    #               edgecolor=(0, 0, 0, 1), ncols=1, title='SHAPE'
    #               )

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim(0, None)

    def _print_stats(label, series):
        arr = np.asarray(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            print(f"{label}: empty")
            return
        mean = float(np.mean(arr))
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        z = mean / sd if sd not in (0, float("nan")) else float("nan")
        print(f"{label}: mean={mean:.4f}, min={amin:.4f}, max={amax:.4f}, sd={sd:.4f}, z={z:.4f}")

    _print_stats("Science topics", science['topics_count'])
    _print_stats("NEJM topics", nejm['topics_count'])
    _print_stats("PNAS topics", pnas['topics_count'])
    _print_stats("Nature topics", nature['topics_count'])
    _print_stats("SHAPE topics", shape['topics_count'])
    _print_stats("Population Studies topics", popstudies['topics_count'])

    ax1.set_ylabel('Count: Science', fontsize=14)
    ax2.set_ylabel('Count: NEJM', fontsize=14)
    ax3.set_ylabel('Count: PNAS', fontsize=14)
    ax4.set_ylabel('Count: Nature', fontsize=14)
    ax5.set_ylabel('Count: SHAPE', fontsize=14)
    ax6.set_ylabel('Count: Population Studies', fontsize=14)
    for ax_twin in [ax1_twin, ax2_twin, ax3_twin, ax4_twin, ax5_twin, ax6_twin]:
        ax_twin.set_yticks([])  # Removes right y-axis tick labels
        ax_twin.tick_params(right=False)

        # for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    # ax.set_xlim(0, ax.get_xlim()[1])
    ax1.set_xlabel('', fontsize=16)
    ax2.set_xlabel('', fontsize=16)
    ax3.set_xlabel('', fontsize=16)
    ax4.set_xlabel('Number of topics', fontsize=16)
    ax5.set_xlabel('Number of topics', fontsize=16)
    ax6.set_xlabel('Number of topics', fontsize=16)
    ax1_twin.set_ylabel('', fontsize=16)
    ax2_twin.set_ylabel('', fontsize=16)
    ax3_twin.set_ylabel('', fontsize=16)
    ax4_twin.set_ylabel('', fontsize=16)
    ax5_twin.set_ylabel('', fontsize=16)
    ax6_twin.set_ylabel('', fontsize=16)

    # @TODO: this can be modularised when less lazy

    n_topics77 = science[science['random_state'] == 77]['topics_count'][999]
    ymin, ymax = ax1.get_ylim()
    ax1.axvline(x=n_topics77,
                ymin=0,
                ymax=1,
                color='red',
                linestyle='--',
                linewidth=2)
    annotation_y = ymin + (ymax - ymin) * 0.8  # 70% up the y-axis
    ax1.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                 xy=(n_topics77,
                     annotation_y),
                 xytext=(n_topics77 + 450,
                         annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1))

    n_topics77 = nejm[nejm['random_state'] == 77]['topics_count'][1000]
    ymin, ymax = ax2.get_ylim()
    ax2.axvline(x=n_topics77,
                ymin=0,
                ymax=1,
                color='red',
                linestyle='--',
                linewidth=2)
    annotation_y = ymin + (ymax - ymin) * 0.8  # 70% up the y-axis
    ax2.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                 xy=(n_topics77,
                     annotation_y),
                 xytext=(n_topics77 - 100,
                         annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1)
                 )
    n_topics77 = pnas[pnas['random_state'] == 77]['topics_count'][1000]
    ymin, ymax = ax3.get_ylim()
    ax3.axvline(x=n_topics77,
                ymin=0,
                ymax=1,
                color='red',
                linestyle='--',
                linewidth=2)
    annotation_y = ymin + (ymax - ymin) * 0.8  # 70% up the y-axis
    ax3.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                 xy=(n_topics77,
                     annotation_y),
                 xytext=(n_topics77 + 500,
                         annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1))

    n_topics77 = nature[nature['random_state'] == 77]['topics_count'][1000]
    ymin, ymax = ax4.get_ylim()
    ax4.axvline(x=n_topics77,
                ymin=0,
                ymax=1,
                color='red',
                linestyle='--',
                linewidth=2)
    annotation_y = ymin + (ymax - ymin) * 0.8  # 70% up the y-axis
    ax4.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                 xy=(n_topics77,
                     annotation_y),
                 xytext=(n_topics77 + 1000,
                         annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1))
    n_topics77 = shape[shape['random_state'] == 77]['topics_count'][1000]
    ymin, ymax = ax5.get_ylim()
    ax5.axvline(x=n_topics77,
                ymin=0,
                ymax=1,
                color='red',
                linestyle='--',
                linewidth=2)
    annotation_y = ymin + (ymax - ymin) * 0.8  # 70% up the y-axis
    ax5.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                 xy=(n_topics77,
                     annotation_y),
                 xytext=(n_topics77 - 45,
                         annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1))

    n_topics77 = popstudies[popstudies['random_state'] == 77]['topics_count'][1000]
    ymin, ymax = ax6.get_ylim()
    ax6.axvline(x=n_topics77,
                ymin=0,
                ymax=1,
                color='red',
                linestyle='--',
                linewidth=2)
    annotation_y = ymin + (ymax - ymin) * 0.8  # 70% up the y-axis
    ax6.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                 xy=(n_topics77,
                     annotation_y),
                 xytext=(n_topics77 - 15,
                         annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1))
    sns.despine()
    plt.savefig(os.path.join(figure_path,
                             'topic_modelling_seeds_histplot.pdf'),
                bbox_inches='tight')
    plt.tight_layout()


def load_scientometrics():
    df_rng = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                      'openalex_rn_papers.csv'))
    df_qrng = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                       'openalex_rn_and_quantum_papers.csv'))
    df_hrng = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                       'openalex_rn_and_hardware_papers.csv'))
    df_prng = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                       'openalex_rn_and_pseudo_papers.csv'))
    df_quarng = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                         'openalex_rn_and_quasi_papers.csv'))
    df_yr = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                     'openalex_year_counts.csv'))
    df_yr_dom = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                         'openalex_domain_year_counts.csv'))
    df_dom = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                      'openalex_domain_counts.csv'))
    df_dom['domain'] = df_dom['domain'].astype(str)
    return df_rng, df_hrng, df_qrng, df_prng, df_quarng, df_yr, df_yr_dom, df_dom


def desc_print_scientometrics(df_rng, df_hrng, df_qrng, df_prng, df_quarng):
    def desc_print(df, term):
        print(f'We have {len(df)} papers for {term}).')

        j_count = df['journal'].value_counts().reset_index()
        j_count_j = j_count['journal'][0]
        j_count_val = j_count['count'][0]
        print(f'Modal journal: {j_count_j} ({j_count_val} papers)')

        sub_count = df['subfield'].value_counts().reset_index()
        sub_count_sub = sub_count['subfield'][0]
        sub_count_val = sub_count['count'][0]
        print(f'Modal subfield: {sub_count_sub} ({sub_count_val} papers)')

        field_count = df['field'].value_counts().reset_index()
        field_count_field = field_count['field'][0]
        field_count_val = field_count['count'][0]
        print(f'Modal field: {field_count_field} ({field_count_val} papers)')

        domain_count = df['domain'].value_counts().reset_index()
        domain_count_domain = domain_count['domain'][0]
        domain_count_val = domain_count['count'][0]
        print(f'Modal domain: {domain_count_domain} ({domain_count_val} papers)')

    desc_print(df_rng, '"random number"')
    print('')
    desc_print(df_hrng, '"random number" and "hardware"')
    print('')
    desc_print(df_qrng, '"random number" and "quantum"')
    print('')
    desc_print(df_prng, '"random number" and "pseudo"')
    print('')
    desc_print(df_quarng, '"random number" and "quasi"')
    print('')


def make_table(df_rng, df_hrng, df_qrng, df_prng, df_quarng, column):
    df_rng_val = df_rng[column].value_counts()
    df_hrng_val = df_hrng[column].value_counts()
    df_qrng_val = df_qrng[column].value_counts()
    df_prng_val = df_prng[column].value_counts()
    df_quarng_val = df_quarng[column].value_counts()
    df_merged = pd.merge(df_rng_val, df_hrng_val, left_index=True, right_index=True, how='left')
    df_merged = df_merged.rename({'count_x': '"Random Numbers"', 'count_y': '"Random Numbers" and "Hardware"'}, axis=1)
    df_merged = pd.merge(df_merged, df_qrng_val, left_index=True, right_index=True, how='left')
    df_merged = df_merged.rename({'count': '"Random Numbers" and "Quantum"'}, axis=1)
    df_merged = pd.merge(df_merged, df_prng_val, left_index=True, right_index=True, how='left')
    df_merged = df_merged.rename({'count': '"Random Numbers" and "Pseudo"'}, axis=1)
    df_merged = pd.merge(df_merged, df_quarng_val, left_index=True, right_index=True, how='left')
    df_merged = df_merged.rename({'count': '"Random Numbers" and "Quasi"'}, axis=1)

    for col in df_merged.columns:
        if df_merged[col].isnull().sum() == 0:
            df_merged[col] = df_merged[col].astype(int)
    return df_merged


def make_scientometric_ts(df_rng, df_hrng, df_qrng, df_prng, df_quarng, df_yr, domain_df):
    df_yr = df_yr.rename({'count': 'total_count'}, axis=1)
    df_yr_rng = pd.DataFrame(df_rng['publication_year'].value_counts())
    df_yr_rng = df_yr_rng.reset_index()
    df_yr_rng = df_yr_rng.rename({'publication_year': 'year', 'count': 'RNG_count'}, axis=1)

    df_yr_qrng = pd.DataFrame(df_qrng['publication_year'].value_counts())
    df_yr_qrng = df_yr_qrng.reset_index()
    df_yr_qrng = df_yr_qrng.rename({'publication_year': 'year', 'count': 'QRNG_count'}, axis=1)

    df_yr_hrng = pd.DataFrame(df_hrng['publication_year'].value_counts())
    df_yr_hrng = df_yr_hrng.reset_index()
    df_yr_hrng = df_yr_hrng.rename({'publication_year': 'year', 'count': 'HRNG_count'}, axis=1)

    df_yr_prng = pd.DataFrame(df_prng['publication_year'].value_counts())
    df_yr_prng = df_yr_prng.reset_index()
    df_yr_prng = df_yr_prng.rename({'publication_year': 'year', 'count': 'PRNG_count'}, axis=1)

    df_yr_quarng = pd.DataFrame(df_quarng['publication_year'].value_counts())
    df_yr_quarng = df_yr_quarng.reset_index()
    df_yr_quarng = df_yr_quarng.rename({'publication_year': 'year', 'count': 'QUASI_count'}, axis=1)

    df_yr = pd.merge(df_yr, df_yr_rng, left_on='year', right_on='year', how='left')
    df_yr = pd.merge(df_yr, df_yr_hrng, left_on='year', right_on='year', how='left')
    df_yr = pd.merge(df_yr, df_yr_qrng, left_on='year', right_on='year', how='left')
    df_yr = pd.merge(df_yr, df_yr_prng, left_on='year', right_on='year', how='left')
    df_yr = pd.merge(df_yr, df_yr_quarng, left_on='year', right_on='year', how='left')
    for rng_type in ['RNG_count', 'HRNG_count', 'QRNG_count', 'PRNG_count', 'QUASI_count']:
        df_yr[rng_type] = df_yr[rng_type] / df_yr['total_count'] * 100
    return df_yr


def _cache_path_yahoo(ticker: str, start: str, end: str) -> Path:
    root = Path.cwd()
    cache_dir = (root / ".." / "data" / "cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"yahoo_{ticker.replace('=', '')}_{start}_{end}.csv"
    return cache_dir / key


def _should_retry(exc: Exception) -> bool:
    msg = str(exc).lower()
    transient_markers = [
        "rate limit",
        "too many requests",
        "429",
        "timed out",
        "timeout",
        "temporarily",
        "unavailable",
        "connection aborted",
        "connection reset",
        "remote disconnected",
        "max retries exceeded",
    ]
    return any(m in msg for m in transient_markers)


def download_and_resample_yahoo(
    ticker: str,
    start: str,
    end: str,
    cache_ttl_days: int = 7,
    max_attempts: int = 7,
    backoff_base: float = 1.5,
    min_sleep: float = 1.0,
    max_sleep: float = 60.0,
) -> pd.DataFrame:
    cpath = _cache_path_yahoo(ticker, start, end)
    if cpath.exists():
        mtime = datetime.fromtimestamp(cpath.stat().st_mtime)
        age_days = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(mtime, tz="UTC")).days
        if age_days < cache_ttl_days:
            df = pd.read_csv(cpath, index_col=0, parse_dates=True)
            df.index.name = None
            if "Close" in df.columns and not df.empty:
                return df

    attempt = 0
    last_exc = None
    while attempt < max_attempts:
        try:
            raw = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=True,
            )
            if isinstance(raw, pd.Series):
                raw = raw.to_frame().T
            if raw is None or raw.empty:
                raise RuntimeError("Empty dataframe returned from Yahoo Finance.")
            if isinstance(raw.columns, pd.MultiIndex):
                if ("Close", ticker) in raw.columns:
                    close = raw[("Close", ticker)].rename("Close").to_frame()
                elif "Close" in raw.columns.get_level_values(0):
                    close = raw["Close"].rename("Close").to_frame()
                else:
                    raise RuntimeError(f"'Close' column not found in Yahoo data (columns={raw.columns})")
            else:
                if "Close" not in raw.columns:
                    raise RuntimeError(f"'Close' column not found in Yahoo data (columns={list(raw.columns)})")
                close = raw[["Close"]].copy()
            close = close.resample("D").ffill().dropna()
            close.to_csv(cpath)
            return close
        except Exception as e:
            last_exc = e
            attempt += 1
            if attempt >= max_attempts or not _should_retry(e):
                break
            sleep_s = min(max_sleep, max(min_sleep, backoff_base**attempt))
            sleep_s *= 0.75 + 0.5 * random.random()
            print(
                f"[yfinance] Transient error (attempt {attempt}/{max_attempts}): {e}. "
                f"Sleeping {sleep_s:.1f}s before retry."
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"Yahoo Finance download failed after {max_attempts} attempts: {last_exc}")


def plot_five_models(
    colors=None,
    figsize=(16, 13),
    fill_color=(254 / 255, 208 / 255, 126 / 255, 10 / 255),
):
    """
    Multi-panel figure:
      a) FX history + random walks
      b) Science topics histogram/KDE
      c) MNIST accuracy histogram/KDE
      d) Schelling segregation stats
      e) mvprobit seed variance (spans both columns)
    """
    colors = colors or [(0 / 255, 28 / 255, 84 / 255, 0.8), "#E89818", "#8b0000", "#8b0000"]
    palette = list(colors)
    if len(palette) < 4:
        palette += [palette[-1]] * (4 - len(palette))

    def _print_stats(label, arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            print(f"{label}: empty")
            return
        mean = float(np.mean(arr))
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        z = mean / sd if sd not in (0, float("nan")) else float("nan")
        print(f"{label}: mean={mean:.4f}, min={amin:.4f}, max={amax:.4f}, sd={sd:.4f}, z={z:.4f}")

    usuk_data = download_and_resample_yahoo(ticker="USDGBP=X", start="2022-10-01", end="2024-06-30")
    rw_usuk_path = os.path.join(os.getcwd(), "..", "data", "random_walk", "random_walks_usuk.zip")
    random_walks_usuk = pd.read_csv(rw_usuk_path, header=None, compression="zip")
    end_date = usuk_data.index[-1]
    start_date = end_date + pd.DateOffset(1)
    random_walks_usuk.index = pd.date_range(start=start_date, periods=len(random_walks_usuk), freq="D")
    # Random-walk stats at final date
    last_rw = random_walks_usuk.iloc[-1]
    _print_stats(
        "[plot_five_models] US/GBP RW final-day distribution",
        last_rw.values,
    )

    metapath = os.path.join(os.getcwd(), "..", "data", "bibliometric", "meta_data")
    science = pd.read_csv(os.path.join(metapath, "metadata_science.csv"))
    mnist = pd.read_csv(os.path.join(os.getcwd(), "..", "data", "MNIST", "results", "mnist_results.csv"))

    schelling_path = os.path.join(os.getcwd(), "..", "data", "schelling", "schelling_df_25_0.3_0.3.csv")
    schelling_df = pd.read_csv(schelling_path, index_col=0)
    schelling = schelling_df[schelling_df["Step"] != "Convergence"].copy()
    schelling_conv = schelling_df[schelling_df["Step"] == "Convergence"]
    if not schelling_conv.empty:
        conv_steps = schelling_conv["Happy Count"].astype(float)
        _print_stats("[plot_five_models] schelling (empty=0.3, threshold=0.3) steps", conv_steps)
    schelling["Step"] = schelling["Step"].astype(int)
    schelling["Happy Count"] = schelling["Happy Count"].astype(float)
    schelling["Happy Count Adjusted"] = schelling.groupby("Step")["Happy Count"].transform(lambda x: x - x.mean())
    filtered_df = schelling[schelling["Step"] <= 29]

    mv_df = pd.read_csv(
        os.path.join(os.getcwd(), "..", "data", "mvprobit", "results_school_total_draws150_total_seeds1000.csv")
    )
    mv_summary = pd.DataFrame(index=mv_df["draws"].unique())
    for draw in mv_df["draws"].unique():
        mv_summary.at[draw, "Min"] = mv_df[mv_df["draws"] == draw]["rho21"].min()
        mv_summary.at[draw, "Max"] = mv_df[mv_df["draws"] == draw]["rho21"].max()
        mv_summary.at[draw, "Median"] = mv_df[mv_df["draws"] == draw]["rho21"].median()
    mv_summary = mv_summary.sort_index()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.6])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    usuk_data["Close"].plot(ax=ax1, color=palette[0], label="Insample")
    random_walks_usuk.min(axis=1).plot(ax=ax1, color=palette[1], linestyle="--", label="Min")
    random_walks_usuk.median(axis=1).plot(ax=ax1, color="k", linestyle="--", label="Median")
    random_walks_usuk.max(axis=1).plot(ax=ax1, color=palette[2], linestyle="--", label="Max")
    random_walks_usuk.quantile(0.05, axis=1).plot(ax=ax1, color="k", linestyle="--", linewidth=0.75)
    random_walks_usuk.quantile(0.95, axis=1).plot(ax=ax1, color="k", linestyle="--", linewidth=0.75)
    ax1.fill_between(
        random_walks_usuk.index,
        random_walks_usuk.min(axis=1),
        random_walks_usuk.max(axis=1),
        color=fill_color,
        label="Range",
    )
    legend_elements = [
        Line2D([0], [0], color=palette[0], linestyle="-", label="Insample", lw=2),
        Line2D([0], [0], color=palette[1], linestyle="--", label="Min", lw=2),
        Line2D([0], [0], color=palette[2], linestyle="--", label="Max", lw=2),
        Line2D([0], [0], color="k", linestyle="--", linewidth=0.75, label="5th/95th percentile"),
        Line2D([0], [0], color="k", linestyle="--", label="Median", lw=2),
        Patch(facecolor=fill_color, edgecolor="k", label="Range"),
    ]
    ax1.set_ylabel("US ($) / UK (£) Exchange Rate", fontsize=14)
    ax1.set_xlabel("")
    ax1.set_title("a.", loc="left", fontsize=22, y=1.02, fontweight="bold")
    ax1.grid(which="major", linestyle="--", alpha=0.225)
    ax1.legend(
        handles=legend_elements,
        loc="upper left",
        ncol=3,
        frameon=True,
        fontsize=10,
        framealpha=1,
        facecolor="w",
        edgecolor="k",
    )

    nbins = 25
    sns.histplot(
        science[science["random_state"] != 77]["topics_count"],
        ax=ax2,
        color=palette[0],
        bins=nbins,
        edgecolor="k",
        stat="density",
        label="Histogram",
    )
    sns.kdeplot(science[science["random_state"] != 77]["topics_count"], ax=ax2, color=palette[1], lw=2, label="KDE")
    ax2.set_xlim(0, ax2.get_xlim()[1])
    ax2.set_title("b.", loc="left", fontsize=22, y=1.02, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=14)
    ax2.set_xlabel("Number of topics", fontsize=14)
    ax2.grid(which="both", linestyle="--", alpha=0.225)
    ax2.legend(loc="upper right", ncol=1, frameon=True, fontsize=10, framealpha=1, facecolor="w", edgecolor="k")
    # Seed annotations for topics (e.g., seed 77)
    if "random_state" in science.columns and "topics_count" in science.columns:
        ylim = ax2.get_ylim()
        y_ann = ylim[1] * 0.8
        for seed in [77]:
            mask = science["random_state"] == seed
            if mask.any():
                try:
                    val = science.loc[mask, "topics_count"].iloc[0]
                    ax2.axvline(x=val, ymin=0, ymax=1, color="red", linestyle="--", linewidth=1.5)
                    ax2.annotate(f"Seed {seed}: topics={val}",
                                 xy=(val, y_ann),
                                 xytext=(val + (ax2.get_xlim()[1]-ax2.get_xlim()[0])*0.08, y_ann),
                                 ha="left", va="center", fontsize=12,
                                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                                 arrowprops=dict(arrowstyle="->", color="black", lw=1))
                except Exception:
                    pass

    sns.histplot(mnist["correct"], ax=ax3, color=palette[0], bins=24, edgecolor="k", stat="density", label="Histogram")
    sns.kdeplot(mnist["correct"], ax=ax3, color=palette[1], lw=2, label="KDE")
    ax3.set_title("c.", loc="left", fontsize=22, y=1.02, fontweight="bold")
    ax3.set_ylabel("Density", fontsize=14)
    ax3.set_xlabel("MNIST accuracy", fontsize=14)
    ax3.grid(which="both", linestyle="--", alpha=0.225)
    ax3.legend(loc="upper left", ncol=1, frameon=True, fontsize=10, framealpha=1, facecolor="w", edgecolor="k")
    # Seed annotations for MNIST (seeds 42 and 123) using the same fixed layout as plot_predictions().
    seed_targets = [42, 123]
    mnist_seed_values = {}
    manual_path = os.path.join(os.getcwd(), "..", "data", "MNIST", "results", "mnist_results_manual_seeds.csv")
    if os.path.exists(manual_path):
        manual = pd.read_csv(manual_path)
        if "correct" in manual.columns and len(manual) >= len(seed_targets):
            mnist_seed_values = {seed: val for seed, val in zip(seed_targets, manual["correct"])}
    # If no manual file, fall back to the known seed values used in plot_predictions().
    if not mnist_seed_values:
        mnist_seed_values = {42: 9625, 123: 9690}

    ymin_c, ymax_c = ax3.get_ylim()
    y_positions = {42: ymin_c + (ymax_c - ymin_c) * 0.6, 123: ymin_c + (ymax_c - ymin_c) * 0.35}
    x_left = ax3.get_xlim()[0]
    x_span = ax3.get_xlim()[1] - ax3.get_xlim()[0]
    text_offset = x_span * 0.30  # place text a bit further to the left
    for seed in seed_targets:
        if seed not in mnist_seed_values:
            continue
        val = mnist_seed_values[seed]
        ax3.axvline(x=val, ymin=0, ymax=1, color="red", linestyle="--", linewidth=1.5)
        ax3.annotate(
            f"Seed {seed}:\nAccuracy={val}",
            xy=(val, y_positions.get(seed, ymin_c + (ymax_c - ymin_c) * 0.5)),
            xytext=(val - text_offset, y_positions.get(seed, ymin_c + (ymax_c - ymin_c) * 0.5)),
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="black", lw=1.5),
        )

    sns.boxplot(
        x="Step",
        y="Happy Count Adjusted",
        data=filtered_df,
        notch=True,
        linewidth=0.75,
        color=palette[1],
        ax=ax4,
        flierprops={
            "marker": "o",
            "markersize": 7.5,
            "markeredgewidth": 0.25,
            "markeredgecolor": palette[0],
            "rasterized": True,
        },
    )
    min_happy = filtered_df.groupby("Step")["Happy Count Adjusted"].min()
    max_happy = filtered_df.groupby("Step")["Happy Count Adjusted"].max()
    ax4.plot(min_happy.index, min_happy.values, color=palette[1], linewidth=0.75, marker="o", markerfacecolor="w", linestyle="--", label="Min")
    ax4.plot(max_happy.index, max_happy.values, color=palette[2], linewidth=0.75, marker="o", markerfacecolor="w", linestyle="--", label="Max")
    ax4.set_title("d.", loc="left", fontsize=22, y=1.02, fontweight="bold")
    ax4.set_xlabel("Step", fontsize=14)
    ax4.set_ylabel("Mean Adjusted Happy Count", fontsize=14)
    ax4.set_xlim(-0.5, 29.5)
    legend_elements4 = [
        Line2D([0], [0], color=palette[2], lw=2, linestyle="--", marker="o", markerfacecolor="w", label="Max"),
        Line2D([0], [0], color=palette[1], lw=2, linestyle="--", marker="o", markerfacecolor="w", label="Min"),
        Line2D([0], [0], color=palette[0], lw=0, linestyle="-", marker="o", markerfacecolor="w", label="Outlier"),
        Patch(facecolor=fill_color, edgecolor="k", label="Range"),
    ]
    ax4.legend(handles=legend_elements4, loc="lower right", frameon=True, fontsize=10, framealpha=1, facecolor="w", edgecolor="k", ncol=2)
    ax4.grid(which="both", linestyle="--", alpha=0.225)

    ax5.plot(mv_summary.index, mv_summary["Median"], color=palette[0], label="Median")
    ax5.plot(mv_summary.index, mv_summary["Max"], linestyle="--", color=palette[2], label="Max")
    ax5.plot(mv_summary.index, mv_summary["Min"], linestyle="--", color=palette[2], label="Min")
    ax5.fill_between(mv_summary.index, mv_summary.min(axis=1), mv_summary.max(axis=1), color=fill_color, label="Range")
    ax5.set_xlabel("Number of Draws", fontsize=14)
    ax5.set_ylabel(r"Simulated ML\nEstimate of $\rho_{21}$", fontsize=14)
    ax5.set_title("e.", loc="left", fontsize=22, y=1.02, fontweight="bold")
    ax5.grid(which="both", linestyle="--", alpha=0.225)
    ax5.legend(loc="upper right", ncol=2, frameon=True, fontsize=10, framealpha=1, facecolor="w", edgecolor="k")

    fig.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "five_models.pdf"), bbox_inches="tight")

    _print_stats("Science topics", science["topics_count"])
    _print_stats("MNIST accuracy", mnist["correct"])
    if mnist_seed_values:
        for seed, val in mnist_seed_values.items():
            print(f"MNIST accuracy for seed {seed}: {val}")
    else:
        print("MNIST seed accuracies unavailable (no seed columns and no manual seeds file).")
    print("The minimum USD/GBP RW forecast is:", random_walks_usuk.min(axis=1).iloc[-1])
    print("The maximum USD/GBP RW forecast is:", random_walks_usuk.max(axis=1).iloc[-1])
    print("The median USD/GBP RW forecast is:", random_walks_usuk.median(axis=1).iloc[-1])
    print(
        "Schelling convergence min/max:",
        schelling_df[schelling_df["Step"] == "Convergence"]["Happy Count"].min(),
        schelling_df[schelling_df["Step"] == "Convergence"]["Happy Count"].max(),
    )
    def _print_mv_stats(draw):
        subset = pd.to_numeric(mv_df.loc[mv_df["draws"] == draw, "rho21"], errors="coerce")
        subset = subset[~subset.isna()]
        if subset.empty:
            print(f"rho21 @ draws={draw}: empty")
            return
        mean = subset.mean()
        sd = subset.std(ddof=1) if len(subset) > 1 else float("nan")
        z = mean / sd if sd not in (0, float("nan")) else float("nan")
        print(
            f"rho21 @ draws={draw}: min={subset.min():.6f}, mean={mean:.6f}, "
            f"max={subset.max():.6f}, sd={sd:.6f}, z={z:.6f}"
        )

    _print_mv_stats(2)
    _print_mv_stats(150)


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

# --- PDF rasterization fixes ---
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['path.simplify'] = False
mpl.rcParams['agg.path.chunksize'] = 0

def _rasterize_axes(ax, rasterize_lines=False):
    """Rasterize heavy plot artists but keep axes/ticks/labels as vector."""
    # Collections: box components, hist bars, etc.
    for coll in ax.collections:
        coll.set_rasterized(True)
    # Patches: skip the axes facecolor patch so the frame stays vector.
    for patch in ax.patches:
        if patch is ax.patch:
            continue
        patch.set_rasterized(True)
    if rasterize_lines:
        for ln in ax.lines:
            ln.set_rasterized(True)

color_list = [
    '#4575b4',  # blue
    '#E6AC00',  # gold
    '#91cf60',  # green
    '#d73027',  # red
]


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from matplotlib import collections as mcoll

# --- PDF rasterization fixes ---
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['path.simplify'] = False
mpl.rcParams['agg.path.chunksize'] = 0

def _rasterize_axes(ax):
    """Rasterize heavy plot artists but keep axes/ticks/labels as vector."""
    ax.set_rasterization_zorder(0.1)
    for coll in ax.findobj(mcoll.Collection):
        coll.set_rasterized(True)
        coll.set_zorder(0.0)
    for ln in ax.lines:
        ln.set_rasterized(True)
        ln.set_zorder(0.0)

color_list = [
    '#4575b4',  # blue
    '#E6AC00',  # gold
    '#91cf60',  # green
    '#d73027',  # red
]




def plot_schelling_examples(figsize=(8.5, 17),colors=color_list):
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import numpy as np

    def _print_conv_stats(label, series):
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            print(f"{label}: no convergence rows")
            return
        arr = series.to_numpy(dtype=float)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        z = mean / sd if sd not in (0, float("nan")) else float("nan")
        print(
            f"{label} -> min={arr.min():.0f}, mean={mean:.2f}, max={arr.max():.0f}, sd={sd:.4f}, z={z:.4f}"
        )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

    # Common styling for boxplots: NO OUTLIERS, BLUE boxes/whiskers/caps/median
    box_kws = dict(
        showfliers=False,            # remove outlier dots entirely
        notch=True,
        linewidth=0.75,
        color=colors[1],             # gold body/lines
        boxprops=dict(edgecolor='k', linewidth=0.75),
        whiskerprops=dict(color=colors[1], linewidth=0.75),
        capprops=dict(color=colors[1], linewidth=0.75),
        medianprops=dict(color='k', linewidth=0.75),
    )

    # Legend elements (Outlier removed)
    legend_elements = [
        Line2D([0], [0], color=colors[3], lw=2, linestyle='--',
               marker='o', markerfacecolor='w', markersize=6, label=r'Max'),
        Line2D([0], [0], color=colors[0], lw=2, linestyle='--',
               marker='o', markerfacecolor='w', markersize=6, label=r'Min'),
    ]

    # ------------------- Panel a -------------------
    df = pd.read_csv(os.path.join(os.getcwd(), '..', 'data', 'schelling', 'schelling_df_25_0.3_0.5.csv'), index_col=0)
    df1 = df[df['Step'] == 'Convergence']
    if not df1.empty:
        _print_conv_stats("[plot_schelling_examples] empty=0.3 threshold=0.5 steps", df1['Happy Count'])
    df = df[df['Step'] != 'Convergence']
    df['Step'] = df['Step'].astype(int)
    df['Happy Count'] = df['Happy Count'].astype(float)
    df['Happy Count Adjusted'] = df.groupby('Step')['Happy Count'].transform(lambda x: x - x.mean())
    filtered_df = df[df['Step'] <= 33]

    sns.boxplot(
        x='Step', y='Happy Count Adjusted', data=filtered_df,
        ax=ax1, **box_kws
    )

    min_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].min()
    max_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].max()

    # Min: blue (colors[0]); Max: red (colors[3])
    ax1.plot(min_happy_count.index, min_happy_count.values, label='Min',
             color=colors[0], linewidth=0.75, marker='o',
             markerfacecolor='w', linestyle='--')
    ax1.plot(max_happy_count.index, max_happy_count.values, label='Max',
             color=colors[3], linewidth=0.75, marker='o',
             markerfacecolor='w', linestyle='--')

    ax1.set_xlabel('Step', fontsize=13)
    ax1.set_ylabel('Mean Adjusted Happy Count', fontsize=13)
    ax1.set_xticks([0, 4, 9, 14, 19, 24, 29, 33])
    ax1.set_axisbelow(True)
    ax1.grid(which="both", linestyle='--', alpha=0.3)

    inset_ax1 = inset_axes(ax1, width="40%", height="25%", loc='lower right', borderpad=2)
    sns.histplot(df1['Happy Count'], ax=inset_ax1, linewidth=0.75,
                 color=colors[0], bins=15, legend=False, alpha=0.9, common_norm=False)
    inset_ax1.xaxis.set_label_position('top')
    inset_ax1.xaxis.tick_top()
    inset_ax1.set_xlabel('Total Steps', fontsize=8)
    inset_ax1.set_ylabel('Frequency', fontsize=8)
    inset_ax1.set_xlim(df1['Happy Count'].min() - 2, df1['Happy Count'].max())
    _ymin, _ymax = inset_ax1.get_ylim()
    inset_ax1.set_ylim(0, _ymax * 1.15 if _ymax else 0.1)
    inset_ax1.set_axisbelow(True)
    inset_ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    inset_ax1.tick_params(axis='both', which='major', labelsize=7)

    ax1.tick_params(width=0.75, length=6.5, axis='both', which='major', labelsize=11)
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w', edgecolor=(0, 0, 0, 1))

    # ------------------- Panel b -------------------
    df = pd.read_csv(os.path.join(os.getcwd(), '..', 'data', 'schelling', 'schelling_df_25_0.5_0.3.csv'), index_col=0)
    df1 = df[df['Step'] == 'Convergence']
    if not df1.empty:
        _print_conv_stats("[plot_schelling_examples] empty=0.5 threshold=0.3 steps", df1['Happy Count'])
    df = df[df['Step'] != 'Convergence']
    df['Step'] = df['Step'].astype(int)
    df['Happy Count'] = df['Happy Count'].astype(float)
    df['Happy Count Adjusted'] = df.groupby('Step')['Happy Count'].transform(lambda x: x - x.mean())
    filtered_df = df[df['Step'] <= 33]

    sns.boxplot(
        x='Step', y='Happy Count Adjusted', data=filtered_df,
        ax=ax2, **box_kws
    )

    min_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].min()
    max_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].max()

    ax2.plot(min_happy_count.index, min_happy_count.values, label='Min',
             color=colors[0], linewidth=0.75, marker='o',
             markerfacecolor='w', linestyle='--')
    ax2.plot(max_happy_count.index, max_happy_count.values, label='Max',
             color=colors[3], linewidth=0.75, marker='o',
             markerfacecolor='w', linestyle='--')

    ax2.set_xlabel('Step', fontsize=13)
    ax2.set_ylabel('Mean Adjusted Happy Count', fontsize=13)
    ax2.set_xticks([0, 4, 9, 14, 19, 24, 29, 33])
    ax2.set_axisbelow(True)
    ax2.grid(which="both", linestyle='--', alpha=0.3)

    inset_ax2 = inset_axes(ax2, width="40%", height="25%", loc='lower right', borderpad=2)
    sns.histplot(df1['Happy Count'], ax=inset_ax2, linewidth=0.75,
                 color=colors[0], bins=15, legend=False, alpha=0.9, common_norm=False)
    inset_ax2.xaxis.set_label_position('top')
    inset_ax2.xaxis.tick_top()
    inset_ax2.set_xlabel('Total Steps', fontsize=8)
    inset_ax2.set_ylabel('Frequency', fontsize=8)
    inset_ax2.set_xlim(df1['Happy Count'].min() - 2, df1['Happy Count'].max())
    _ymin, _ymax = inset_ax2.get_ylim()
    inset_ax2.set_ylim(0, _ymax * 1.15 if _ymax else 0.1)
    inset_ax2.set_axisbelow(True)
    inset_ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    inset_ax2.tick_params(axis='both', which='major', labelsize=7)

    ax2.tick_params(width=0.75, length=6.5, axis='both', which='major', labelsize=11)
    ax2.legend(handles=legend_elements, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w', edgecolor=(0, 0, 0, 1))

    # ------------------- Panel c -------------------
    df = pd.read_csv(os.path.join(os.getcwd(), '..', 'data', 'schelling', 'schelling_df_25_0.5_0.5.csv'), index_col=0)
    df1 = df[df['Step'] == 'Convergence']
    if not df1.empty:
        _print_conv_stats("[plot_schelling_examples] empty=0.5 threshold=0.5 steps", df1['Happy Count'])
    df = df[df['Step'] != 'Convergence']
    df['Step'] = df['Step'].astype(int)
    df['Happy Count'] = df['Happy Count'].astype(float)
    df['Happy Count Adjusted'] = df.groupby('Step')['Happy Count'].transform(lambda x: x - x.mean())
    filtered_df = df[df['Step'] <= 33]

    sns.boxplot(
        x='Step', y='Happy Count Adjusted', data=filtered_df,
        ax=ax3, **box_kws
    )

    min_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].min()
    max_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].max()

    ax3.plot(min_happy_count.index, min_happy_count.values, label='Min',
             color=colors[0], linewidth=0.75, marker='o',
             markerfacecolor='w', linestyle='--')
    ax3.plot(max_happy_count.index, max_happy_count.values, label='Max',
             color=colors[3], linewidth=0.75, marker='o',
             markerfacecolor='w', linestyle='--')

    ax3.set_xlabel('Step', fontsize=13)
    ax3.set_ylabel('Mean Adjusted Happy Count', fontsize=13)
    ax3.set_xticks([0, 4, 9, 14, 19, 24, 29, 33])
    ax3.set_axisbelow(True)
    ax3.grid(which="both", linestyle='--', alpha=0.3)

    inset_ax3 = inset_axes(ax3, width="40%", height="25%", loc='lower right', borderpad=2)
    sns.histplot(df1['Happy Count'], ax=inset_ax3, linewidth=0.75,
                 color=colors[0], bins=15, legend=False, alpha=0.9, common_norm=False)
    inset_ax3.xaxis.set_label_position('top')
    inset_ax3.xaxis.tick_top()
    inset_ax3.set_xlabel('Total Steps', fontsize=8)
    inset_ax3.set_ylabel('Frequency', fontsize=8)
    inset_ax3.set_xlim(df1['Happy Count'].min() - 2, df1['Happy Count'].max())
    _ymin, _ymax = inset_ax3.get_ylim()
    inset_ax3.set_ylim(0, _ymax * 1.15 if _ymax else 0.1)
    inset_ax3.set_axisbelow(True)
    inset_ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    inset_ax3.tick_params(axis='both', which='major', labelsize=7)

    ax3.tick_params(width=0.75, length=6.5, axis='both', which='major', labelsize=11)
    ax3.legend(handles=legend_elements, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w', edgecolor=(0, 0, 0, 1))

    # Layout and save (NO rasterisation anywhere)
    fig.subplots_adjust(wspace=0.25)
    filename = 'schelling_supplement'
    sns.despine(ax=ax1); sns.despine(ax=ax2); sns.despine(ax=ax3)
    ax1.set_title('a.', loc='left', fontsize=22, y=1.0, x=-.05, fontweight='bold')
    ax2.set_title('b.', loc='left', fontsize=22, y=1.0, x=-.05, fontweight='bold')
    ax3.set_title('c.', loc='left', fontsize=22, y=1.0, x=-.05, fontweight='bold')

    outdir = os.path.join(os.getcwd(), '..', 'figures')
    fig.savefig(os.path.join(outdir, filename + '.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(outdir, filename + '.png'),
                bbox_inches='tight', dpi=800)



def plot_compas_recidivism():
    # === Robust plotting cell: neutral scatters + AA/non-AA inset % bars; per-panel hexbin colourbars ===
    # Loads predictions from ./outputs/data; derives summaries if missing; aligns race to UID order.

    import os, io, re, glob, urllib.request
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogLocator, MaxNLocator, FormatStrFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import seaborn as sns

    mpl.rcParams['font.family'] = 'Helvetica'

    # ----------------------------- Paths & discovery -----------------------------
    DATA_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    data_dir  = "../data/compass/results"
    plot_dir  = os.path.join('../figures'); os.makedirs(plot_dir, exist_ok=True)

    # Prefer the canonical 10000/2 file; else pick the most recent match.
    default_pred = os.path.join(data_dir, "uid_oof_predictions_10000seeds_rf_lr_nn_compas_2folds.csv")
    if os.path.exists(default_pred):
        pred_path = default_pred
    else:
        cand = glob.glob(os.path.join(data_dir, "uid_oof_predictions_*seeds_rf_lr_nn_compas_*folds.csv"))
        if not cand:
            raise FileNotFoundError("No predictions CSV found in ./outputs/data/.")
        pred_path = max(cand, key=os.path.getmtime)

    m = re.search(r"uid_oof_predictions_(\d+)seeds_rf_lr_nn_compas_(\d+)folds\.csv$", os.path.basename(pred_path))
    N_SEEDS = int(m.group(1)) if m else 10000
    N_SPLITS = int(m.group(2)) if m else 2
    THRESH = 0.5

    summary_path = os.path.join(data_dir, f"uid_summary_instability_{N_SEEDS}seeds_{N_SPLITS}folds_rf_lr_nn.csv")

    # ----------------------------- Load predictions & build OOF matrices -----------------------------
    df_pred = pd.read_csv(pred_path, index_col=0)
    df_pred.index = df_pred.index.astype(str).str.strip()
    uids = df_pred.index.to_numpy()

    def extract_seednum(colname: str) -> int:
        mm = re.search(r"seed(\d+)$", colname); return int(mm.group(1)) if mm else -1

    def stack_model_oof(df: pd.DataFrame, prefix: str) -> np.ndarray:
        cols = sorted([c for c in df.columns if c.startswith(prefix)], key=extract_seednum)
        if not cols:
            raise ValueError(f"No columns with prefix '{prefix}' in predictions.")
        return df[cols].to_numpy(dtype=float)

    oof_lr_all = stack_model_oof(df_pred, "y_hat_lr_seed")
    oof_rf_all = stack_model_oof(df_pred, "y_hat_rf_seed")
    oof_nn_all = stack_model_oof(df_pred, "y_hat_nn_seed")

    # ----------------------------- Per-UID stats (if summary missing, compute) -----------------------------
    def per_uid_stats(oof: np.ndarray, thr: float):
        mu = np.nanmean(oof, axis=1)
        sd = np.nanstd(oof, axis=1, ddof=1)
        bin_ = (oof >= thr).astype(int)
        ones = bin_.sum(axis=1)
        flips = np.minimum(ones, oof.shape[1] - ones)
        fliprate = flips / oof.shape[1]
        return mu, sd, fliprate

    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path, index_col=0)
        summary.index = summary.index.astype(str).str.strip()
        summary = summary.reindex(uids)
        age    = summary["age"].to_numpy()
        priors = summary["priors"].to_numpy()
        lr_mu, lr_sd, lr_fliprate = summary["lr_mu"].to_numpy(), summary["lr_sd"].to_numpy(), summary["lr_fliprate"].to_numpy()
        rf_mu, rf_sd, rf_fliprate = summary["rf_mu"].to_numpy(), summary["rf_sd"].to_numpy(), summary["rf_fliprate"].to_numpy()
        nn_mu, nn_sd, nn_fliprate = summary["nn_mu"].to_numpy(), summary["nn_sd"].to_numpy(), summary["nn_fliprate"].to_numpy()
    else:
        lr_mu, lr_sd, lr_fliprate = per_uid_stats(oof_lr_all, THRESH)
        rf_mu, rf_sd, rf_fliprate = per_uid_stats(oof_rf_all, THRESH)
        nn_mu, nn_sd, nn_fliprate = per_uid_stats(oof_nn_all, THRESH)
        # Reload raw features to align age/priors
        with urllib.request.urlopen(DATA_URL) as resp:
            df_raw = pd.read_csv(io.BytesIO(resp.read()))
        df_raw = df_raw[
            (df_raw["days_b_screening_arrest"] <= 30) &
            (df_raw["days_b_screening_arrest"] >= -30) &
            (df_raw["is_recid"] != -1) &
            (df_raw["c_charge_degree"] != "O") &
            (df_raw["score_text"] != "N/A")
        ].copy()
        if df_raw["id"].duplicated().any():
            df_raw["id"] = df_raw["id"].astype(str) + "_" + df_raw.groupby("id").cumcount().astype(str)
        df_raw.index = df_raw["id"].astype(str).str.strip()
        age    = df_raw.reindex(uids)["age"].to_numpy()
        priors = df_raw.reindex(uids)["priors_count"].to_numpy()

    # ----------------------------- Race mask aligned to UID order -----------------------------
    with urllib.request.urlopen(DATA_URL) as resp:
        df_race = pd.read_csv(io.BytesIO(resp.read()))
    df_race = df_race[
        (df_race["days_b_screening_arrest"] <= 30) &
        (df_race["days_b_screening_arrest"] >= -30) &
        (df_race["is_recid"] != -1) &
        (df_race["c_charge_degree"] != "O") &
        (df_race["score_text"] != "N/A")
    ].copy()
    if df_race["id"].duplicated().any():
        df_race["id"] = df_race["id"].astype(str) + "_" + df_race.groupby("id").cumcount().astype(str)
    df_race.index = df_race["id"].astype(str).str.strip()
    is_black_aligned = df_race.reindex(uids)["race"].eq("African-American").fillna(False).to_numpy()

    # ----------------------------- Plot configuration & helpers -----------------------------
    colors = globals().get('colors', ['#4575b4',  # blue (from Spectral)
        '#E6AC00',  # gold (custom)
        '#91cf60',  # green (from Spectral)
        '#d73027',  ])  # [primary, accent, ...]

    def _model_stats(mu, fliprate, sd):
        m = np.isfinite(mu) & np.isfinite(fliprate) & np.isfinite(sd)
        x, y, s = mu[m], fliprate[m], sd[m]
        near_005 = np.mean(np.abs(x - 0.5) <= 0.05)
        near_010 = np.mean(np.abs(x - 0.5) <= 0.10)
        rho = np.corrcoef(x, y)[0, 1] if x.size and y.size else np.nan
        pct_flip_pos = np.mean(y > 0.0)
        return {
            "N": int(x.size),
            "flip_p90": float(np.percentile(y, 90)),
            "sd_mean": float(np.mean(s)),
            "near_005": float(near_005),#
            "near_010": float(near_010),
            "rho_mu_flip": float(rho),
            "pct_flip_pos": float(pct_flip_pos),
        }

    def _fmt_stats(st):
        return (
            f"N={st['N']:,}\n"
            f"p90 flip={st['flip_p90']:.3f}\n"
            f"% flip>0: {st['pct_flip_pos']:.1%}\n"
            f"mean SD(p)={st['sd_mean']:.3f}\n"
            f"|μ−0.5|≤0.05: {st['near_005']:.1%}\n"
            f"|μ−0.5|≤0.10: {st['near_010']:.1%}\n"
            f"ρ(μ, flip)={st['rho_mu_flip']:.3f}"
        )

    def _flip_counts(fliprate, race_mask):
        m = np.isfinite(fliprate)
        aa_count  = int(np.sum((fliprate[m] > 0.0) & race_mask[m]))
        non_count = int(np.sum((fliprate[m] > 0.0) & (~race_mask[m])))
        aa_total  = int(np.sum(race_mask[m]))
        non_total = int(np.sum(~race_mask[m]))
        return aa_count, non_count, aa_total, non_total

    def _panel_norm(arr, lo=1.0, hi=99.0, eps=1e-8):
        a = arr[np.isfinite(arr)]
        vmin, vmax = np.percentile(a, [lo, hi]) if a.size else (eps, 1.0)
        vmin = max(float(vmin), eps)
        vmax = max(float(vmax), vmin * (1 + 1e-6))
        use_log = (vmax / vmin) > 50
        return (LogNorm(vmin=vmin, vmax=vmax) if use_log else Normalize(vmin=vmin, vmax=vmax)), use_log

    # Stats for annotation + inset percentages
    stats_lr, stats_rf, stats_nn = _model_stats(lr_mu, lr_fliprate, lr_sd), _model_stats(rf_mu, rf_fliprate, rf_sd), _model_stats(nn_mu, nn_fliprate, nn_sd)
    lr_counts, rf_counts, nn_counts = _flip_counts(lr_fliprate, is_black_aligned), _flip_counts(rf_fliprate, is_black_aligned), _flip_counts(nn_fliprate, is_black_aligned)

    # African-American subset flip percentages (any flip > 0) printed for traceability; figure unchanged
    def _flip_pct_for_mask(fliprate, mask):
        valid = np.isfinite(fliprate) & mask
        total = int(np.sum(valid))
        flips = int(np.sum((fliprate > 0.0) & valid))
        pct = (100.0 * flips / total) if total else float("nan")
        return pct, total

    black_pct_lr, n_black_lr = _flip_pct_for_mask(lr_fliprate, is_black_aligned)
    black_pct_rf, n_black_rf = _flip_pct_for_mask(rf_fliprate, is_black_aligned)
    black_pct_nn, n_black_nn = _flip_pct_for_mask(nn_fliprate, is_black_aligned)
    print(
        "[plot_compas_recidivism] African-American flip% (any flip>0): "
        f"lr={black_pct_lr:.1f}% (N={n_black_lr}), "
        f"rf={black_pct_rf:.1f}% (N={n_black_rf}), "
        f"nn={black_pct_nn:.1f}% (N={n_black_nn})"
    )

    # ----------------------------- Figure: 2×3 -----------------------------
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1], sharey=ax4)
    ax6 = fig.add_subplot(gs[1, 2], sharey=ax4)

    # Top row: neutral scatters with **inset percent bars** (AA vs non-AA)
    def _scatter_plus_inset(ax, mu, fliprate, stats_text, flip_counts_tuple):
        m = np.isfinite(mu) & np.isfinite(fliprate)
        ax.scatter(mu[m], fliprate[m], s=12, alpha=1.0, facecolors='white', edgecolors='k', linewidths=0.6, rasterized=True)
        ax.axvline(THRESH, linestyle="--", color='k')
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"Mean predicted risk ($\bar{p}_i$)", fontsize=14)
        ax.set_ylabel("Flip rate", fontsize=14)
        ax.text(0.02, 0.98, stats_text, va="top", ha="left",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.2"))

        aa_count, non_count, aa_total, non_total = flip_counts_tuple
        aa_pct  = 100.0 * aa_count  / aa_total  if aa_total  > 0 else 0.0
        non_pct = 100.0 * non_count / non_total if non_total > 0 else 0.0

        axins = inset_axes(ax, width="26%", height="28%", loc="upper right", borderpad=0.6)
        bars = axins.bar([0, 1], [aa_pct, non_pct], color=[colors[0], colors[1]], width=0.8, edgecolor='k', linewidth=0.6)
        axins.set_xticks([0, 1]); axins.set_xticklabels(["African American", "Other"], fontsize=8, rotation=90)
        axins.set_ylim(0, 100)
        axins.yaxis.set_visible(False)
        for b, val in zip(bars, [aa_pct, non_pct]):
            axins.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
                       f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        # Keep only the bottom spine (opaque)
        axins.spines['left'].set_visible(False)
        axins.spines['left'].set_linewidth(0.0)
        axins.spines['right'].set_visible(False)
        axins.spines['top'].set_visible(False)
        axins.spines['bottom'].set_alpha(1.0)
        axins.spines['bottom'].set_linewidth(1.0)

    _scatter_plus_inset(ax1, lr_mu, lr_fliprate, _fmt_stats(stats_lr), lr_counts)
    _scatter_plus_inset(ax2, rf_mu, rf_fliprate, _fmt_stats(stats_rf), rf_counts)
    _scatter_plus_inset(ax3, nn_mu, nn_fliprate, _fmt_stats(stats_nn), nn_counts)
    ax2.tick_params(labelleft=False); ax3.tick_params(labelleft=False)

    # Bottom row: feature-instability hexbins (SD) with per-panel colourbars
    def _hex_panel(ax, x, y, z):
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        norm, is_log = _panel_norm(z[mask])
        hb = ax.hexbin(x[mask], y[mask], C=z[mask], reduce_C_function=np.mean, gridsize=27, mincnt=1,
                       norm=norm, cmap='Spectral_r', edgecolor='k')
        cbar = fig.colorbar(hb, ax=ax); cbar.set_label(r"Mean SD of $\hat{p}_i$", fontsize=12)
        if is_log: cbar.ax.yaxis.set_major_locator(LogLocator(base=10))
        else:
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        return hb

    _hex_panel(ax4, age, priors, lr_sd); ax4.set_xlabel("Age", fontsize=14); ax4.set_ylabel("Priors count", fontsize=14)
    _hex_panel(ax5, age, priors, rf_sd); ax5.set_xlabel("Age", fontsize=14); ax5.tick_params(labelleft=False)
    _hex_panel(ax6, age, priors, nn_sd); ax6.set_xlabel("Age", fontsize=14); ax6.tick_params(labelleft=False)

    # Housekeeping
    grid_kws = dict(which="both", linestyle='--', alpha=0.3, zorder=0)
    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.set_axisbelow(True); ax.grid(**grid_kws)

    ax1.set_title('a.', loc='left', fontsize=19, y=1.025, x=-0.075, fontweight='bold')
    ax2.set_title('b.', loc='left', fontsize=19, y=1.025, x=-0.075, fontweight='bold')
    ax3.set_title('c.', loc='left', fontsize=19, y=1.025, x=-0.075, fontweight='bold')
    ax4.set_title('d.', loc='left', fontsize=19, y=1.025, x=-0.075, fontweight='bold')
    ax5.set_title('e.', loc='left', fontsize=19, y=1.025, x=-0.075, fontweight='bold')
    ax6.set_title('f.', loc='left', fontsize=19, y=1.025, x=-0.075, fontweight='bold')

    plt.tight_layout()
    sns.despine()
    out_pdf = os.path.join(plot_dir, "instability_2x3_lr_rf_mlp_inset_percents.pdf")
    plt.savefig(out_pdf)


def plot_three_supplementary_rws(figsize,
                                 colors = ['#001c54', '#E89818', '#8b0000'],
                                 fill_color = (255 / 255, 223 / 255, 0 / 255, 5 / 255)):
    def _print_stats(label, arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            print(f"{label}: empty")
            return
        mean = float(np.mean(arr))
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        z = mean / sd if sd not in (0, float("nan")) else float("nan")
        print(f"{label}: mean={mean:.4f}, min={amin:.4f}, max={amax:.4f}, sd={sd:.4f}, z={z:.4f}")
    def download_and_resample(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        data = data.resample('D').ffill().dropna()  # Forward fill to handle any missing days
        return data

    import matplotlib.ticker as ticker

    btc_data = download_and_resample('BTC-USD', start="2022-10-01", end="2024-06-30")
    rw_btc_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_btc.zip')
    random_walks_btc = pd.read_csv(rw_btc_path, header=None, compression='zip')

    nasdaq_data = download_and_resample('^IXIC', start="2022-10-01", end="2024-06-30")
    rw_nasdaq_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_nasdaq.zip')
    random_walks_nasdaq = pd.read_csv(rw_nasdaq_path, header=None, compression='zip')

    nvidia_data = download_and_resample('NVDA', start="2022-10-01", end="2024-06-30")
    rw_nvidia_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_nvidia.zip')
    random_walks_nvidia = pd.read_csv(rw_nvidia_path, header=None, compression='zip')

    def adjust_index(data, rw_data):
        end_date = data.index[-1]
        start_date = end_date + pd.DateOffset(1)  # Start the random walk data the day after the end_date
        new_index = pd.date_range(start=start_date, periods=len(rw_data), freq='D')
        rw_data.index = new_index
        return rw_data

    random_walks_btc = adjust_index(btc_data, random_walks_btc)
    random_walks_nasdaq = adjust_index(nasdaq_data, random_walks_nasdaq)
    random_walks_nvidia = adjust_index(nvidia_data, random_walks_nvidia)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

    btc_data['Close'].plot(ax=ax1, color=colors[0], legend=False)
    random_walks_btc.min(axis=1).plot(ax=ax1, color=colors[1], alpha=1, linestyle='--')
    random_walks_btc.median(axis=1).plot(ax=ax1, color='k', alpha=1, linestyle='--')
    random_walks_btc.max(axis=1).plot(ax=ax1, color=colors[2], linestyle='--')
    random_walks_btc.quantile(0.05, axis=1).plot(ax=ax1, color='k', linestyle='--', alpha=1, linewidth=0.75)
    random_walks_btc.quantile(0.95, axis=1).plot(ax=ax1, color='k', linestyle='--', alpha=1, linewidth=0.75)

    nasdaq_data['Close'].plot(ax=ax2, color=colors[0], legend=False)
    random_walks_nasdaq.min(axis=1).plot(ax=ax2, color=colors[1], alpha=1, linestyle='--')
    random_walks_nasdaq.median(axis=1).plot(ax=ax2, color='k', alpha=1, linestyle='--')
    random_walks_nasdaq.max(axis=1).plot(ax=ax2, color=colors[2], linestyle='--')
    random_walks_nasdaq.quantile(0.05, axis=1).plot(ax=ax2, color='k', linestyle='--', alpha=1, linewidth=0.75)
    random_walks_nasdaq.quantile(0.95, axis=1).plot(ax=ax2, color='k', linestyle='--', alpha=1, linewidth=0.75)

    nvidia_data['Close'].plot(ax=ax3, color=colors[0], legend=False)
    random_walks_nvidia.min(axis=1).plot(ax=ax3, color=colors[1], alpha=1, linestyle='-')
    random_walks_nvidia.median(axis=1).plot(ax=ax3, color='k', alpha=1, linestyle='--')
    random_walks_nvidia.max(axis=1).plot(ax=ax3, color=colors[2], linestyle='--')
    random_walks_nvidia.quantile(0.05, axis=1).plot(ax=ax3, color='k', linestyle='--', alpha=1, linewidth=0.75)
    random_walks_nvidia.quantile(0.95, axis=1).plot(ax=ax3, color='k', linestyle='--', alpha=1, linewidth=0.75)

    legend_elements = [
        Line2D([0], [0], color=colors[2], linestyle='--',
               label=r'Max', lw=2),
        Line2D([0], [0], color=colors[1], linestyle='--',
               label=r'Min', lw=2),
        Line2D([0], [0], color=colors[0], linestyle='-',
               label=r'Insample', lw=2),
        Line2D([0], [0], color='k', linestyle='--',
               label=r'Median', lw=2),
        Line2D([0], [0], color='k', linestyle='--', alpha=1, linewidth=0.75,
               label=r'95th Percentile', lw=2),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Range')
    ]
#    ax1.legend(handles=legend_elements, loc='lower left', frameon=True,
#               fontsize=11.25, framealpha=1, facecolor='w',
#               edgecolor=(0, 0, 0, 1), ncols=3
#               )
    ax1.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(0.01, 0.35),   # Pushes it outside ax2, vertically centered
        frameon=True,
        fontsize=10,
        framealpha=1,
        facecolor='w',
        edgecolor=(0, 0, 0, 1),
        ncols=3
    )
    ax2.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(0.01, 0.35),   # Pushes it outside ax2, vertically centered
        frameon=True,
        fontsize=10,
        framealpha=1,
        facecolor='w',
        edgecolor=(0, 0, 0, 1),
        ncols=3
    )
    ax3.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(0.01, 0.35),   # Pushes it outside ax2, vertically centered
        frameon=True,
        fontsize=10,
        framealpha=1,
        facecolor='w',
        edgecolor=(0, 0, 0, 1),
        ncols=3
    )
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax1.grid(which="major", linestyle='--', alpha=0.225)
    ax2.grid(which="major", linestyle='--', alpha=0.225)
    ax3.grid(which="major", linestyle='--', alpha=0.225)
    ax1.set_title('a.', loc='left', fontsize=22, y=1.035, fontweight='bold')
    ax2.set_title('b.', loc='left', fontsize=22, y=1.035, fontweight='bold')
    ax3.set_title('c.', loc='left', fontsize=22, y=1.035, fontweight='bold')

    ax1.fill_between(random_walks_btc.index,
                     random_walks_btc.min(axis=1),
                     random_walks_btc.max(axis=1),
                     color=fill_color)
    ax2.fill_between(random_walks_nasdaq.index,
                     random_walks_nasdaq.min(axis=1),
                     random_walks_nasdaq.max(axis=1),
                     color=fill_color)
    ax3.fill_between(random_walks_nvidia.index,
                    random_walks_nvidia.min(axis=1),
                    random_walks_nvidia.max(axis=1),
                    color=fill_color)

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'${y / 1000:.0f}k'))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1:.0f}'))
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'${y / 1:.0f}'))
    ax1.set_ylabel('Bitcoin Price', fontsize=14)
    ax2.set_ylabel('NASDAQ Composite', fontsize=14)
    ax3.set_ylabel('NVidia Share Price', fontsize=14)

    inset_ax = inset_axes(ax1, width="40%", height="25%", loc='upper left', borderpad=2.5)
    sns.histplot(random_walks_btc.iloc[-1], ax=inset_ax,
                 color=colors[0], bins=15,
                 legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax.set_xlabel('Bitcoin Price')
    inset_ax.set_ylabel('Frequency')
    inset_ax.set_axisbelow(True)
    inset_ax.yaxis.set_label_position("right")
    inset_ax.yaxis.tick_right()
    inset_ax.spines['left'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
#    inset_ax.grid(which="both", linestyle='--', alpha=0.3)
#    sns.despine(ax=inset_ax, left=True, top=True, right=False, bottom=False)
    
    inset_ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'${y / 1000:.0f}k'))
    
   
    inset_ax = inset_axes(ax2, width="40%", height="25%", loc='upper left', borderpad=2.5)
    sns.histplot(random_walks_nasdaq.iloc[-1], ax=inset_ax,
                 color=colors[0], bins=15,
                 legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax.set_xlabel('NASDAQ Composite')
    inset_ax.set_ylabel('Frequency')
    inset_ax.set_axisbelow(True)
    inset_ax.yaxis.set_label_position("right")
    inset_ax.yaxis.tick_right()
    inset_ax.spines['left'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
#    inset_ax.grid(which="both", linestyle='--', alpha=0.3)
#    sns.despine(ax=inset_ax, left=True, top=True, right=False, bottom=False)

    inset_ax = inset_axes(ax3, width="40%", height="25%", loc='upper left', borderpad=2.5)
    sns.histplot(random_walks_nvidia.iloc[-1], ax=inset_ax,
                 color=colors[0], bins=15,
                 legend=False, alpha=0.9,
                 common_norm=False
                 )
    inset_ax.set_xlabel('NVidia Share Price')
    inset_ax.set_ylabel('Frequency')
    inset_ax.set_axisbelow(True)
    inset_ax.yaxis.set_label_position("right")
    inset_ax.yaxis.tick_right()
    inset_ax.spines['left'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
#    inset_ax.grid(which="both", linestyle='--', alpha=0.3)
#    sns.despine(ax=inset_ax, left=True, top=True, right=False, bottom=False)
    inset_ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f'${int(x):,}')
    )

    sns.despine(ax=ax1, left=False, top=True, right=True, bottom=False)
    sns.despine(ax=ax2, left=False, top=True, right=True, bottom=False)
    sns.despine(ax=ax3, left=False, top=True, right=True, bottom=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        plt.tight_layout()


    
    filename = 'three_supplementary_rws'
    plt.savefig(os.path.join(os.getcwd(), '..', 'figures', filename + '.pdf'),
                bbox_inches='tight')

    # Final-day descriptive stats (min/mean/max/sd/z) for each random walk
    _print_stats("[plot_three_supplementary_rws] BTC RW final-day", random_walks_btc.iloc[-1].values)
    _print_stats("[plot_three_supplementary_rws] NASDAQ RW final-day", random_walks_nasdaq.iloc[-1].values)
    _print_stats("[plot_three_supplementary_rws] NVDA RW final-day", random_walks_nvidia.iloc[-1].values)


def plot_topics_supplement(figure_path = os.path.join(os.getcwd(),
                                                      '..',
                                                      'figures'),
                           figsize=(14, 9),
                           colors = ['#4575b4', '#E6AC00', '#d73027']):
    def _print_stats(label, series):
        arr = np.asarray(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            print(f"{label}: empty")
            return
        mean = float(np.mean(arr))
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        z = mean / sd if sd not in (0, float("nan")) else float("nan")
        print(f"{label}: mean={mean:.4f}, min={amin:.4f}, max={amax:.4f}, sd={sd:.4f}, z={z:.4f}")

    metapath = os.path.join(os.getcwd(),
                            '..',
                            'data',
                            'bibliometric',
                            'meta_data'
                            )
#    science = pd.read_csv(os.path.join(metapath,
#                                       'metadata_science.csv')
#                          )
    pnas = pd.read_csv(os.path.join(metapath,
                                    'metadata_pnas.csv')
                       )
    nejm = pd.read_csv(os.path.join(metapath,
                                    'metadata_nejm.csv')
                       )
    nature = pd.read_csv(os.path.join(metapath,
                                      'metadata_nature.csv'
                                      )
                         )

#    shape = pd.read_csv(os.path.join(metapath,
#                                     'metadata_shape.csv'
#                                     )
#                        )

    popstudies = pd.read_csv(os.path.join(metapath,
                                          'metadata_popstudies.csv'
                                          )
                             )
    fig, ((ax1, ax2),
          (ax3, ax4)
          ) = plt.subplots(2, 2, figsize=figsize)
    nbins = 25

    sns.histplot(nejm[nejm['random_state'] != 77]['topics_count'],
                 ax=ax1,
                 color=colors[0],
                 bins=nbins)
    ax1_twin = ax1.twinx()
    sns.kdeplot(nejm[nejm['random_state'] != 77]['topics_count'], ax=ax1_twin, color=colors[1])

    sns.histplot(pnas[pnas['random_state'] != 77]['topics_count'],
                 ax=ax2,
                 color=colors[0],
                 bins=nbins)
    ax2_twin = ax2.twinx()
    sns.kdeplot(pnas[pnas['random_state'] != 77]['topics_count'], ax=ax2_twin, color=colors[1])

    sns.histplot(nature[nature['random_state'] != 77]['topics_count'],
                 ax=ax3,
                 color=colors[0],
                 bins=nbins)
    ax3_twin = ax3.twinx()
    sns.kdeplot(nature[nature['random_state'] != 77]['topics_count'], ax=ax3_twin, color=colors[1])

    sns.histplot(popstudies[popstudies['random_state'] != 77]['topics_count'],
                 ax=ax4,
                 color=colors[0],
                 bins=nbins)
    ax4_twin = ax4.twinx()
    sns.kdeplot(popstudies[popstudies['random_state'] != 77]['topics_count'],
                ax=ax4_twin, color=colors[1])

    ax1.set_title('a.', loc='left', fontsize=23, fontweight='bold')
    ax2.set_title('b.', loc='left', fontsize=23, fontweight='bold')
    ax3.set_title('c.', loc='left', fontsize=22, fontweight='bold')
    ax4.set_title('d.', loc='left', fontsize=22, fontweight='bold')
    # ax1.set_xlim(0, 400)
    # ax2.set_xlim(0, 160)
    # ax3.set_xlim(0, ax3.get_xlim()[1])
    # ax4.set_xlim(0, ax4.get_xlim()[1])
    ax1.grid(which="both", linestyle='--', alpha=0.225)
    ax2.grid(which="both", linestyle='--', alpha=0.225)
    ax3.grid(which="both", linestyle='--', alpha=0.225)
    ax4.grid(which="both", linestyle='--', alpha=0.225)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax4.set_axisbelow(True)

    legend_elements1 = [
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1),
              label=r'Histogram'),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='-',
               label=r'Kernel Density', alpha=1)
    ]
    ax2.legend(handles=legend_elements1, loc='center right', frameon=True,
               fontsize=12, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1,  # title='PNAS'
               )

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, None)

    _print_stats("NEJM topics", nejm['topics_count'])
    _print_stats("PNAS topics", pnas['topics_count'])
    _print_stats("Nature topics", nature['topics_count'])
    _print_stats("Population Studies topics", popstudies['topics_count'])

    ax1.set_ylabel('Count: NEJM', fontsize=14)
    ax2.set_ylabel('Count: PNAS', fontsize=14)
    ax3.set_ylabel('Count: Nature', fontsize=14)
    ax4.set_ylabel('Count: Population Studies', fontsize=14)
    for ax_twin in [ax1_twin, ax2_twin, ax3_twin, ax4_twin]:
        ax_twin.set_yticks([])  # Removes right y-axis tick labels
        ax_twin.tick_params(right=False)

    ax1.set_xlabel('Number of topics', fontsize=16)
    ax2.set_xlabel('Number of topics', fontsize=16)
    ax3.set_xlabel('Number of topics', fontsize=16)
    ax4.set_xlabel('Number of topics', fontsize=16)
    ax1_twin.set_ylabel('', fontsize=16)
    ax2_twin.set_ylabel('', fontsize=16)
    ax3_twin.set_ylabel('', fontsize=16)
    ax4_twin.set_ylabel('', fontsize=16)

    # @TODO: this can be modularised when less lazy

    n_topics77 = nejm[nejm['random_state'] == 77]['topics_count'].iloc[0]
    ymin, ymax = ax1.get_ylim()
    ax1.axvline(x=n_topics77,
                ymin=0,
                ymax=1,
                color='red',
                linestyle='--',
                linewidth=2)
    annotation_y = ymin + (ymax - ymin) * 0.8  # 70% up the y-axis
    ax1.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                 xy=(n_topics77,
                     annotation_y),
                 xytext=(n_topics77 - 100,
                         annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1)
                 )
    n_topics77 = pnas[pnas['random_state'] == 77]['topics_count'].iloc[0]

    ymin, ymax = ax2.get_ylim()
    ax2.axvline(x=n_topics77,
                ymin=0,
                ymax=1,
                color='red',
                linestyle='--',
                linewidth=2)

    annotation_y = ymin + (ymax - ymin) * 0.8  # 70% up the y-axis
    ax2.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                 xy=(n_topics77,
                     annotation_y),
                 xytext=(n_topics77 + 500,
                         annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1))

    n_topics77 = nature[nature['random_state'] == 77]['topics_count'].iloc[0]
    ymin, ymax = ax3.get_ylim()
    ax3.axvline(x=n_topics77,
                ymin=0,
                ymax=1,
                color='red',
                linestyle='--',
                linewidth=2)
    annotation_y = ymin + (ymax - ymin) * 0.8  # 70% up the y-axis
    ax3.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                 xy=(n_topics77,
                     annotation_y),
                 xytext=(n_topics77 + 1000,
                         annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1))

    n_topics77 = popstudies[popstudies['random_state'] == 77]['topics_count'].iloc[0]
    ymin, ymax = ax4.get_ylim()
    ax4.axvline(x=n_topics77,
                ymin=0,
                ymax=1,
                color='red',
                linestyle='--',
                linewidth=2)
    annotation_y = ymin + (ymax - ymin) * 0.8  # 70% up the y-axis
    ax4.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                 xy=(n_topics77,
                     annotation_y),
                 xytext=(n_topics77 - 15,
                         annotation_y),
                 ha='center',
                 va='center',
                 fontsize=12,  # Adjust fontsize for better visibility
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=0",
                                 color='black',
                                 mutation_scale=20,
                                 lw=1))
    sns.despine()
    plt.savefig(os.path.join(figure_path,
                             'topic_modelling_supplementary.pdf'),
                bbox_inches='tight')
    plt.tight_layout()


def plot_topics_supplement2(figure_path=os.path.join(os.getcwd(),
                                                     '..',
                                                     'figures'),
                            figsize=(8, 8),
                            colors=['#4575b4', '#E6AC00']):
    def _print_stats(label, series):
        arr = np.asarray(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            print(f"{label}: empty")
            return
        mean = float(np.mean(arr))
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        z = mean / sd if sd not in (0, float("nan")) else float("nan")
        print(f"{label}: mean={mean:.4f}, min={amin:.4f}, max={amax:.4f}, sd={sd:.4f}, z={z:.4f}")

    metapath = os.path.join(os.getcwd(),
                            '..',
                            'data',
                            'bibliometric',
                            'meta_data')
    shape = pd.read_csv(os.path.join(metapath, 'metadata_shape.csv'))

    fig, ax = plt.subplots(figsize=figsize)
    nbins = 25

    sns.histplot(shape[shape['random_state'] != 77]['topics_count'],
                 ax=ax,
                 color=colors[0],
                 bins=nbins)
    ax_twin = ax.twinx()
    sns.kdeplot(shape[shape['random_state'] != 77]['topics_count'],
                ax=ax_twin,
                color=colors[1])

    ax.set_title('a.', loc='left', fontsize=23, fontweight='bold')
    ax.grid(which="both", linestyle='--', alpha=0.225)
    ax.set_axisbelow(True)
    ax.set_xlim(0, None)
    ax.set_ylabel('Count: SHAPE', fontsize=14)
    ax.set_xlabel('Number of topics', fontsize=16)
#    ax_twin.set_yticks([])
#    ax_twin.tick_params(right=False)

    legend_elements = [
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1),
              label=r'Histogram'),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='-',
               label=r'Kernel Density', alpha=1)
    ]
    ax.legend(handles=legend_elements, loc='center right', frameon=True,
              fontsize=12, framealpha=1, facecolor='w',
              edgecolor=(0, 0, 0, 1), ncols=1)

    n_topics77 = shape[shape['random_state'] == 77]['topics_count'].iloc[0]
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=n_topics77,
               ymin=0,
               ymax=1,
               color='red',
               linestyle='--',
               linewidth=2)
    annotation_y = ymin + (ymax - ymin) * 0.8
    ax.annotate('   Seed 77:\n  Topics = ' + str(n_topics77),
                xy=(n_topics77, annotation_y),
                xytext=(n_topics77 - 45, annotation_y),
                ha='center',
                va='center',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="w", facecolor="w"),
                arrowprops=dict(arrowstyle='->',
                                connectionstyle="arc3,rad=0",
                                color='black',
                                mutation_scale=20,
                                lw=1))
#    sns.despine()

    filename = 'topics_supplement_shape'
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, filename + '.pdf'),
                bbox_inches='tight')

    _print_stats("SHAPE topics", shape['topics_count'])
