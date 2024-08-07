import ast
import os
import math
import json
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas_datareader import data as pdr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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

    ax1.grid(which="both", linestyle='--', alpha=0.3)
    ax2.grid(which="both", linestyle='--', alpha=0.3)

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

    ax1.annotate('$\mu$ = ' + str(np.round(np.mean(final_collisions), 2)) + ', $\sigma$ = ' +
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


def plot_four_simple_examples(figure_path):
    # Load data for 1a here
    file_path_100 = os.path.join(os.getcwd(),
                                 '..',
                                 'data',
                                 'rnom',
                                 'rnom_samples100_seeds100000_results.json')
    with open(file_path_100, 'r') as file:
        data_dict_100 = json.load(file)

    print('Loading in the rnom samples for 100 draws, 1000000')
    print(f"The minimum mean value is: {data_dict_100['min_val']}")
    print(f"The maximum mean value is: {data_dict_100['max_val']}")

    # Load data for 1b here
    df_buffon = pd.read_csv(os.path.join(os.getcwd(),
                                         '..', 'data',
                                         'needles',
                                         'results',
                                         'throw100_25000_5000seeds.csv'),
                            names=['Throws', 'Min', '25th_PC',
                                   'Median', '75th_PC', 'Max'])

    # Load data for 1c here

    collisions, final_collisions = load_collisions()

    #    results_path = os.path.join(os.getcwd(), '..', 'data', 'mcs',
    #                                'results', 'merged_files',
    #                                'merged_csvs.csv')
    #    df = pd.read_csv(results_path, index_col=False)
    #    min_series = df.min(axis=1).sort_values().reset_index()[0]
    #    max_series = df.max(axis=1).sort_values().reset_index()[0]
    #    med_series = df.median(axis=1).sort_values().reset_index()[0]
    #    all_in_one_list = list(df.melt().drop('variable', axis=1).rename({'value': 'A'},
    #                                                                     axis=1)['A'])

    # Load data for 1d here
    size = 366
    btc_data = yf.download('BTC-USD', start="2021-05-08", end="2023-05-15")
    rw_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks.csv')
    random_walks = pd.read_csv(rw_path, header=None)
    btc_data = btc_data / 1000
    random_walks = random_walks / 1000
    index = pd.DataFrame(index=btc_data.index +
                               pd.DateOffset(len(btc_data) + 1))[0:size + 1].index
    random_walks.index = index
    rw_index = random_walks.index
    rw_min = random_walks.min(axis=1)
    rw_max = random_walks.max(axis=1)
    rw_med = random_walks.median(axis=1)

    # Start figure here
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10.5))
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    nbins = 20

    #################
    # Colors
    #################
    colors = ['#001c54', '#E89818', '#8b0000']
    fill_color = (255 / 255, 223 / 255, 0 / 255, 19 / 255)

    #################
    # Figure 1a here#
    #################

    sns.histplot(data_dict_100['max_list'],
                 ax=ax1,
                 color=colors[0],
                 bins=nbins, alpha=1)
    ax1_twin = ax1.twinx()
    sns.kdeplot(data_dict_100['max_list'],
                ax=ax1_twin,
                color=colors[0],
                linestyle='--')

    sns.histplot(data_dict_100['min_list'],
                 ax=ax1,
                 color=colors[1],
                 bins=nbins, alpha=1)
    sns.kdeplot(data_dict_100['min_list'],
                ax=ax1_twin,
                color=colors[1],
                linestyle='--',
                alpha=1)
    ax1_twin.set_ylim(0, .4)
    ax1.set_ylim(0, 14)
    legend_elements1 = [
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1),
              label=r'Maximal Sum'),
        Patch(facecolor=colors[1], edgecolor=(0, 0, 0, 1),
              label=r'Minimal Sum')]

    ax1.legend(handles=legend_elements1, loc='upper right', frameon=True,
               fontsize=11, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1
               )
    ax1_twin.yaxis.set_label_position("right")
    ax1_twin.yaxis.tick_right()
    ax1_twin.set_ylabel('Density', fontsize=16)
    ax1.set_ylabel('Count', fontsize=16)
    ax1.set_xlabel('Draw', fontsize=16)

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
    ax2.fill_between(df_buffon.index, df_buffon['Min'], df_buffon['Max'],
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

    collisions[0].plot(color=colors[1], linestyle='--', ax=ax3)
    collisions[1].plot(color=colors[0], linestyle='-', ax=ax3)
    collisions[2].plot(color=colors[2], linestyle='--', ax=ax3)

    ax3.fill_between(collisions.index, collisions[0], collisions[2],
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

    ax3_inset = ax3.inset_axes([0.015, 0.55, 0.35, 0.35], transform=ax3.transAxes)
    nbins = 25
    sns.histplot(final_collisions,
                 ax=ax3_inset,
                 color=colors[0],
                 bins=nbins,
                 alpha=0.9
                 )
    ax3_twin = ax3_inset.twinx()
    sns.kdeplot(final_collisions,
                ax=ax3_twin,
                color=colors[1],
                linestyle='--',
                linewidth=2
                )
    #    ax3_inset.set_ylim(0, 1500)
    #    ax3_twin.set_ylim(0, 0.045)
    ax3_inset.annotate('$\mu$ = ' + str(np.round(np.mean(final_collisions), 1)) + ', $\sigma$ = ' +
                       str(np.round(np.std(final_collisions), 1)),
                       xy=(0.5, 1), xytext=(0.5, 1.1),
                       xycoords='axes fraction',
                       fontsize=11, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                       arrowprops=dict(arrowstyle='-[, widthB=5.0, lengthB=1',
                                       lw=1.0)
                       )
    #    ax3_twin.tick_params(width=1, length=8, axis='both', which='major', labelsize=14)
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

    #    ax3.plot(min_series.index, min_series,
    #             color=colors[1], linestyle='-',
    #             alpha=0.8)
    #    ax3.plot(max_series.index, max_series,
    #             color=colors[1], linestyle='-',
    #             alpha=0.8)
    #    ax3.plot(med_series.index, med_series,
    #             color=colors[0], linestyle='--',
    #             alpha=0.8)
    #    ax3.set_ylabel(r'Effect Size ($\rm{\hat{\beta}}$)',
    #                   fontsize=16)
    #    ax3.set_xlabel(r'Specification (n)',
    #                   fontsize=16)
    #    legend_elements3 = [Line2D([0], [0],
    #                               color=colors[0],
    #                               lw=1,
    #                               linestyle='--',
    #                               label=r'Median',
    #                               alpha=0.7),
    #                        Line2D([0], [0],
    #                               color=colors[1],
    #                               lw=1, linestyle='-',
    #                               label=r'Min\Max',
    #                               alpha=0.7), ]
    #    ax3.legend(handles=legend_elements3,
    #               loc='upper left', frameon=True,
    #               fontsize=label_fontsize - 4,
    #               framealpha=1, facecolor='w',
    #               edgecolor=(0, 0, 0, 1))
    #    ax3.hlines(y=0,
    #               xmin=ax3.get_xlim()[0],
    #               xmax=ax3.get_xlim()[1],
    #               color='k',
    #               linewidth=1,
    #               linestyle='--',
    #               alpha=0.5)
    #    ax3.fill_between(min_series.index,
    #                     min_series,
    #                     y2=max_series,
    #                     color=colors[1],
    #                     alpha=0.075)
    #
    #    inset_ax = inset_axes(ax3,
    #                          width="37%",
    #                          height="85%",
    #                          loc='lower right',
    #                          bbox_to_anchor=(-0.005, 0.0975, 1, 0.295),
    #                          bbox_transform=ax3.transAxes)
    #    inset_ax.set_xlabel(r'Effect Size ($\rm{\hat{\beta}}$)',
    #                        fontsize=label_fontsize - 5, labelpad=-3)
    #    inset_ax.set_ylabel('Frequency', fontsize=label_fontsize - 5)
    #    inset_ax.hist(all_in_one_list, bins=50, color=colors[0],
    #                  alpha=0.6, edgecolor='k')
    #    letter_fontsize = 24
    #    label_fontsize = 18
    #    mpl.rcParams['font.family'] = 'Helvetica'

    #################
    # Figure 1d here#
    #################
    btc_data['Close'].plot(ax=ax4, color=colors[0])
    rw_min.plot(ax=ax4, color=colors[1], alpha=0.8, linestyle='--')
    rw_max.plot(ax=ax4, color=colors[2], alpha=0.8, linestyle='--')
    rw_med.plot(ax=ax4, color=colors[0], linestyle='-')
    legend_elements4 = [
        Line2D([0], [0], color=colors[2], linestyle='--',
               label=r'Max', lw=2),
        Line2D([0], [0], color=colors[1], linestyle='--',
               label=r'Min', lw=2),
        Line2D([0], [0], color=colors[0], linestyle='-',
               label=r'Median', lw=2),
        Patch(facecolor=fill_color, edgecolor=(0, 0, 0, 1),
              label=r'Variance')
    ]
    ax4.legend(handles=legend_elements4, loc='upper right', frameon=True,
               fontsize=11, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
               )
    ylabels = ['${:,.0f}'.format(x) + 'k' for x in ax4.get_yticks()]
    ax4.set_ylabel(r'Price', fontsize=16)
    ax4.set_yticklabels(ylabels)
    ax4.set_xlabel('')
    ax4.fill_between(rw_index, rw_min, rw_max, color=fill_color)

    ax1.grid(which="both", linestyle='--', alpha=0.3)
    ax2.grid(which="both", linestyle='--', alpha=0.3)
    ax3.grid(which="both", linestyle='--', alpha=0.3)
    ax4.grid(which="major", linestyle='--', alpha=0.3)

    ax2.tick_params(width=1, length=8, axis='both', which='major', labelsize=14)
    ax1.set_title('a.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax2.set_title('b.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax3.set_title('c.', loc='left', fontsize=letter_fontsize, y=1.0)
    ax4.set_title('d.', loc='left', fontsize=letter_fontsize, y=1.0)

    ax2.tick_params(width=1, length=8)
    ax2.set_zorder(3)

    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax4.set_axisbelow(True)

    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1_twin.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=14)

    for ax in [ax3_twin, ax3_inset]:
        sns.despine(ax=ax,
                    left=True,
                    right=True,
                    top=True,
                    bottom=False)

    plt.tight_layout()
    filename = 'four_simple_examples'
    plt.savefig(os.path.join(figure_path, filename + '.pdf'),
                bbox_inches='tight')


def plot_mvprobit(figure_path):
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

    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 4))
    colors = ['#001c54', (255/255, 223/255, 0/255, 10/255), '#8b0000']
    mpl.rcParams['font.family'] = 'Helvetica'
    new_df['Median'].plot(ax=ax1, color=colors[0])
    new_df['Max'].plot(ax=ax1, linestyle='--', color=colors[2])
    new_df['Min'].plot(ax=ax1, linestyle='--', color=colors[2])

    ax1.grid(which = "both", linestyle='--', alpha=0.25)

    ax1.set_title('a.', loc='left', fontsize=18)
    ax1.set_xlabel('Number of Draws', fontsize=14)
    ax1.set_ylabel("Simulated Maximum \n " +
                   r"Likelihood Estimate of $\rho_{21}$", fontsize=14)

    ax1.fill_between(new_df.index, new_df.min(axis=1),
                     new_df.max(axis=1), color=colors[1])

    legend_elements2 = [
        Line2D([0], [0], color=colors[0], lw=2, linestyle='-',
               label=r'Median', alpha=0.7),
       Line2D([0], [0], color=colors[2], lw=2, linestyle='--',
               label=r'Min/Max', alpha=0.7),
       Line2D([0], [0], color='k',linewidth=0.5, linestyle='dashed', alpha=.75,
               label=r'ML Estimate'),
        Patch(facecolor=colors[1], edgecolor=(0,0,0,1),
              label=r'Seed Variance')]
    ax1.legend(handles=legend_elements2, loc='upper right', frameon=True,
               fontsize=12, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=2
              )
    plt.hlines(y=-0.270, xmin=2, xmax=150,
               color='k', linewidth=0.5, linestyle='dashed', alpha=.75)
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
    ax1.grid(which = "both", linestyle='--', alpha=0.3)
    ax2.grid(which = "both", linestyle='--', alpha=0.3)

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
    ax2.grid(which="major", linestyle='--', alpha=0.3)
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
    ax1.grid(which = "both", linestyle='--', alpha=0.3)
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
    ax2.grid(which="major", linestyle='--', alpha=0.3)
    ax2.set_zorder(3)
    sns.despine()
    plt.tight_layout()
    filename = 'buffon_and_rw'
    plt.savefig(os.path.join(figure_path, filename + '.pdf'),
                bbox_inches = 'tight')



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
    return flat_list


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
    ax1.set_title('A.', loc='left', fontsize=letter_fontsize, y=1.035)
    ax2.set_title('B.', loc='left', fontsize=letter_fontsize, y=1.035)
    ax3.set_title('C.', loc='left', fontsize=letter_fontsize, y=1.035)
    ax4.set_title('D.', loc='left', fontsize=letter_fontsize, y=1.035)

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
        ax.annotate('E(ROC-AUC) = ' + str(round(mean, 4)) + ', $\sigma$(ROC-AUC) = ' + str(round(var, 4)),
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
    ax.grid(which = "both", linestyle='--', alpha=0.3)
#    inset_ax.set_title('.', loc='left', fontsize=letter_fontsize - 8, y=1.035, x=-0.12)
    sns.despine()
    plt.savefig(os.path.join(figure_path, 'mcs_seeds.pdf'), bbox_inches='tight')


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
    ax.grid(which = "both", linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(figure_path,
                             'buffon_seeds.pdf'),
                bbox_inches='tight')


def plot_three_predictions(first_wave_10k_stratified_list, figure_path):
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(14, 6.5))
    colors = ['#001c54', '#E89818']
    housing = pd.read_csv(os.path.join(os.getcwd(),
                                       '..',
                                       'data',
                                       'housing',
                                       'results',
                                       'r2.csv'),
                          index_col=0)
    housing = housing.reset_index()
    mnist = pd.read_csv(os.path.join(os.path.join(os.getcwd(),
                                                  '..',
                                                  'data',
                                                  'MNIST',
                                                  'results',
                                                  'mnist_results.csv')))
    print('Covid min: ', np.min(first_wave_10k_stratified_list))
    print('Covid max: ', np.max(first_wave_10k_stratified_list))
    print('Covid mean: ', np.mean(first_wave_10k_stratified_list))
    print('Housing min: ', housing['r2'].min())
    print('Housing max: ', housing['r2'].max())
    print('Housing mean: ', housing['r2'].mean())
    print('MNIST min: ', mnist['correct'].min())
    print('MNIST max: ', mnist['correct'].max())
    print('MNIST mean: ', mnist['correct'].mean())
    nbins = 18
    letter_fontsize = 24
    label_fontsize = 18
    mpl.rcParams['font.family'] = 'Helvetica'
    csfont = {'fontname': 'Helvetica'}
    sns.histplot(first_wave_10k_stratified_list, edgecolor='k',
                 color = colors[0], alpha=0.7, stat='density',
                 ax=ax1, bins=nbins)
    sns.kdeplot(first_wave_10k_stratified_list, color=colors[1], ax=ax1, common_norm=True)

    sns.histplot(housing['r2'], edgecolor='k',
                 color = colors[1], alpha=0.7, stat='density',
                 ax=ax2, bins=nbins)
    sns.kdeplot(housing['r2'], color=colors[0], ax=ax2, common_norm=True)

    sns.histplot(mnist['correct'], edgecolor='k',
                 color = colors[0], alpha=0.7, stat='density',
                 ax=ax3, bins=nbins)
    sns.kdeplot(mnist['correct'], color=colors[1], ax=ax3, common_norm=True)
    ax1.set_ylabel('Density', fontsize=label_fontsize + 2)
    ax2.set_ylabel('')
    ax3.set_ylabel('')
    ax1.set_xlabel('ROC-AUC', fontsize=16)
    ax2.set_xlabel('R$^2$', fontsize=16)
    ax3.set_xlabel('Accuracy', fontsize=16)
    ax1.set_title('a.', loc='left', fontsize=18, y=1.015, x=-0.075)
    ax2.set_title('b.', loc='left', fontsize=18, y=1.015, x=-0.075)
    ax3.set_title('c.', loc='left', fontsize=18, y=1.015, x=-0.075)
    legend_elements1 = [Patch(facecolor=colors[0], edgecolor='k',
                              label=r'Bins', alpha=0.7),
                        Line2D([0], [0], color=colors[1], lw=1, linestyle='-',
                               label=r'KDE', alpha=0.7), ]
    legend_elements2 = [Patch(facecolor=colors[1], edgecolor='k',
                              label=r'Bins', alpha=0.7),
                        Line2D([0], [0], color=colors[0], lw=1, linestyle='-',
                               label=r'KDE', alpha=0.7), ]
    ax1.legend(handles=legend_elements1, loc='center left', frameon=True,
               fontsize=label_fontsize - 2, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1)
               )
    ax2.legend(handles=legend_elements2, loc='center left', frameon=True,
               fontsize=label_fontsize - 2, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1)
               )
    ax3.legend(handles=legend_elements1, loc='center left', frameon=True,
               fontsize=label_fontsize - 2, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1)
               )
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] / 7)
    mean = round(np.nanmean(first_wave_10k_stratified_list), 3)
    var = round(np.nanstd(first_wave_10k_stratified_list), 3)
    ax1.annotate('E(AUC) = ' + str(mean) + ', $\sigma$(AUC) = ' + str(var),
                 xy=(0.5, 0.875), xytext=(0.5, 0.925), xycoords='axes fraction',
                 fontsize=13, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                 arrowprops=dict(arrowstyle='-[, widthB=9.0, lengthB=1',
                                 lw=1.0))
    mean = round(np.nanmean(housing['r2']), 3)
    var = round(np.nanstd(housing['r2']), 3)
    ax2.annotate('E(R$^2$) = ' + str(mean) + ', $\sigma$(R$^2$) = ' + str(var),
                 xy=(0.5, 0.875), xytext=(0.5, 0.925), xycoords='axes fraction',
                 fontsize=13, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                 arrowprops=dict(arrowstyle='-[, widthB=9.0, lengthB=1',
                                 lw=1.0))
    mean = round(np.nanmean(mnist['correct']), 3)
    var = round(np.nanstd(mnist['correct']), 3)
    ax3.annotate('E(acc) = ' + str(mean) + ', $\sigma$(acc) = ' + str(var),
                 xy=(0.5, 0.875), xytext=(0.5, 0.925), xycoords='axes fraction',
                 fontsize=13, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white'),
                 arrowprops=dict(arrowstyle='-[, widthB=9.0, lengthB=1',
                                 lw=1.0))
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax1.grid(which="both", linestyle='--', alpha=0.3)
    ax2.grid(which="both", linestyle='--', alpha=0.3)
    ax3.grid(which="both", linestyle='--', alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, 'prediction_seeds.pdf'), bbox_inches='tight')


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
    g.ax_joint.grid(which = "both", linestyle='--', alpha=0.5)
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
    ax1.grid(which = "both", linestyle='--', alpha=0.4)
    ax3.grid(which = "both", linestyle='--', alpha=0.4)
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

def plot_topics_barplot(figure_path):

    science = pd.read_csv(os.path.join(os.getcwd(),
                                       '..',
                                       'data',
                                       'bibliometric',
                                       'meta_data',
                                       'metadata_science.csv')
                          )
    ###################################
    ## This needs to be uncommented  ##
    ###################################
    pnas = science
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
    ###################################
    ## This needs to be uncommented  ##
    ###################################
    nature = nejm
    # nature = pd.read_csv(os.path.join(os.getcwd(),
    #                     '..',
    #                     'data',
    ##                     'bibliometric',
    #                     'meta_data',
    #                     'metadata_nature.csv')
    #                    )
    fig, ((ax1, ax2),
          (ax3, ax4)
          ) = plt.subplots(2, 2, figsize=(14, 8))
    colors = ['#001c54', '#E89818']
    nbins = 25
    sns.histplot(science['topics_count'],
                 ax=ax1,
                 color=colors[0],
                 bins=nbins)
    ax1_twin = ax1.twinx()
    sns.kdeplot(science['topics_count'], ax=ax1_twin, color=colors[1])
    sns.histplot(nejm['topics_count'],
                 ax=ax2,
                 color=colors[0],
                 bins=nbins)
    ax2_twin = ax2.twinx()
    sns.kdeplot(nejm['topics_count'], ax=ax2_twin, color=colors[1])
    sns.histplot(pnas['topics_count'],
                 ax=ax3,
                 color=colors[0],
                 bins=nbins)
    ax3_twin = ax3.twinx()
    sns.kdeplot(pnas['topics_count'], ax=ax3_twin, color=colors[1])
    sns.histplot(nature['topics_count'],
                 ax=ax4,
                 color=colors[0],
                 bins=nbins)
    ax4_twin = ax4.twinx()
    sns.kdeplot(nature['topics_count'], ax=ax4_twin, color=colors[1])
    ax1.set_title('a.', loc='left', fontsize=23)
    ax2.set_title('b.', loc='left', fontsize=23)
    ax3.set_title('c.', loc='left', fontsize=22)
    ax4.set_title('d.', loc='left', fontsize=22)
    ax1.set_xlim(0, 400)
    ax2.set_xlim(0, 160)
    ax3.set_xlim(0, ax3.get_xlim()[1])
    ax4.set_xlim(0, ax4.get_xlim()[1])
    ax1.grid(which="both", linestyle='--', alpha=0.25)
    ax2.grid(which="both", linestyle='--', alpha=0.25)
    ax3.grid(which="both", linestyle='--', alpha=0.25)
    ax4.grid(which="both", linestyle='--', alpha=0.25)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax4.set_axisbelow(True)
    legend_elements1 = [
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1),
              label=r'Histogram'),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='-',
               label=r'Kernel Density', alpha=0.7)
    ]
    ax1.legend(handles=legend_elements1, loc='upper right', frameon=True,
               fontsize=12, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1, title='Science'
               )
    ax2.legend(handles=legend_elements1, loc='upper left', frameon=True,
               fontsize=12, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1, title='New England Journal\n       Of Medicine'
               )
    ax3.legend(handles=legend_elements1, loc='upper right', frameon=True,
               fontsize=12, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1, title='PNAS'
               )
    ax4.legend(handles=legend_elements1, loc='upper left', frameon=True,
               fontsize=12, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1, title='Nature'
               )

    print(f"Science mean number of topics: {science['topics_count'].mean()}")
    print(f"Science min number of topics: {science['topics_count'].min()}")
    print(f"Science max number of topics: {science['topics_count'].max()}")
    print(f"NEJM mean number of topics: {nejm['topics_count'].mean()}")
    print(f"NEJM min number of topics: {nejm['topics_count'].min()}")
    print(f"NEJM max number of topics: {nejm['topics_count'].max()}")
    print(f"PNAS mean number of topics: {science['topics_count'].mean()}")
    print(f"PNAS min number of topics: {science['topics_count'].min()}")
    print(f"PNAS max number of topics: {science['topics_count'].max()}")
    print(f"Nature mean number of topics: {nejm['topics_count'].mean()}")
    print(f"Nature min number of topics: {nejm['topics_count'].min()}")
    print(f"Nature max number of topics: {nejm['topics_count'].max()}")

    ax3.text(0.5, 0.5, 'PNAS Placeholder', transform=ax3.transAxes,
             fontsize=40, color='red',
             #         alpha=0.5,
             ha='center', va='center', rotation=30)
    ax4.text(0.5, 0.5, 'Nature Placeholder', transform=ax4.transAxes,
             fontsize=40, color='red',
             #         alpha=0.5,
             ha='center', va='center', rotation=30)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylabel('Count', fontsize=16)
        ax.set_xlabel('Number of topics', fontsize=16)
    ax1_twin.set_ylabel('Density', fontsize=16)
    ax2_twin.set_ylabel('Density', fontsize=16)
    ax3_twin.set_ylabel('Density', fontsize=16)
    ax4_twin.set_ylabel('Density', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path,
                             'topic_modelling_seeds_histplot.pdf'),
                bbox_inches='tight')

def load_scientometrics():
    df_rng = pd.read_csv('../data/openalex_returns/openalex_rn_papers.csv')
    df_qrng = pd.read_csv('../data/openalex_returns/openalex_rn_and_quantum_papers.csv')
    df_hrng = pd.read_csv('../data/openalex_returns/openalex_rn_and_hardware_papers.csv')
    df_prng = pd.read_csv('../data/openalex_returns/openalex_rn_and_pseudo_papers.csv')
    df_yr = pd.read_csv('../data/openalex_returns/openalex_year_counts.csv')
    df_yr_dom = pd.read_csv('../data/openalex_returns/openalex_domain_year_counts.csv')
    return df_rng, df_hrng, df_qrng, df_prng, df_yr, df_yr_dom


def desc_print_scientometrics(df_rng, df_hrng, df_qrng, df_prng):
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

def make_table(df_rng, df_hrng, df_qrng, df_prng, column):
    df_rng_val = df_rng[column].value_counts()
    df_hrng_val = df_hrng[column].value_counts()
    df_qrng_val = df_qrng[column].value_counts()
    df_prng_val = df_prng[column].value_counts()
    df_merged = pd.merge(df_rng_val, df_hrng_val, left_index=True, right_index=True, how='left')
    df_merged = df_merged.rename({'count_x': '"Random Numbers"', 'count_y': '"Random Numbers" and "Hardware"'}, axis=1)
    df_merged = pd.merge(df_merged, df_qrng_val, left_index=True, right_index=True, how='left')
    df_merged = df_merged.rename({'count': '"Random Numbers" and "Quantum"'}, axis=1)
    df_merged = pd.merge(df_merged, df_prng_val, left_index=True, right_index=True, how='left')
    df_merged = df_merged.rename({'count': '"Random Numbers" and "Pseudo"'}, axis=1)
    return df_merged


def make_scientometric_ts(df_rng, df_hrng, df_qrng, df_prng, df_yr, domain_df):
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

    df_yr = pd.merge(df_yr, df_yr_rng, left_on='year', right_on='year', how='left')
    df_yr = pd.merge(df_yr, df_yr_hrng, left_on='year', right_on='year', how='left')
    df_yr = pd.merge(df_yr, df_yr_qrng, left_on='year', right_on='year', how='left')
    df_yr = pd.merge(df_yr, df_yr_prng, left_on='year', right_on='year', how='left')
    for rng_type in ['RNG_count', 'HRNG_count', 'QRNG_count', 'PRNG_count']:
        df_yr[rng_type] = df_yr[rng_type] / df_yr['total_count'] * 100

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    fig, (ax1) = plt.subplots(1, 1, figsize=(14, 7))
    df_yr[(df_yr['year'] > 1960) & (df_yr['year'] <= 2020)].set_index('year')[['RNG_count']].plot(ax=ax1)

    ax_inset1 = inset_axes(ax1, width="30%", height="30%", loc='upper left',
                           bbox_to_anchor=(0.05, 0, 1, 1), bbox_transform=ax1.transAxes, borderpad=2)
    df_yr[(df_yr['year'] > 1960) & (df_yr['year'] <= 2020)].set_index('year')[['HRNG_count',
                                                                               'QRNG_count',
                                                                               'PRNG_count']].plot(ax=ax_inset1,
                                                                                                   kind='area')

    ax_inset2 = inset_axes(ax1, width="30%", height="30%", loc='lower right',
                           bbox_to_anchor=(0.0, 0, 1, 1), bbox_transform=ax1.transAxes, borderpad=2)

    domain_df.reindex(index=domain_df.index[::-1]).plot(kind='barh', ax=ax_inset2, legend=False)
    plt.show()
    return df_yr