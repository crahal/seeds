import ast
import os
import math
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



def plot_schelling_examples(colors = ['#001c54', '#E89818', '#8b0000']):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))


    df = pd.read_csv(os.path.join(os.getcwd(),
                                  '..',
                                  'data',
                                  'schelling',
                                  'schelling_df_25_0.3_0.3.csv'),
                     index_col=0)

    df1 = df[df['Step'] == 'Convergence']
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
                notch=True,
                linewidth=0.75,
                linecolor='k',
                color=colors[1],
                ax=ax1,
                flierprops={'marker': 'o', 'markersize': 7.5, 'markeredgewidth': 0.25, 'markeredgecolor': colors[0]}
                )
    min_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].min()
    max_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].max()
    ax1.plot(min_happy_count.index,
             min_happy_count.values,
             label='Min',
             color=colors[1], linewidth=0.75,
             marker='o',
             markerfacecolor='w',
             linestyle='--')
    ax1.plot(max_happy_count.index,
             max_happy_count.values,
             label='Max',
             color=colors[2], linewidth=0.75,
             marker='o',
             markerfacecolor='w',
             linestyle='--')
    ax1.set_xlabel('Step', fontsize=13)
    ax1.set_ylabel('Mean Adjusted Happy Count', fontsize=13)
    ax1.legend()
    ax1.set_xticks([0, 4, 9, 14, 19, 24])
    ax1.set_axisbelow(True)
    ax1.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax1 = inset_axes(ax1, width="40%", height="25%", loc='lower right', borderpad=2)
    sns.histplot(df1['Happy Count'], ax=inset_ax1, linewidth=0.75,
                 color=colors[0], bins=15, legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax1.xaxis.set_label_position('top')
    inset_ax1.xaxis.tick_top()
    inset_ax1.set_xlabel('Total Steps', fontsize=8)
    inset_ax1.set_ylabel('Frequency', fontsize=8)
    inset_ax1.set_xlim(df1['Happy Count'].min() - 2,
                       df1['Happy Count'].max())
    inset_ax1.set_axisbelow(True)
    #    inset_ax1.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    inset_ax1.tick_params(axis='both', which='major', labelsize=7)
    ax1.tick_params(width=0.75, length=6.5, axis='both', which='major', labelsize=11)
    legend_elements2 = [
        Line2D([0], [0], color=colors[2],
               lw=2, linestyle='--', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Max', alpha=1),
        Line2D([0], [0], color=colors[1],
               lw=2, linestyle='--', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Max', alpha=1),
        Line2D([0], [0], color=colors[0], lw=0,
               linestyle='-', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Outlier', alpha=1)
    ]
    ax1.legend(handles=legend_elements2, loc='upper right',
               frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1)
    print(df1['Happy Count'].min(), df1['Happy Count'].max())





    df = pd.read_csv(os.path.join(os.getcwd(),
                                  '..',
                                  'data',
                                  'schelling',
                                  'schelling_df_25_0.3_0.5.csv'),
                     index_col=0)
    df1 = df[df['Step'] == 'Convergence']
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
                notch=True,
                linewidth=0.75,
                linecolor='k',
                color=colors[1],
                ax=ax2,
                flierprops={'marker': 'o', 'markersize': 7.5, 'markeredgewidth': 0.25, 'markeredgecolor': colors[0]}
                )
    min_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].min()
    max_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].max()
    ax2.plot(min_happy_count.index,
             min_happy_count.values,
             label='Min',
             color=colors[1],
             linewidth=0.75,
             marker='o',
             markerfacecolor='w',
             linestyle='--')
    ax2.plot(max_happy_count.index,
             max_happy_count.values,
             label='Max',
             color=colors[2], linewidth=0.75,
             marker='o',
             markerfacecolor='w',
             linestyle='--')
    ax2.set_xlabel('Step', fontsize=13)
    ax2.set_ylabel('Mean Adjusted Happy Count', fontsize=13)
    ax2.legend()
    ax2.set_xticks([0, 4, 9, 14, 19, 24])
    ax2.set_axisbelow(True)
    ax2.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax2 = inset_axes(ax2, width="40%", height="25%", loc='lower right', borderpad=2)
    sns.histplot(df1['Happy Count'], ax=inset_ax2, linewidth=0.75,
                 color=colors[0], bins=15, legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax2.xaxis.set_label_position('top')
    inset_ax2.xaxis.tick_top()
    inset_ax2.set_xlabel('Total Steps', fontsize=8)
    inset_ax2.set_ylabel('Frequency', fontsize=8)
    inset_ax2.set_xlim(df1['Happy Count'].min() - 2,
                       df1['Happy Count'].max())
    inset_ax2.set_axisbelow(True)
    #    inset_ax2.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    inset_ax2.tick_params(axis='both', which='major', labelsize=7)
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
        Line2D([0], [0], color=colors[0], lw=0,
               linestyle='-', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Outlier', alpha=1)
    ]
    ax2.legend(handles=legend_elements2, loc='upper right',
               frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1)
    print(df1['Happy Count'].min(), df1['Happy Count'].max())








    df = pd.read_csv(os.path.join(os.getcwd(),
                                  '..',
                                  'data',
                                  'schelling',
                                  'schelling_df_25_0.5_0.3.csv'),
                     index_col=0)
    df1 = df[df['Step'] == 'Convergence']
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
                notch=True,
                linewidth=0.75,
                linecolor='k',
                color=colors[1],
                ax=ax3,
                flierprops={'marker': 'o', 'markersize': 7.5, 'markeredgewidth': 0.25, 'markeredgecolor': colors[0]}
                )
    min_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].min()
    max_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].max()
    ax3.plot(min_happy_count.index,
             min_happy_count.values,
             label='Min',
             color=colors[1], linewidth=0.75,
             marker='o',
             markerfacecolor='w',
             linestyle='--')
    ax3.plot(max_happy_count.index,
             max_happy_count.values,
             label='Max',
             color=colors[2], linewidth=0.75,
             marker='o',
             markerfacecolor='w',
             linestyle='--')
    ax3.set_xlabel('Step', fontsize=13)
    ax3.set_ylabel('Mean Adjusted Happy Count', fontsize=13)
    ax3.legend()
    ax3.set_xticks([0, 4, 9, 14, 19, 24])
    ax3.set_axisbelow(True)
    ax3.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax3 = inset_axes(ax3, width="40%", height="25%", loc='lower right', borderpad=2)
    sns.histplot(df1['Happy Count'], ax=inset_ax3, linewidth=0.75,
                 color=colors[0], bins=15, legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax3.xaxis.set_label_position('top')
    inset_ax3.xaxis.tick_top()
    inset_ax3.set_xlabel('Total Steps', fontsize=8)
    inset_ax3.set_ylabel('Frequency', fontsize=8)
    inset_ax3.set_xlim(df1['Happy Count'].min() - 2,
                       df1['Happy Count'].max())
    inset_ax3.set_axisbelow(True)
    #    inset_ax3.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    inset_ax3.tick_params(axis='both', which='major', labelsize=7)
    ax3.tick_params(width=0.75, length=6.5, axis='both', which='major', labelsize=11)
    legend_elements2 = [
        Line2D([0], [0], color=colors[2],
               lw=2, linestyle='--', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Max', alpha=1),
        Line2D([0], [0], color=colors[1], lw=2,
               linestyle='-', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Min', alpha=1),
        Line2D([0], [0], color=colors[0], lw=0,
               linestyle='-', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Outlier', alpha=1)
    ]
    ax3.legend(handles=legend_elements2, loc='upper right',
               frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1)
    print(df1['Happy Count'].min(), df1['Happy Count'].max())






    df = pd.read_csv(os.path.join(os.getcwd(),
                                  '..',
                                  'data',
                                  'schelling',
                                  'schelling_df_25_0.5_0.5.csv'),
                     index_col=0)
    df1 = df[df['Step'] == 'Convergence']
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
                notch=True,
                linewidth=0.75,
                linecolor='k',
                color=colors[1],
                ax=ax4,
                flierprops={'marker': 'o', 'markersize': 7.5, 'markeredgewidth': 0.25, 'markeredgecolor': colors[0]}
                )
    min_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].min()
    max_happy_count = filtered_df.groupby('Step')['Happy Count Adjusted'].max()
    ax4.plot(min_happy_count.index,
             min_happy_count.values,
             label='Min',
             color=colors[1], linewidth=0.75,
             marker='o',
             markerfacecolor='w',
             linestyle='--')
    ax4.plot(max_happy_count.index,
             max_happy_count.values,
             label='Max',
             color=colors[2], linewidth=0.75,
             marker='o',
             markerfacecolor='w',
             linestyle='--')
    ax4.set_xlabel('Step', fontsize=13)
    ax4.set_ylabel('Mean Adjusted Happy Count', fontsize=13)
    ax4.legend()
    ax4.set_xticks([0, 4, 9, 14, 19, 24])
    ax4.set_axisbelow(True)
    ax4.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax4 = inset_axes(ax4, width="40%", height="25%", loc='lower right', borderpad=2)
    sns.histplot(df1['Happy Count'], ax=inset_ax4, linewidth=0.75,
                 color=colors[0], bins=15, legend=False, alpha=0.9,
                 common_norm=False)
    inset_ax4.xaxis.set_label_position('top')
    inset_ax4.xaxis.tick_top()
    inset_ax4.set_xlabel('Total Steps', fontsize=8)
    inset_ax4.set_ylabel('Frequency', fontsize=8)
    inset_ax4.set_xlim(df1['Happy Count'].min() - 2,
                       df1['Happy Count'].max())
    inset_ax4.set_axisbelow(True)
    #    inset_ax4.grid(which="both", linestyle='--', alpha=0.3)
    inset_ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y / 1000:.0f}k'))
    inset_ax4.tick_params(axis='both', which='major', labelsize=7)
    ax4.tick_params(width=0.75, length=6.5, axis='both', which='major', labelsize=11)
    legend_elements2 = [
        Line2D([0], [0], color=colors[2],
               lw=2, linestyle='--', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Max', alpha=1),
        Line2D([0], [0], color=colors[1], lw=2,
               linestyle='-', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Min', alpha=1),
        Line2D([0], [0], color=colors[0], lw=0,
               linestyle='-', marker='o',
               markerfacecolor='w', markersize=6,
               label=r'Outlier', alpha=1)
    ]
    ax4.legend(handles=legend_elements2, loc='upper right',
               frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1), ncols=1)
    print(df1['Happy Count'].min(), df1['Happy Count'].max())

    #    for ax in [inset_ax1, inset_ax2, inset_ax3, inset_ax4]:
    #        ax.spines['right'].set_visible(False)
    #        ax.spines['bottom'].set_visible(False)

    fig.subplots_adjust(wspace=0.25)
    filename = 'schelling'
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3)
    sns.despine(ax=ax4)
    ax1.set_title('a.', loc='left', fontsize=22, y=1.0, x=-.05)
    ax2.set_title('b.', loc='left', fontsize=22, y=1.0, x=-.05)
    ax3.set_title('c.', loc='left', fontsize=22, y=1.0, x=-.05)
    ax4.set_title('d.', loc='left', fontsize=22, y=1.0, x=-.05)

    plt.savefig(os.path.join(os.getcwd(), '..', 'figures', filename + '.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(os.getcwd(), '..', 'figures', filename + '.png'),
                bbox_inches='tight', dpi=600)


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
                                       'r2.csv'),
                          index_col=0)
    housing = housing.reset_index()

    titanic = pd.read_csv(os.path.join(os.getcwd(),
                                       '..',
                                       'data',
                                       'titanic',
                                       'results',
                                       'titanic_outputs.csv'))
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

    folding_seed_variance = housing.groupby('Modeling_Seed')['R2'].std().reset_index()
    folding_seed_variance.columns = ['Modeling_Seed', 'R2_variance']
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

    folding_seed_variance = titanic.groupby('Modeling_Seed')['IMV'].std().reset_index()
    folding_seed_variance.columns = ['Modeling_Seed', 'IMV_variance']
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
    ax2.set_xticks([0, 4, 9, 14, 19, 24])
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

    print('The minimum USD/GDP RW forecast is:', random_walks_usuk.min(axis=1)[-1])
    print('The maximum USD/GDP RW RW forecast is:', random_walks_usuk.max(axis=1)[-1])
    print('The median USD/GDP RW RW forecast is:', random_walks_usuk.median(axis=1)[-1])
    print('The minimum BTC RW forecast is:', random_walks_btc.min(axis=1)[-1])
    print('The maximum BTC RW forecast is:', random_walks_btc.max(axis=1)[-1])
    print('The median BTC RW forecast is:', random_walks_btc.median(axis=1)[-1])
    print('The minimum NASDAQ RW forecast is:', random_walks_nasdaq.min(axis=1)[-1])
    print('The maximum NASDAQ RW forecast is:', random_walks_nasdaq.max(axis=1)[-1])
    print('The median NASDAQRW forecast is:', random_walks_nasdaq.median(axis=1)[-1])
    print('The minimum NVIDIA RW forecast is:', random_walks_nvidia.min(axis=1)[-1])
    print('The maximum NVIDIA RW forecast is:', random_walks_nvidia.max(axis=1)[-1])
    print('The median NVIDIA RW forecast is:', random_walks_nvidia.median(axis=1)[-1])

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
    return flat_list


def plot_ffc(ffc, figure_path=None):
    # Define the gridspec layout with adjusted wspace
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.1], hspace=0.25, wspace=0.4)  # Adjusted wspace

    # Create subplots using gridspec
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    colormap = 'Spectral_r'

    gpa = ffc[(ffc['outcome'] == 'gpa') & (ffc['account'] == 'ols')]
    hb2 = ax1.hexbin(gpa['beta'], gpa['r2_holdout'], cmap=colormap, gridsize=55,
                     mincnt=1, linewidths=0.1, edgecolor='k')

    grit = ffc[(ffc['outcome'] == 'grit') & (ffc['account'] == 'ols')]
    hb3 = ax2.hexbin(grit['beta'], grit['r2_holdout'], cmap=colormap, gridsize=55,
                     mincnt=1, linewidths=0.1, edgecolor='k')

    materialHardship = ffc[(ffc['outcome'] == 'materialHardship') & (ffc['account'] == 'ols')]
    hb6 = ax3.hexbin(materialHardship['beta'], materialHardship['r2_holdout'], cmap=colormap, gridsize=55,
                     mincnt=1, linewidths=0.1, edgecolor='k')

    eviction = ffc[(ffc['outcome'] == 'eviction') & (ffc['account'] == 'logit')]
    hb1 = ax4.hexbin(eviction['beta'], eviction['r2_holdout'], cmap=colormap, gridsize=55,
                     mincnt=1, linewidths=0., edgecolor='k')

    jobTraining = ffc[(ffc['outcome'] == 'jobTraining') & (ffc['account'] == 'logit')]
    hb4 = ax5.hexbin(jobTraining['beta'], jobTraining['r2_holdout'], cmap=colormap, gridsize=55,
                     mincnt=1, linewidths=0.1, edgecolor='k')

    layoff = ffc[(ffc['outcome'] == 'layoff') & (ffc['account'] == 'logit')]
    hb5 = ax6.hexbin(layoff['beta'], layoff['r2_holdout'], cmap=colormap, gridsize=45,
                     mincnt=1, linewidths=0.1, edgecolor='k')

    # Manually position the colorbar
    cbar_ax = fig.add_axes([0.85, 0.1085, 0.02, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(hb1, cax=cbar_ax, spacing='uniform', extend='max')
    #    cbar.set_label('Counts', fontsize=14)
    cbar.ax.set_title('Count', fontsize=14)

    for ax, title in zip([ax1, ax2, ax3, ax4, ax5, ax6],
                         ['a.', 'b.', 'c.', 'd.', 'e.', 'f.']):
        ax.set_axisbelow(True)
        ax.grid(which="both", linestyle='--', alpha=0.225)
        ax.set_title(title, loc='left', fontsize=18, y=1.025, x=-0.075)
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

    print(f"Science mean number of topics: {science['topics_count'].mean()}")
    print(f"Science min number of topics: {science['topics_count'].min()}")
    print(f"Science max number of topics: {science['topics_count'].max()}")
    print(f"NEJM mean number of topics: {nejm['topics_count'].mean()}")
    print(f"NEJM min number of topics: {nejm['topics_count'].min()}")
    print(f"NEJM max number of topics: {nejm['topics_count'].max()}")
    print(f"PNAS mean number of topics: {pnas['topics_count'].mean()}")
    print(f"PNAS min number of topics: {pnas['topics_count'].min()}")
    print(f"PNAS max number of topics: {pnas['topics_count'].max()}")
    print(f"Nature mean number of topics: {nejm['topics_count'].mean()}")
    print(f"Nature min number of topics: {nejm['topics_count'].min()}")
    print(f"Nature max number of topics: {nejm['topics_count'].max()}")
    print(f"SHAPE mean number of topics: {shape['topics_count'].mean()}")
    print(f"SHAPE min number of topics: {shape['topics_count'].min()}")
    print(f"SHAPE max number of topics: {shape['topics_count'].max()}")
    print(f"Population Studies mean number of topics: {popstudies['topics_count'].mean()}")
    print(f"Population Studies min number of topics: {popstudies['topics_count'].min()}")
    print(f"Population Studies max number of topics: {popstudies['topics_count'].max()}")

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
                                      'openalex_rn_papers.zip'),
                         compression='zip')
    df_qrng = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                       'openalex_rn_and_quantum_papers.zip'),
                          compression='zip')
    df_hrng = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                       'openalex_rn_and_hardware_papers.zip'),
                          compression='zip')
    df_prng = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                       'openalex_rn_and_pseudo_papers.zip'),
                          compression='zip')
    df_quarng = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                         'openalex_rn_and_quasi_papers.zip'),
                            compression='zip')
    df_yr = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                     'openalex_year_counts.zip'),
                        compression='zip')
    df_yr_dom = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                         'openalex_domain_year_counts.zip'),
                            compression='zip')
    df_dom = pd.read_csv(os.path.join(os.getcwd(), '..' , 'data', 'openalex_returns',
                                      'openalex_domain_counts.zip'),
                         compression='zip')
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