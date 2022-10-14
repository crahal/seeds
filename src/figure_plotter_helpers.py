import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import ast
import os

def ffc_plotter(df, figure_path):
    def jointplotter(df, outcome, model, counter):
        df1 = df[(df['outcome']==outcome) &
                 (df['account']==model)][0:10000]
        title_list = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.']
        title = title_list[counter]
        g =  sns.jointplot(x=df1['beta'],
                           y=df1['r2_holdout'],
                           kind='hex',
                           marginal_kws=dict(bins=25,
                                             fill=False,
                                             color='k',
                                             zorder=0))
        g.plot_joint(sns.kdeplot, color="r", levels=6)
        g.ax_marg_x.annotate(title, xy=(-0.1, .45), xycoords='axes fraction',
                    ha='left', va='center', fontsize=24)
        if counter in [0, 3]:
            g.ax_joint.set_ylabel(r'Pseudo R$^2$', fontsize=16)
        else:
            g.ax_joint.set_ylabel('')
        if counter in [3,4,5]:
            g.ax_joint.set_xlabel('Lagged Coefficient', fontsize=16)
        else:
            g.ax_joint.set_xlabel('')
        return g

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3)
    figurez = []
    outcomes = ['gpa', 'grit', 'materialHardship',
                'eviction', 'jobTraining', 'layoff']
    for outcome, counter in zip(outcomes, range(0,6)):
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
    gs.update(hspace=0.1)
    plt.savefig(os.path.join(figure_path, 'ffc_seeds.pdf'), bbox_inches='tight')
    plt.show()


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


def covid_plotter(list1, list2, list3, list4):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    ax1.hist(list1)
    ax2.hist(list2)
    ax3.hist(list3)
    ax4.hist(list4)