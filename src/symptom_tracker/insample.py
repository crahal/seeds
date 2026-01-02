import matplotlib.pyplot as plt
from matplotlib import collections as matcoll
import numpy as np

def plot_and_print_OR(regr):
    conf = np.exp(regr.conf_int())
    conf['Odds Ratio'] = np.exp(regr.params)
    conf.columns = ['5%', '95%', 'Odds Ratio']
    print(conf)
    conf = conf.sort_values(by='Odds Ratio', ascending=True)
    lines=[]
    for i, j in zip(range(0,15), zip(conf['5%'].tolist(), conf['95%'].tolist())):
        pair = [(j[0], i), (j[1], i)]
        lines.append(pair)
    linecoll = matcoll.LineCollection(lines, colors='k')
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(conf['5%'], conf.index, 'rs', markersize=4)
    ax.plot(conf['95%'], conf.index, 'bo', markersize=4)
    ax.add_collection(linecoll)