import matplotlib.pyplot as plt 

import seaborn as sns
import pandas as pd 
import numpy as np

def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, alpha=None, **kwargs):
    if smooth > 1 or alpha:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        new_data = []
        y = np.ones(smooth)
        for i, datum in enumerate(data):
            new_data.append(datum.copy(True))
            datum = new_data[-1]
            if alpha:
                datum[value] = datum[value].ewm(alpha=alpha).mean()
            else:
                x = np.asarray(datum[value])
                z = np.ones(len(x))
                smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
                datum[value] = smoothed_x
        data = new_data

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="white", font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, errorbar='sd', **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
    Changes the colorscheme and the default legend style, though.
    """
    # plt.legend(loc='best').set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:
    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)