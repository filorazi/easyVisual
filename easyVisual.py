import pandas as pd
import json
from matplotlib import pyplot as plt
import numpy as np
 
figsize=(9, 3)


def plot_series(data, labels=None,
                    windows=None,
                    predictions=None,
                    highlights=None,
                    val_start=None,
                    test_start=None,
                    figsize=figsize):
    # Open a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot data
    plt.plot(data.index, data.values, zorder=0, label='data')
    # Rotated x ticks
    plt.xticks(rotation=45)
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels],
                    color=anomaly_color, zorder=2,
                    label='labels')
    # Plot windows
    if windows is not None:
        for _, wdw in windows.iterrows():
            plt.axvspan(wdw['begin'], wdw['end'],
                        color=anomaly_color, alpha=0.3, zorder=1)
    
    # Plot training data
    if val_start is not None:
        plt.axvspan(data.index[0], val_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is None and test_start is not None:
        plt.axvspan(data.index[0], test_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is not None:
        plt.axvspan(val_start, test_start,
                    color=validation_color, alpha=0.1, zorder=-1)
    if test_start is not None:
        plt.axvspan(test_start, data.index[-1],
                    color=test_color, alpha=0.3, zorder=0)
    # Predictions
    if predictions is not None:
        plt.scatter(predictions.values, data.loc[predictions],
                    color=prediction_color, alpha=.4, zorder=3,
                    label='predictions')
    plt.legend()
    plt.tight_layout()


def plot_autocorrelation(data, max_lag=100, figsize=figsize):
    # Open a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Autocorrelation plot
    pd.plotting.autocorrelation_plot(data['value'])
    # Customized x limits
    plt.xlim(0, max_lag)
    # Rotated x ticks
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_histogram(data, bins=10, vmin=None, vmax=None, figsize=figsize):
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot a histogram
    plt.hist(data, density=True, bins=bins)
    # Update limits
    lims = plt.xlim()
    if vmin is not None:
        lims = (vmin, lims[1])
    if vmax is not None:
        lims = (lims[0], vmax)
    plt.xlim(lims)
    plt.tight_layout()


def plot_histogram2d(xdata, ydata, bins=10, figsize=figsize):
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot a histogram
    plt.hist2d(xdata, ydata, density=True, bins=bins)
    plt.tight_layout()


def plot_density_estimator_1D(estimator, xr, figsize=figsize):
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot the estimated density
    xvals = xr.reshape((-1, 1))
    dvals = np.exp(estimator.score_samples(xvals))
    plt.plot(xvals, dvals)
    plt.tight_layout()


def plot_density_estimator_2D(estimator, xr, yr, figsize=figsize):
    # Plot the estimated density
    nx = len(xr)
    ny = len(yr)
    xc = np.repeat(xr, ny)
    yc = np.tile(yr, nx)
    data = np.vstack((xc, yc)).T
    dvals = np.exp(estimator.score_samples(data))
    dvals = dvals.reshape((nx, ny))
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    plt.pcolor(dvals)
    plt.tight_layout()
    # plt.xticks(np.arange(0, len(xr)), xr)
    # plt.yticks(np.arange(0, len(xr)), yr)

def plot_word_cloud(text = None, width = 400, height = 200, mask = None, stopwords = None, show=True, figsize = figsize, savepath=None, imgname="img"):
    '''

    Attributes
    ----------
    text : str, list, np.Array, pd.Series
        if a string is passed all words need to be separated by a blank space (default = None)
    width : int
        width of the image (default = 400)
    height : int
        height of the image (default = 200)
    mask : Image
        Image that will shape the cloud, black pixels are considered as space to be filled, white as empty space (default = None)
    stopwords : List[String]
        Words that will be removed from the text, if None the worldcloud STOPWORD will be used (default = None)
    show : Bool
        Show the image as a plt plot (default = True)
    figsize : (int, int)
        size of the figure shown (default = (9, 3))
    savepath : String 
        If specified saves the image in the path (default = None)
    imgname : String
        If savepath is specified the image is saved as imgname.png (default = "img")

    '''
    from wordcloud import WordCloud, STOPWORDS
    plt.figure(figsize=figsize)
    if mask is not None:
        mask = np.array(mask)
    if type(text) is str:
        wc = WordCloud(width = width, height = height, mask = mask, stopwords = stopwords).generate(text)
    else:
        wc = WordCloud(width = width, height = height, mask = mask, stopwords = stopwords).generate(" ".join(text))
    plt.imshow(wc)
    if pd.notnull(savepath):
        plt.savefig(fname=savepath.join(imgname))
    plt.axis("off")
    plt.show()

text = "a s d f aijg g d aina s g gsa eh td jtyk uk l df j ra ra gar gr str jsy jdtyk  da rgr gr agsr htr jd jd jtsd b adrb b vr r rs ht hdt rj jdy  da a v ar var sr htr rtsf js jj sjtj"

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('a.jpg',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
plot_word_cloud(text, width= 1000, mask = None)
