import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_loess_curve(df, x_col, y_col, frac=0.66, it=0, delta=0.):
    x = df[x_col].values
    y = df[y_col].values
    smoothed = lowess(y, x, frac=frac, it=it, delta=delta)
    plt.figure()
    plt.scatter(x, y, alpha=0.5)
    plt.plot(smoothed[:,0], smoothed[:,1], color='red')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()
    return smoothed
