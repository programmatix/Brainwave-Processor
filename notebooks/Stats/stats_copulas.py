import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.stats import t
from scipy.special import gamma
import math
from scipy.optimize import newton
try:
    import pyvinecopulib as pv
except ImportError:
    pv = None

def plot_empirical_copula(df, col1, col2, ax=None):
    x = df[col1]
    y = df[col2]
    n = len(x)
    u = x.rank(method='average')/(n+1)
    v = y.rank(method='average')/(n+1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(u, v, alpha=0.5, s=10)
    ax.set_xlabel(f"F_{col1}")
    ax.set_ylabel(f"F_{col2}")
    ax.set_title("Empirical Copula")
    return ax

def fit_gaussian_copula(df, col1, col2):
    x = df[col1].rank(method='average')/(len(df)+1)
    y = df[col2].rank(method='average')/(len(df)+1)
    x_norm = norm.ppf(x)
    y_norm = norm.ppf(y)
    return np.corrcoef(x_norm, y_norm)[0,1]

def plot_gaussian_copula(df, col1, col2, grid_points=100, ax=None):
    rho = fit_gaussian_copula(df, col1, col2)
    u = np.linspace(0.001, 0.999, grid_points)
    U, V = np.meshgrid(u, u)
    X = norm.ppf(U)
    Y = norm.ppf(V)
    cov = [[1, rho], [rho, 1]]
    Z = multivariate_normal(mean=[0,0], cov=cov).pdf(np.stack([X, Y], axis=-1))
    Z = Z/(norm.pdf(X)*norm.pdf(Y))
    if ax is None:
        fig, ax = plt.subplots()
    cp = ax.contourf(U, V, Z, levels=20)
    plt.colorbar(cp, ax=ax)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(f"Gaussian Copula Density, rho={rho:.2f}")
    return ax

def fit_t_copula(df, col1, col2, df0=4):
    x = df[col1].rank(method='average')/(len(df)+1)
    y = df[col2].rank(method='average')/(len(df)+1)
    X = t.ppf(x, df0)
    Y = t.ppf(y, df0)
    return np.corrcoef(X, Y)[0,1]

def plot_t_copula(df, col1, col2, df0=4, grid_points=100, ax=None):
    rho = fit_t_copula(df, col1, col2, df0)
    u = np.linspace(0.001, 0.999, grid_points)
    U, V = np.meshgrid(u, u)
    X = t.ppf(U, df0)
    Y = t.ppf(V, df0)
    det = 1 - rho**2
    Q = (X**2 - 2*rho*X*Y + Y**2) / det
    const = gamma((df0+2)/2) / (gamma(df0/2) * (df0 * math.pi) * np.sqrt(det))
    f2 = const * (1 + Q/df0)**(-(df0+2)/2)
    f1 = t.pdf(X, df0)
    f2m = t.pdf(Y, df0)
    Z = f2 / (f1 * f2m)
    if ax is None:
        fig, ax = plt.subplots()
    cp = ax.contourf(U, V, Z, levels=20)
    plt.colorbar(cp, ax=ax)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(f"Student-t Copula Density, rho={rho:.2f}, df={df0}")
    return ax

def fit_archimedean_copula(df, col1, col2, family):
    tau = df[[col1, col2]].corr(method='kendall').iloc[0,1]
    if family == 'clayton':
        theta = 2 * tau / (1 - tau)
    elif family == 'gumbel':
        theta = 1 / (1 - tau)
    elif family == 'frank':
        if tau == 0:
            theta = 0
        else:
            func = lambda th: 1 - 4/th * (1 - debye(1, th)) - tau
            theta = newton(func, x0=5)
    else:
        raise ValueError("Unsupported copula family")
    return theta

def plot_clayton_copula(df, col1, col2, grid_points=100, ax=None):
    theta = fit_archimedean_copula(df, col1, col2, 'clayton')
    u = np.linspace(0.001, 0.999, grid_points)
    U, V = np.meshgrid(u, u)
    Z = (1 + theta) * (U * V)**(-theta - 1) * (U**(-theta) + V**(-theta) - 1)**(-2 - 1/theta)
    if ax is None:
        fig, ax = plt.subplots()
    cp = ax.contourf(U, V, Z, levels=20)
    plt.colorbar(cp, ax=ax)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(f"Clayton Copula Density, theta={theta:.2f}")
    return ax

def plot_gumbel_copula(df, col1, col2, grid_points=100, ax=None):
    theta = fit_archimedean_copula(df, col1, col2, 'gumbel')
    u = np.linspace(0.001, 0.999, grid_points)
    U, V = np.meshgrid(u, u)
    Lu = -np.log(U)
    Lv = -np.log(V)
    A1 = Lu**(theta - 1) / U
    B1 = Lv**(theta - 1) / V
    S = Lu**theta + Lv**theta
    W = S**(1/theta)
    C_uv = np.exp(-W)
    Z = C_uv * A1 * B1 * S**(1/theta - 2) * (W + theta - 1)
    if ax is None:
        fig, ax = plt.subplots()
    cp = ax.contourf(U, V, Z, levels=20)
    plt.colorbar(cp, ax=ax)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(f"Gumbel Copula Density, theta={theta:.2f}")
    return ax

def plot_frank_copula(df, col1, col2, grid_points=100, ax=None):
    theta = fit_archimedean_copula(df, col1, col2, 'frank')
    u = np.linspace(0.001, 0.999, grid_points)
    U, V = np.meshgrid(u, u)
    A = np.exp(-theta * U) - 1
    B = np.exp(-theta * V) - 1
    D = np.exp(-theta) - 1
    num = theta * np.exp(-theta * (U + V)) * D
    den = (A * B + D)**2
    Z = num / den
    if ax is None:
        fig, ax = plt.subplots()
    cp = ax.contourf(U, V, Z, levels=20)
    plt.colorbar(cp, ax=ax)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(f"Frank Copula Density, theta={theta:.2f}")
    return ax

def fit_vine_copula(df, family_set=None):
    if pv is None:
        raise ImportError("pyvinecopulib is required for vine copulas")
    data = df.rank(method='average')/(len(df)+1)
    if family_set is None:
        family_set = pv.FamilySet(["gaussian", "student", "clayton", "gumbel", "frank"])
    controls = pv.FitControlsVinecop(family_set=family_set)
    model = pv.Vinecop(data.values, controls)
    return model

def sample_vine_copula(model, n_samples=1000):
    if pv is None:
        raise ImportError("pyvinecopulib is required for vine copulas")
    return model.simulate(n_samples)
