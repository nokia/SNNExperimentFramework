# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Optional, Callable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mplt
import matplotlib.pylab as mpyl
from SNN2.src.decorators.decorators import plot
from sklearn.preprocessing import normalize as norm
from statsmodels.graphics.tsaplots import plot_acf


@plot
def kde(*args, data: pd.DataFrame = None, **kwargs) -> None:
    return sns.kdeplot(data=data, *args, **kwargs)

@plot
def joint(*args, data: pd.DataFrame = None, **kwargs) -> None:
    return sns.jointplot(data=data, *args, **kwargs)

@plot
def ecdf(*args, data: pd.DataFrame = None, **kwargs) -> None:
    return sns.ecdfplot(data=data, *args, **kwargs)

@plot
def displot(*args, data: pd.DataFrame = None, **kwargs) -> mplt.axes.Axes:
    return sns.displot(data=data, *args, **kwargs)

@plot
def scatter(*args, data: pd.DataFrame = None, **kwargs) -> None:
    return sns.scatterplot(data=data, *args, **kwargs)

@plot
def line(*args, data: pd.DataFrame = None, **kwargs) -> mplt.axes.Axes:
    return sns.lineplot(data=data, *args, **kwargs)

@plot
def violin(*args, data: pd.DataFrame = None, **kwargs) -> mplt.axes.Axes:
    return sns.violinplot(data=data, *args, **kwargs)

@plot
def boxplot(*args, data: pd.DataFrame = None, **kwargs) -> mplt.axes.Axes:
    return sns.boxplot(data=data, *args, **kwargs)

@plot
def distplot(*args, data: pd.DataFrame = None, **kwargs) -> mplt.axes.Axes:
    return sns.boxplot(data=data, *args, **kwargs)

def draw_heatmap(pvt_kwarg=None, htm_kwarg=None, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot_table(**pvt_kwarg)
    sns.heatmap(data=d, **htm_kwarg, **kwargs)

@plot
def heatplot(*args,
             data: Optional[pd.DataFrame] = None,
             col: Optional[str] = None,
             pivot_kwarg: Optional[Dict[str, Any]] = None,
             heatmap_kwarg: Optional[Dict[str, Any]] = None,
             cbar_kws: Optional[Dict[str, Any]] = {'label': 'Action P'},
             **kwargs) -> mplt.axes.Axes:
    assert col is not None

    fg = sns.FacetGrid(data, col=col)
    cbar_ax = fg.fig.add_axes([1.0, .15, .03, .7])
    fg.map_dataframe(draw_heatmap, pvt_kwarg=pivot_kwarg, htm_kwarg=heatmap_kwarg,
                     cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    return fg

@plot
def lineStack(*args,
              data: Optional[pd.DataFrame] = None,
              x: Optional[str] = None,
              y: Optional[str] = None,
              hue: Optional[str] = None,
              normalize: bool = False,
              **kwargs) -> mplt.axes.Axes:
    if data is None:
        raise Exception("The dataframe for a lineStack cannot be None")

    x = data[x].unique()
    a = np.stack([data[data[hue] == h][y].values for h in data[hue].unique()], axis=-1)
    if normalize:
        a = norm(a, axis=1, norm='l1')
    a = np.around(a, 3)
    a = np.transpose(a)
    y_plt = {
            h: a[i] for i, h in enumerate(data[hue].unique())
        }
    fig, ax = mplt.pyplot.subplots()
    ax.stackplot(x, y_plt.values(), *args, labels=y_plt.keys(), **kwargs)
    ax.legend(loc='upper left')
    return ax

@plot
def heatmap(*args, data: pd.DataFrame = None, **kwargs) -> mplt.axes.Axes:
    return sns.heatmap(data=data, *args, **kwargs)

@plot
def cat(*args, data: Optional[pd.DataFrame] = None, **kwargs) -> None:
    if data is None:
        raise Exception("The dataframe for a catplot cannot be None")
    return sns.catplot(data=data, *args, **kwargs)

@plot
def histplot(*args, data: Optional[pd.DataFrame] = None, **kwargs) -> None:
    if data is None:
        raise Exception("The dataframe for a catplot cannot be None")
    return sns.histplot(data=data, *args, **kwargs)

@plot
def multiDimEvolution(*args,
                      data: Optional[pd.DataFrame] = None,
                      facet_kwargs: Optional[Dict[str, Any]] = None,
                      map_fct: Optional[Callable] = None,
                      **kwargs) -> mplt.axes.Axes:
    if data is None:
        raise Exception("The dataframe cannot be None")
    default_facet_kw = {}
    if facet_kwargs is not None:
        default_facet_kw.update(facet_kwargs)

    map_fct = sns.kdeplot if map_fct is None else map_fct
    fg = sns.FacetGrid(data, **default_facet_kw)
    fg.map(map_fct, *args, **kwargs)
    return fg

@plot
def multiLineplot(key_col: str, *args,
                  data: Optional[pd.DataFrame] = None,
                  **kwargs) -> mplt.axes.Axes:
    if data is None:
        raise Exception("The dataframe cannot be None")

    objs = data[key_col].unique()
    ax = None
    for c in objs:
        sub_df = data[data[key_col] == c]
        if ax is None:
            ax = sns.lineplot(sub_df, *args, **kwargs)
        else:
            ax = sns.lineplot(sub_df, *args, ax=ax, **kwargs)
        ax.fill_between(sub_df["Step"], sub_df["Value"], 0.0, alpha=1)
    return ax

@plot
def acf(*args, data: Optional[pd.DataFrame] = None, column: Optional[str] = None, **kwargs):
    if column is None:
        raise Exception("A column must be specified")
    return plot_acf(data[column].values)

@plot
def bar(*args, data: Optional[pd.DataFrame] = None, **kwargs):
    if data is None:
        raise Exception("The dataframe cannot be None")
    ax = sns.barplot(data=data, *args, **kwargs)
    return ax
