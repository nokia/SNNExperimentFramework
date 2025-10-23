# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2020 Mattia Milani <mattia.milani.ext@nokia.com>

"""
plotter module
==============

Use this module to create beautiful plots

"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from SNN2.src.decorators.decorators import plot_functions
from SNN2.src.util.strings import s

from typing import List, Callable, Optional, Tuple, Any

class plotter:
    """plotter.
    Class to plot single dataframe statistics
    """


    def __init__(self, data: pd.DataFrame,
                 show: bool = False,
                 format: List[str] = ["pdf"],
                 font_scale: float = 1.5,
                 palette=None):
        """__init__.

        Parameters
        ----------
        data : pd.DataFrame
            dataframe that should be plotted
        show : bool
            show flag
        """
        self.__df = data
        self.__show = show
        self.__format = format
        self.__font = "times"
        self.__font_scale = font_scale
        self.__palette = palette
        self.plot = None

        # Seaborn settings
        self.sns_reset()
        # sns.set_theme(context="paper")

    def __call__(self, f_name, *args, **kwargs):
        if f_name in plot_functions.keys():
            self.plot = plot_functions[f_name](*args, data=self.__df, **kwargs, palette=self.__palette)
        else:
            raise Exception(f"{f_name} Not implemented")

    def save(self, outputFile: str,
             legend: bool = True,
             **kwargs) -> None:
        for fmt in self.__format:
            outputFile = f"{outputFile.split('.')[0]}.{fmt}"
            plt.savefig(outputFile, format=fmt, bbox_inches="tight")
        plt.close()
        # self.sns_reset()

    def set_legend(self, x: float = 0.5,
                   y: float = -0.32,
                   loc: str = "lower center",
                   ncol: int = 3,
                   borderaxespad: int = 0,
                   **kwargs) -> None:
        self.plot.axes.legend(bbox_to_anchor=(x, y),
                              loc=loc,
                              ncol=ncol,
                              borderaxespad=borderaxespad,
                              **kwargs)

    def move_legend(self, *args, obj: Optional[Any] = None, **kwargs):
        if obj is None:
            obj = self.plot
        sns.move_legend(obj, *args, **kwargs)

    def sns_set(self, *args, **kwargs) -> None:
        sns.set(*args, **kwargs)

    def sns_set_api(self, interface: Callable = sns.set, *args, **kwargs) -> None:
        interface(*args, **kwargs)

    def sns_reset(self) -> None:
        self.sns_set_api(sns.set, font_scale=self.__font_scale)
        self.sns_set_api(sns.set_style, "white")
        self.sns_set_api(sns.set_palette, sns.color_palette())

    def set(self, *args, **kwargs) -> None:
        if self.plot is not None:
            self.plot.set(*args, **kwargs)
        else:
            raise Exception("A plot must be first created")

    def pdf(self,*args, **kwargs) -> None:
        self.plot = sns.kdeplot(data=self.__df, *args, **kwargs)

    def ecdf(self,*args, **kwargs) -> None:
        self.plot = sns.ecdfplot(data=self.__df, *args, **kwargs)

    def scatter(self,*args, **kwargs) -> None:
        self.plot = sns.scatterplot(data=self.__df, *args, **kwargs)
