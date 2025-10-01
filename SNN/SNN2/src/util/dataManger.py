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
# Copyright (C) 2020 Mattia Milani <mattia.milani@nokia.com>

import pandas as pd

from typing import Any, Dict, List

from SNN2.src.actions.actionWrapper import action_selector

class DataManager:


    def __init__(self, columns: List[str]):
        self.columns = columns
        self.data = {k: [] for k in self.columns}

    def append(self, data: Dict[str, Any]):
        for k, v in data.items():
            if k not in self.data.keys():
                raise Exception(f"{k} not found in the original columns, available values: {self.data.keys()}")
            self.data[k].append(v)

    def save(self, output: str) -> None:
        df = pd.DataFrame(self.data)
        action_selector("write", output, df=df, index=False)

