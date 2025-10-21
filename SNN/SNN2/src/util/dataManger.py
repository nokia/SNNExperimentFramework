# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

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

