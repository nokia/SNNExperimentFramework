# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
DataManager module
==================

Use this module to manage a data object.
"""

from __future__ import annotations
from collections import UserDict
from copy import deepcopy
import numpy as np

import tensorflow as tf

from typing import Any, Callable, Dict, List, Optional, Tuple
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.decorators.decorators import c_logger

@c_logger
class DataObject(UserDict):
    """DataObject.

    Manage a single data object
    """

    def __init__(self,
                 object: Dict[str, Any],
                 *args,
                 default_field: Optional[str] = None,
                 **kwargs) -> None:
        assert "columns" in object.keys()
        assert "post_operation" in object.keys()

        self.default_field = default_field

        self.columns = object["columns"]
        self.post_operation = object["post_operation"]

        self.post_operation_args = self.__define_args(object)
        self.post_operation_kwargs = self.__define_kwargs(object)

        obj = {
                "columns": self.columns,
                "post_operation": self.post_operation,
                "post_operation_args": self.post_operation_args,
                "post_operation_kwargs": self.post_operation_kwargs,
            }
        col_list = ["columns", "post_operation", "post_operation_args", "post_operation_kwargs"]
        for key in col_list:
            if key in object.keys():
                del object[key]

        obj.update(object)
        self.write_msg(f"Object produced: {obj}", level=LH.DEBUG)
        super().__init__(obj, *args, **kwargs)

    def __return_default(self, object: Dict[str, Any], request: str) -> Tuple[bool, Optional[Any]]:
        if self.post_operation is None:
            return True, None

        if request in list(object.keys()):
            assert len(object[request]) == len(self.post_operation)
            return True, object[request]

        return False, None

    def __define_args(self, object: Dict[str, Any]) -> Optional[List[Tuple[Any, ...]]]:
        exists, res = self.__return_default(object, "post_operation_args")
        if exists:
            return res
        return [() for op in self.post_operation]

    def __define_kwargs(self, object: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        exists, res = self.__return_default(object, "post_operation_kwargs")
        if exists:
            return res
        return [{} for op in self.post_operation]

    def set_default(self, object: Any) -> None:
        if self.default_field is None:
            raise Exception("Cannot set default if the field is None")

        self[self.default_field] = object

    def append_default(self, object: Any) -> None:
        if self.default_field is None:
            raise Exception("Cannot set default if the field is None")

        if self.default_field not in self.keys():
            self.set_default([object])
        else:
            self[self.default_field].append(object)

    def dft(self) -> Any:
        if self.default_field is None:
            raise Exception
        return self[self.default_field]

    def __sub_set(self, object: Any, idx: np.ndarray) -> Any:
        if isinstance(object, np.ndarray):
            self.write_msg(f"{object} is a numpy array, idx required: {idx}, len(object): {len(object)}, len(idx): {len(idx)}")
            return np.take(object, idx)
        if isinstance(object, tf.Tensor):
            self.write_msg(f"{object} is a tensor idx required: {idx}, len(object): {len(object)}, len(idx): {len(idx)}")
            return tf.gather(object, idx)
        if isinstance(object, (tf.DType, type)):
            return object
        raise Exception(f"object {object} subset not supported for type {type(object)}")

    def sub_select(self, indexes: np.ndarray) -> DataObject:
        self.write_msg(f"keys: {list(self.keys())}")
        tmp_obj = deepcopy(self)
        exclude_clm = ["columns", "post_operation", "post_operation_args", "post_operation_kwargs",
                       "TfDataset"]
        # self.write_msg(f"current columns: {tmp_obj.keys()}")
        # self.write_msg(f"Values as np: {tmp_obj['Values']}")
        # self.write_msg(f"Values as dft: {tmp_obj.dft()}")
        # self.write_msg(f"Shape Values: {tmp_obj.dft().shape}")
        # self.write_msg(f"Shape Values as np: {tmp_obj['Values'].shape}")
        dft_obj = tmp_obj.dft()
        if isinstance(dft_obj, List):
            dft_obj = dft_obj[0]

        if "Values" in tmp_obj.keys() and tmp_obj["Values"].shape != dft_obj.shape:
            self.write_msg(f"Super imposing the dft values over the numpy values")
            raise Exception("VALUES SHOULD NOT BE IN THE OBJECT, do not keep the numpy alternative")
            tmp_obj["Values"] = dft_obj.numpy()
        for clm in tmp_obj.keys():
            if clm not in exclude_clm:
                self.write_msg(f"Available items in {clm}: {len(tmp_obj[clm]) if not isinstance(tmp_obj[clm], (tf.DType, type)) else 'no len'}")
                if isinstance(tmp_obj[clm], List):
                    self.write_msg(f"Subsetting {clm} as list")
                    tmp_obj[clm] = [self.__sub_set(obj, indexes) for obj in tmp_obj[clm]]
                else:
                    tmp_obj[clm] = self.__sub_set(tmp_obj[clm], indexes)
        return tmp_obj

    def __concat(self, key, other_obj) -> None:
        if isinstance(self[key], np.ndarray):
            self[key] = np.concatenate((self[key], other_obj), axis=0)
            return
        if isinstance(self[key], tf.Tensor):
            self[key] = tf.concat([self[key], other_obj], 0)
            return
        if isinstance(object, (tf.DType, type)):
            self[key] = object
            return
        raise Exception(f"{type(self[key])} not recognized")

    def concat(self, other: DataObject) -> None:
        exclude_clm = ["columns", "post_operation", "post_operation_args", "post_operation_kwargs"]
        for clm in self.keys():
            if clm not in exclude_clm:
                self.__concat(clm, other[clm])

    def __getitem__(self, item: str) -> Any:
        return self.data[item]

    def __str__(self) -> str:
        return str(self.data)

@c_logger
class DataManager(UserDict):
    """DataManager.

    Manage a data object that contains data descriptors objects.
    Default access to the tensorflow version of the data, else can access
    to numpy and other formats.
    The class automatically manage the reordering of the information.
	"""


    def __init__(self,
                 *args,
                 default_field: Optional[str] = None,
                 **kwargs) -> None:
        """__init__.

	    Define the data objects that the class should manage.
		----------
		"""
        self.default_field = default_field
        super().__init__(*args, **kwargs)
        self.write_msg(f"Data manager of the object: {self}", level=LH.DEBUG)

    def __setitem__(self, key: str, item: Optional[Dict[str, Any]]) -> None:
        # print(help(DataObject))
        if item is None:
            item = {"columns": None, "post_operation": None}
        obj = DataObject(item, default_field=self.default_field, logger=self.logger)
        return super().__setitem__(key, obj)

    def __str__(self) -> str:
        res = "Objects: "
        for key in self:
            res += f"\n{key}: {self[key]}"
        return res

    def dft(self, item: str) -> Any:
        assert self.default_field is not None
        return self.data[item][self.default_field]

    def __getitem__(self, item: str) -> Dict[str, Any] | Any:
        return self.data[item]

    def sub_select(self, indexes: np.ndarray, inplace: bool = False) -> DataManager:
        self.write_msg(f"keys: {self.keys()}")
        tmp_obj = deepcopy(self) if not inplace else self
        for key in tmp_obj:
            tmp_obj[key] = tmp_obj[key].sub_select(indexes)
        # tmp_obj["OriginIndexes"] = None
        # tmp_obj["OriginIndexes"].set_default(tf.convert_to_tensor(indexes))
        return tmp_obj

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for key in self:
            d[key] = self[key].data
        return d

    def transform_toDataset(self, inplace: bool = False,
                            label: Optional[str] = None,
                            keys: Optional[List[str]] = None,
                            limit: Optional[int] = None) -> DataManager:
        tmp_obj = deepcopy(self) if not inplace else self
        keys = list(tmp_obj.keys()) if keys is None else keys
        label = self.default_field if label is None else label

        for key in keys:
            tmp_obj[key][label] = tf.data.Dataset.from_tensor_slices(tmp_obj.dft(key))
            if not limit is None:
                tmp_obj[key][label] = tmp_obj[key][label].take(limit)
        return tmp_obj

    def concat(self, other: DataManager) -> None:
        for key in self:
            if key not in other:
                raise Exception(f"{key} not found in the other DataManager")
            self[key].concat(other[key])

    @classmethod
    def merge(cls, datasets: List[DataManager]) -> DataManager:
        assert len(datasets) >= 1

        if len(datasets) == 1:
            return deepcopy(datasets[0])

        first = deepcopy(datasets[0])
        for other in datasets[1:]:
            first.concat(other)

        return first

    def log_dump(self, label: str) -> None:
        self.write_msg(f"Dump of {label} object")
        for p in self:
            if (self.default_field is None) or \
                    (self[p].default_field == "TfDataset"):
                continue

            tmp_dft = self.dft(p)

            if isinstance(tmp_dft, List):
                tmp_dft = self.dft(p)[0]

            self.write_msg(f"{label}[{p}]: {tmp_dft}", level=LH.DEBUG)
            self.write_msg(f"len({label}[{p}]): {len(tmp_dft)}", level=LH.DEBUG)
            self.write_msg(f"{label}[{p}].dtype: {tmp_dft.dtype}", level=LH.DEBUG)
            self.write_msg(f"{label}[{p}] shape: {tf.shape(tmp_dft)}", level=LH.DEBUG)
            if tmp_dft.dtype is tf.string:
                pass
            elif tmp_dft.shape[0] == 0:
                self.write_msg(f"{label}[{p}] is Empty", level=LH.DEBUG)
            elif len(tf.shape(tmp_dft).numpy()) == 1:
                id, _, count = tf.unique_with_counts(tmp_dft)
                self.write_msg(f"{label}[{p}] unique values: {id.numpy(), count.numpy()}", level=LH.DEBUG)
                self.write_msg(f"{label}[{p}] minimum: {tf.math.reduce_min(tmp_dft)}", level=LH.DEBUG)
                self.write_msg(f"{label}[{p}] maximum: {tf.math.reduce_max(tmp_dft)}", level=LH.DEBUG)
                if "ExpectedLabel" in self:
                    self.write_msg(f"{label}[{p}] expected positives minimum: {tf.math.reduce_min(tf.gather(tmp_dft, tf.where(self['ExpectedLabel']['tf_values'] == 0)[:, 0]))}", level=LH.DEBUG)
                    self.write_msg(f"{label}[{p}] expected positives maximum: {tf.math.reduce_max(tf.gather(tmp_dft, tf.where(self['ExpectedLabel']['tf_values'] == 0)[:, 0]))}", level=LH.DEBUG)
                    self.write_msg(f"{label}[{p}] expected negatives minimum: {tf.math.reduce_min(tf.gather(tmp_dft, tf.where(self['ExpectedLabel']['tf_values'] == 1)[:, 0]))}", level=LH.DEBUG)
                    self.write_msg(f"{label}[{p}] expected negatives maximum: {tf.math.reduce_max(tf.gather(tmp_dft, tf.where(self['ExpectedLabel']['tf_values'] == 1)[:, 0]))}", level=LH.DEBUG)
