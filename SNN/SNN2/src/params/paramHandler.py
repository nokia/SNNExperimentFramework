# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import ast
import copy
import numpy as np
from SNN2.src.decorators.decorators import c_logger

from SNN2.src.util.strings import s
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.IOHandler import IOHandler as IOH
from SNN2.src.io.configuration import ConfHandler as CH
from SNN2.src.params.paramWrapper import Param_selector

from operator import itemgetter
from hashlib import blake2b
from typing import Any, Dict, List, Optional, Tuple, Union

@c_logger
class ParamHandler:

    def __init__(self, objects: List[Dict[str, str]],
                 ioh: IOH):
        self.ioh = ioh
        self.objects = {}
        for obj in objects:
            self.objects[obj[s.io_name_key]] = self.define_object(obj)
        self.write_msg("ParamHandler class initialized", level=LH.DEBUG)
        self._force_hash = None

    @classmethod
    def filter(cls, ph: ParamHandler, filter: str) -> ParamHandler:
        keys = ph.filter_keys(filter)
        new_ph = copy.deepcopy(ph)
        new_ph.objects = {k: new_ph.get_handler(k) for k in new_ph.objects.keys() & keys}
        return new_ph

    @classmethod
    def get_param_sorting_order(cls, items: List[Dict[str, str]], io: IOH) -> List[int]:
        itms_names = np.array([x[s.io_name_key] for x in items])
        itms_names = np.append(itms_names, list(io.objects.keys()))
        itms_str = np.array(['-'.join([x[s.param_value], x.get(s.param_action_args, ""), x.get(s.param_action_kwargs, "")]) for x in items])
        itms_str = np.append(itms_str, ["-"]*len(list(io.objects.keys())))
        assert len(itms_names) == len(itms_str)
        itms_order = np.array([np.inf]*len(itms_str))
        # itms_to_solve = np.array([x.count('{') for x in itms_str])
        itms_str = np.array([re.sub(r"\\\{", r"\#", x) for x in itms_str])
        itms_str = np.array([re.sub(r"\\\}", r"\$", x) for x in itms_str])
        itms_objects = np.array([np.array(re.findall(r"\{[^\}]*\}", x)) for x in itms_str], dtype=object)
        itms_objects = np.array([np.array([x.replace('{', '').replace('}', '') for x in y]) for y in itms_objects], dtype=object)
        itms_to_solve = np.array([len(x) for x in itms_objects])
        itms_order[itms_to_solve == 0] = 0
        changes = True
        counter = 0
        while changes:
            changes = False
            subset_idx = np.where(itms_order <= counter)[0]
            current_subset = itms_names[subset_idx]
            still_to_solve_idx = np.where(itms_order == np.inf)[0]
            still_to_solve = itms_objects[still_to_solve_idx]
            solved_idx = still_to_solve_idx[np.where(np.array([all(np.in1d(x, current_subset)) for x in still_to_solve]))[0]]
            if len(solved_idx) > 0:
                changes = True
            if len(solved_idx) == 0 and len(still_to_solve_idx) > 0:
                print(still_to_solve)
                print("csv_path" in itms_names)
                raise Exception(f"Was not possible to solve some names")

            itms_order[solved_idx] = counter+1
            counter += 1
        itms_order[itms_order == np.inf] = counter+1
        return itms_order

    @classmethod
    def from_cfg(cls, conf: CH, *args, **kwargs) -> ParamHandler:
        items = []
        for key in conf.cfg.keys():
            if len(conf.cfg.items(key)) > 0:
                d = dict(conf.cfg.items(key))
                d[s.io_name_key] = key
                if d[s.io_type_key] in s.param_types:
                    items.append(d)
        # items = sorted(items, key=lambda x: '-'.join([x[s.param_value], x.get(s.param_action_args, ""), x.get(s.param_action_kwargs, "")]).count('{'))
        order = cls.get_param_sorting_order(items, args[0])
        for d, o in zip(items, order):
            d["Order"] = o
        items = sorted(items, key=lambda x: x["Order"])
        for d in items:
            del d["Order"]
        return cls(items, *args, **kwargs)

    def define_object(self, obj: Dict[str, str]):
        """define_object.

        Parameters
        ----------
        """
        self.write_msg(f"Evaluation of {obj[s.param_value]}", level=LH.DEBUG)
        obj[s.param_value] = self.evaluate(obj[s.param_value])
        if obj[s.io_type_key] in s.param_args_ast:
            self.write_msg(f"{obj[s.param_value]} is in the args list")
            if s.param_action_args in obj.keys() and obj[s.param_action_args] is not None:
                self.write_msg(f"{obj[s.param_value]} Has an args list")
                coma = ""
                if ',' not in obj[s.param_action_args]:
                    coma=","
                obj[s.param_action_args] = f"({self.evaluate(obj[s.param_action_args])}{coma})"
                obj[s.param_action_args] = ast.literal_eval(obj[s.param_action_args])
            else:
                self.write_msg(f"{obj[s.param_value]} doesn't have an args list")
                obj[s.param_action_args] = ()
            if s.param_action_kwargs in obj.keys() and obj[s.param_action_kwargs] is not None:
                self.write_msg(f"{obj[s.param_value]} has a kwargs list")
                obj[s.param_action_kwargs] = f"{{{self.evaluate(obj[s.param_action_kwargs])}}}"
                obj[s.param_action_kwargs] = ast.literal_eval(obj[s.param_action_kwargs])
            else:
                self.write_msg(f"{obj[s.param_value]} doesn't have a kwargs list")
                obj[s.param_action_kwargs] = {}

        self.write_msg(f"After evaluation: {obj[s.param_value]}", level=LH.DEBUG)
        return Param_selector(obj[s.io_type_key], obj, logger=self.logger)

    def evaluate_obj(self, obj: str) -> Tuple[str, str]:
        obj = obj.replace('{', '').replace('}', '')
        if obj in self.objects.keys():
            self.write_msg(f"{obj} is contained in the self objects", level=LH.DEBUG)
            return obj, self.objects[obj].value
        self.write_msg(f"{obj} is not contained in the self objects, asking the IOH", level=LH.DEBUG)
        return obj, self.ioh.evaluate_obj(obj)[1]

    def evaluate_objs(self, objs: List[str]) -> List[Tuple[str, str]]:
        return [self.evaluate_obj(obj) for obj in objs]

    def evaluate(self, pth: str) -> str:
        """evaluate.
        Evaluate a path if it contains other objects in it

        Parameters
        ----------
        pth : str
            pth

        Returns
        -------
        str

        """
        self.write_msg(f"Evaluating {pth}", level=LH.DEBUG)
        pth = re.sub(r"\\\{", r"\#", pth)
        pth = re.sub(r"\\\}", r"\$", pth)
        result = re.findall(r"\{[^\}]*\}", pth)
        res = {x[0]: x[1] for x in self.evaluate_objs(result)}
        pth = pth.replace('!', '').format(**res)
        pth = pth.replace(r"\#", "{")
        pth = pth.replace(r"\$", "}")
        self.write_msg(f"After the evaluation: {pth}", level=LH.DEBUG)
        return pth

    def get(self, elem: str, literal_eval: bool = False) -> Union[str, Any]:
        """get.
        Returns the path of an object

        Parameters
        ----------
        elem : str
            elem

        Returns
        -------
        str

        """
        self.write_msg(f"Required the value for object: {elem}", level=LH.DEBUG)
        if elem not in self.objects:
            raise Exception(f"Available objects: {self.objects.keys()}, {elem} not found")
        return self.objects[elem].value if not literal_eval else ast.literal_eval(self.objects[elem].value)

    def get_handler(self, elem: str) -> _param:
        """get.
        Returns the handler of an object

        Parameters
        ----------
        elem : str
            elem

        Returns
        -------
        str

        """
        self.write_msg(f"Required the handler for object: {elem}", level=LH.DEBUG)
        if elem not in self.objects:
            raise Exception(f"Available objects: {self.objects.keys()}, {elem} not found")
        return self.objects[elem]


    def __getitem__(self, item: Union[str, Tuple[str, bool]]) -> Union[str, Any]:
        """__getitem__.

        Parameters
        ----------
        item : str
            item that should be returned

        Returns
        -------
        str

        """
        if isinstance(item, str):
            item = (item, False)
        assert len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], bool)
        return self.get(item[0], literal_eval=item[1])

    def check(self, key: str) -> bool:
        return key in self.objects.keys()

    def filter_keys(self, type: str) -> List[str]:
        res = []
        self.write_msg(f"Filter keys on {type} type", level=LH.DEBUG)
        for key, elem in self.objects.items():
            if elem.type == type:
                res.append(elem.name)
        self.write_msg(f"Elements found: {len(res)}", level=LH.DEBUG)
        return res

    def filter_handler(self, type: str) -> List[str]:
        res = []
        self.write_msg(f"Filter handler on {type} type", level=LH.DEBUG)
        for elem in self.objects.values():
            if elem.type == type:
                res.append(elem)
        self.write_msg(f"Elements found: {len(res)}", level=LH.DEBUG)
        return res

    def __str__(self) -> str:
        """__str__.
        Return an object in string format

        Returns
        -------
        str

        """

        res = ""
        for key in self.objects:
            res += f"{key}: {self.objects[key]}\n"
        return res

    @property
    def force_hash(self) -> Optional[str]:
        return self._force_hash

    @force_hash.setter
    def force_hash(self, hash: str) -> None:
        self._force_hash = hash

    @property
    def hash(self, digest_size: int = 8) -> str:
        if self.force_hash is not None:
            return self.force_hash
        return blake2b(str(self).encode('UTF-8'), digest_size=digest_size).hexdigest()
