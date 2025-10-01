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

import functools
import inspect
from time import time
from typing import Callable, Dict, Any, Optional

from SNN2.src.io.logger import LogHandler

actions: Dict[str, Any] = {}
callbacks: Dict[str, Any] = {}
flows: Dict[str, Any] = {}
models: Dict[str, Any] = {}
grays: Dict[str, Any] = {}
plot_functions: Dict[str, Any] = {}
loss_functions: Dict[str, Any] = {}
loss_parameters: Dict[str, Any] = {}
embeddings: Dict[str, Any] = {}
layers: Dict[str, Any] = {}
fitMethods: Dict[str, Any] = {}
reward_functions: Dict[str, Any] = {}
perfEval_functions: Dict[str, Any] = {}
obsPP_functions: Dict[str, Any] = {}
actPolicy_functions: Dict[str, Any] = {}
envExit_functions: Dict[str, Any] = {}
metrics: Dict[str, Any] = {}
RLModelHandlers: Dict[str, Any] = {}
ModelManagers: Dict[str, Any] = {}
interpolate_functions: Dict[str, Any] = {}

def get_realName(obj: Any):
    if obj.__name__ == "augmented_cls":
        return obj.c_passed_name
    return obj.__qualname__

def collector(obj: Callable):
    @functools.wraps(obj)
    def wrapper(*args, **kwargs):
        return obj(*args, **kwargs)
    return wrapper

def action(action):
    actions[action.__qualname__] = action
    return collector(action)

def ccb(callback_class):
    name = get_realName(callback_class)
    callbacks[name] = callback_class
    return collector(callback_class)

def cflow(flow_class):
    name = get_realName(flow_class)
    flows[name] = flow_class
    return flow_class

def train_enhancement(gray_method):
    grays[gray_method.__qualname__] = gray_method
    return collector(gray_method)

def plot(plot_function):
    plot_functions[plot_function.__qualname__] = plot_function
    return collector(plot_function)

def loss(loss_function):
    loss_functions[loss_function.__qualname__] = loss_function
    return collector(loss_function)

def reward(reward_function):
    reward_functions[reward_function.__qualname__] = reward_function
    return collector(reward_function)

def reward_cls(reward_class):
    name = get_realName(reward_class)
    reward_functions[name] = reward_class
    return collector(reward_class)

def RLPerfEval(perfEval_function):
    perfEval_functions[perfEval_function.__qualname__] = perfEval_function
    return collector(perfEval_function)

def RLPerfEval_cls(perfEval_cls):
    name = get_realName(perfEval_cls)
    perfEval_functions[name] = perfEval_cls
    return collector(perfEval_cls)

def RLObservationPP(obsPP_function):
    obsPP_functions[obsPP_function.__qualname__] = obsPP_function
    return collector(obsPP_function)

def RLObservationPP_cls(obsPP_cls):
    name = get_realName(obsPP_cls)
    obsPP_functions[name] = obsPP_cls
    return collector(obsPP_cls)

def RLActionPolicies(actPolicy_function):
    name = get_realName(actPolicy_function)
    actPolicy_functions[name] = actPolicy_function
    return collector(actPolicy_function)

def RLEnvExitFunction(envExit_class):
    name = get_realName(envExit_class)
    envExit_functions[name] = envExit_class
    return collector(envExit_class)

def loss_param(loss_param):
    loss_parameters[loss_param.__qualname__] = loss_param
    return collector(loss_param)

def cmodel(model_class):
    def _wrp_cls(model_class):
        models[model_class.__qualname__] = model_class
        return model_class
    return _wrp_cls(model_class)

def cembedding(embedding_class):
    def _wrp_cls(embedding_class):
        embeddings[embedding_class.__qualname__] = embedding_class
        return embedding_class
    return _wrp_cls(embedding_class)

def clayer(layer_call):
    layers[layer_call.__qualname__] = layer_call
    return collector(layer_call)

def cclayer(layer_class):
    def _wrp_cls(layer_class):
        layers[layer_class.__qualname__] = layer_class
        return layer_class
    return _wrp_cls(layer_class)

def fitMethod(customFit):
    fitMethods[customFit.__qualname__] = customFit
    return collector(customFit)

def metric(mtr_cls):
    def _wrp_cls(mtr_cls):
        metrics[mtr_cls.__qualname__] = mtr_cls
        return mtr_cls
    return _wrp_cls(mtr_cls)

def RLModelHandlerWrapper(RLModelHandler_class):
    def _wrp_cls(RLModelHandler_class):
        RLModelHandlers[RLModelHandler_class.__qualname__] = RLModelHandler_class
        return RLModelHandler_class
    return _wrp_cls(RLModelHandler_class)

def ModelManager(ModelManager_cls):
    def _wrp_cls(ModelManager_cls):
        ModelManagers[ModelManager_cls.__qualname__] = ModelManager_cls
        return ModelManager_cls
    return _wrp_cls(ModelManager_cls)

def interpolator(interpolate_fn):
    interpolate_functions[interpolate_fn.__qualname__] = interpolate_fn
    return collector(interpolate_functions)


def c_logger(c_passed):
    class augmented_cls(c_passed):
        c_passed_name = c_passed.__qualname__

        def __write_msg(self, msg: str, level: int = LogHandler.INFO):
            if self.logger is None:
                return
            self.logger(f"{c_passed.__qualname__}", f"{msg}", level)

        def __dummy_log(self, *args, **kwargs):
            pass

        def __init__(self, *args, logger: Optional[LogHandler] = None, **kwargs):
            self.logger = logger
            self.write_msg = self.__write_msg
            super().__init__(*args, **kwargs)

    return augmented_cls

def f_logger(f_passed):
    oldsig = inspect.signature(f_passed)
    params = list(oldsig.parameters.values())

    # def add_kwarg_assignment(name, f, idx=0):
    #     previous_code = f.__code__

    #     source_code = inspect.getsource(f)
    #     tree = ast.parse(source_code)
    #     print("----------------------")
    #     print(ast.dump(tree))

    #     comment = None
    #     slice=ast.Index(ast.Constant(name, None))
    #     value=ast.Subscript(ast.Name("kwargs", ast.Load()), slice, ast.Load())
    #     targets = [ast.Name(name, ast.Store())]
    #     log_assign = ast.Assign(targets, value, comment)
    #     tree.body[0].body.insert(idx, log_assign)
    #     ast.fix_missing_locations(tree)

    #     new_code_obj = compile(tree, previous_code.co_filename, 'exec')
    #     print(new_code_obj.co_consts)
    #     f.__code__ = new_code_obj.co_consts[-2]
    #     source_code = inspect.getsource(f)
    #     tree = ast.parse(source_code)
    #     print("----------------------")
    #     print(ast.dump(tree))
    #     # globals = f.__globals__
    #     # f.__code__.replace(new_code_obj)
    #     # exec_scope = {}

    #     # exec(new_code_obj, globals, exec_scope)
    #     # print("--------- exec scope -----------")
    #     # print(exec_scope)
    #     return f
    #     # new_function = FunctionType(new_code_obj.co_consts[-2], f.__globals__)

    def __dummy_log(self, *args, **kwargs):
        pass

    def add_kwarg(name, default, f=None):
        if f is None:
            f = f_passed

        if name in oldsig.parameters:
            raise Exception(f"logger parameter already present in {f_passed.__qualname__} signature")

        position = len(params)
        for i, param in enumerate(params):
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                position = i
                break

        newparam = inspect.Parameter(name,
                                     inspect.Parameter.KEYWORD_ONLY,
                                     default = default)
        params.insert(position, newparam)
        # new_f = add_kwarg_assignment(name, f=f)
        # return new_f

    add_kwarg("logger", None)
    # print(astor.to_source(inspect.getsource(new_f)))
    add_kwarg("write_msg", __dummy_log)
    # print(astor.to_source(inspect.getsource(new_f)))

    sig = oldsig.replace(parameters = params)

    @functools.wraps(f_passed)
    def wrapper(*args, logger: Optional[LogHandler] = None, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        def __write_msg(msg: str, level: int = LogHandler.INFO):
            if logger is None:
                return
            logger(f"{f_passed.__qualname__}", f"{msg}", level)

        bound.arguments["logger"] = logger
        bound.arguments["write_msg"] = __write_msg
        return f_passed(*bound.args, **bound.kwargs)

    wrapper.__signature__ = sig
    return wrapper

def timeit(func):

    @functools.wraps(func)
    @f_logger
    def new_func(*args, **kwargs):
        logger, write_msg = kwargs["logger"], kwargs["write_msg"]
        del kwargs["logger"]
        del kwargs["write_msg"]

        start_time = time()
        res = func(*args, **kwargs)
        delta_t = time() - start_time
        write_msg(f"Function {func.__name__} finished in {int(delta_t*1000)} ms", LogHandler.DEBUG)
        # print(f"Function {func.__name__} finished in {int(delta_t*1000)} ms")
        return res
    return new_func

