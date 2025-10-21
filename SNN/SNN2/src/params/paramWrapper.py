# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import functools

from SNN2.src.params.parameters import _param
from SNN2.src.params.parameters import TypeSpecific_param
from SNN2.src.params.parameters import Callable_param
from SNN2.src.params.parameters import NumpyRng_param
from SNN2.src.actions.actionWrapper import action_selector as AS
from SNN2.src.model.embeddings.embeddingsWrapper import Embedding_selector as ES
from SNN2.src.model.layers.layersWrapper import Layer_selector as LS
from SNN2.src.model.metrics.metricsWrapper import Metrics_selector as MS
from SNN2.src.model.losses.loss import fn as LW
from SNN2.src.model.callbacks.callbacksWrapper import Callback_Selector as CS
from SNN2.src.core.data.flow.aggregator import Flow_Selector as FlwS
from SNN2.src.core.gray.grayWrapper import Gray_Selector as GS
from SNN2.src.model.fit.fitWrapper import Fit_Selector as FS
from SNN2.src.model.losses.parameters import fn as LP
from SNN2.src.model.reward.reward import selector as RF
from SNN2.src.model.RLPerfEvaluation.perfEval import selector as PerfEval
from SNN2.src.model.RLObservationPP.obsPP import selector as obsPP
from SNN2.src.model.RLActionPolicies.actPolicy import selector as actP
from SNN2.src.model.RLEnvExitFunctions.exitFunctions import selector as EnvExitFunct

from SNN2.src.util.strings import s

classes = {}

def cls_wrapper(obj):
    """fn_wrapper.

    Parameters
    ----------
    func : Callable
        func
    """
    classes[obj.__qualname__] = obj
    @functools.wraps(obj)
    def wrapper(*args, **kwargs):
        return obj(*args, **kwargs)
    return wrapper

@cls_wrapper
def generic(*args, **kwargs) -> _param:
    return _param(*args, **kwargs)

@cls_wrapper
def preprocessing(*args, **kwargs) -> TypeSpecific_param:
    return TypeSpecific_param(*args, param_type=s.param_PreProcessing_type, **kwargs)

@cls_wrapper
def environment(*args, **kwargs) -> TypeSpecific_param:
    return TypeSpecific_param(*args, param_type=s.param_Environment_type, **kwargs)

@cls_wrapper
def numpyRng(*args, **kwargs) -> NumpyRng_param:
    return NumpyRng_param(*args, **kwargs)

@cls_wrapper
def action(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=AS, param_type=s.param_action_type, **kwargs)

@cls_wrapper
def experiment(*args, **kwargs) -> TypeSpecific_param:
    return TypeSpecific_param(*args, param_type=s.param_experiment_type, **kwargs)

@cls_wrapper
def model(*args, **kwargs) -> TypeSpecific_param:
    return TypeSpecific_param(*args, param_type=s.param_model_type, **kwargs)

@cls_wrapper
def reinforcementModel(*args, **kwargs) -> TypeSpecific_param:
    return TypeSpecific_param(*args, param_type=s.param_reinforce_model_type, **kwargs)

@cls_wrapper
def embedding(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=ES, param_type=s.param_embedding_type, **kwargs)

@cls_wrapper
def layer(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=LS, param_type=s.param_layer_type, **kwargs)

@cls_wrapper
def metric(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=MS, param_type=s.param_metric_type, **kwargs)

@cls_wrapper
def lossParam(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=LP, param_type=s.param_lossParam_type, **kwargs)

@cls_wrapper
def loss(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=LW, param_type=s.param_loss_type, **kwargs)

@cls_wrapper
def callback(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=CS, param_type=s.param_callback_type, **kwargs)

@cls_wrapper
def flow(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=FlwS, param_type=s.param_flow_type, **kwargs)

@cls_wrapper
def fitMethod(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=FS, param_type=s.param_fitmethod_type, **kwargs)

@cls_wrapper
def study(*args, **kwargs) -> TypeSpecific_param:
    return TypeSpecific_param(*args, param_type=s.param_study_type, **kwargs)

@cls_wrapper
def grayEvolution(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=GS, param_type=s.param_grayEvolution_type, **kwargs)

@cls_wrapper
def rewardFunction(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=RF, param_type=s.param_reward_function_type, **kwargs)

@cls_wrapper
def RLPerfEval(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=PerfEval, param_type=s.param_RLPerfEval_type, **kwargs)

@cls_wrapper
def RLObservationPP(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=obsPP, param_type=s.param_RLObsPP_type, **kwargs)

@cls_wrapper
def RLActionPolicy(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=actP, param_type=s.param_RLActPolicy_type, **kwargs)

@cls_wrapper
def RLEnvExitFunction(*args, **kwargs) -> Callable_param:
    return Callable_param(*args, function=EnvExitFunct, param_type=s.param_RLEnvExitFunction_type, **kwargs)

def Param_selector(obj, *args, **kwargs):
    if obj in classes.keys():
        return classes[obj](*args, **kwargs)

