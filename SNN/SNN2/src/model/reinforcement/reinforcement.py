# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
from typing import Dict, Tuple

import tensorflow as tf
from SNN2.src.core.data.PreProcessing import PreProcessing
from SNN2.src.io.directories import DirectoryHandler

from SNN2.src.model.history.history import History
from SNN2.src.model.modelHandler import DefaultModelHandler
from SNN2.src.model.replayMemory.mem import ReplayMemory
from SNN2.src.params.paramHandler import ParamHandler as PH
from SNN2.src.util.strings import s
from SNN2.src.io.logger import LogHandler as LH

class RLModelHandler(DefaultModelHandler):

    def __init__(self, pp: PreProcessing,
                 rewardFunction: PH,
                 RLPerfEvaluation: PH,
                 ObsPP: PH,
                 ActPolicy: PH,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pp = pp
        self.rewardFunctionsH: PH = rewardFunction
        self.perfEvaluationH: PH = RLPerfEvaluation
        self.observationPPH: PH = ObsPP
        self.actPolicyH: PH = ActPolicy
        self.csv_output: str = self.params["ReinforcementCSVoutput"]
        self.required_metrics = ast.literal_eval(self.params["ReinforcementMetrics"])
        self.required_losses = ast.literal_eval(self.params["ReinforcementLosses"])
        self.loss_parameters = ast.literal_eval(self.params["ReinforcementLossParameters"])
        self.output_weights = self.params["reinforcement_weights_file"]
        self.MAX_STEPS = ast.literal_eval(self.params["max_steps"])
        self.actions = ast.literal_eval(self.params["reinforce_actions"])
        self.reward_cls_flag = ast.literal_eval(self.params["reward_class_flag"])
        if not self.reward_cls_flag:
            self.reward_function = self.rewardFunctionsH.get_handler(self.params["RewardFunction"])
        else:
            self.reward = self.rewardFunctionsH.get_handler(self.params["RewardFunction"])(logger=self.logger)
        self.perfEval_cls_flag = ast.literal_eval(self.params["perfEval_class_flag"])
        if not self.perfEval_cls_flag:
            self.perfEval_function = self.perfEvaluationH.get_handler(self.params["PerfEval"])
        else:
            self.perfEval_function = self.perfEvaluationH.get_handler(self.params["PerfEval"])(logger=self.logger)
        self.obsPp_cls_flag = ast.literal_eval(self.params["ObsPP_class_flag"])
        if not self.obsPp_cls_flag:
            self.observation_preproc = self.observationPPH.get_handler(self.params["ObservationPreProcessing"])
        else:
            self.observation_preproc = self.observationPPH.get_handler(self.params["ObservationPreProcessing"])(
                        logger=self.logger
                    )
        self.action_policy = self.actPolicyH.get_handler(self.params["ActionPolicy"])

        self.history: Dict[str, History] = {
                    "state": History(logger=self.logger),
                    "reward": History(logger=self.logger),
                    "action": History(logger=self.logger),
                    "previous_accuracy": History(logger=self.logger),
                    "action_probs": History(logger=self.logger),
                }

        self.mem_output_dir = DirectoryHandler(self.params["memory_output_folder"])

        layers_names = ast.literal_eval(self.params["reinforcementLayers"])
        layers = [self.layers.get_handler(l)() for l in layers_names]
        self.emb = self.embeddings.get_handler(self.params["reinforcement_embedding"])(
                    *layers
                ).layers

    def get_correct_wrong(self, obtained: tf.Tensor, expected: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        correct = tf.reshape(tf.gather(obtained, tf.where(obtained == expected)), [-1])
        wrong = tf.reshape(tf.gather(obtained, tf.where(obtained != expected)), [-1])
        return correct, wrong

    def log_memory_state(self) -> None:
        self.write_msg(f"Current shapes of histories: state {self.history['state'].shape} action {self.history['action'].shape} reward {self.history['reward'].shape}")

    def log_history_state(self) -> None:
        for key in self.history:
            self.write_msg(f"{key} history: {self.history[key].history}, {self.history[key].shape}", level=LH.DEBUG)

    def calculate_reward(self, *args, **kwargs) -> int:
        raise NotImplementedError("Calculate reward must be implemented by the child object")

    def get_action(self, *args, **kwargs) -> int:
        raise NotImplementedError("Get action must be implemented by the child object")

    def execute_episode(self, *args, **kwargs) -> int:
        raise NotImplementedError("Execute episode must be implemented by the child object")

