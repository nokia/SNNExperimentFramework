# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from SNN2.src.model.callbacks.callbacksWrapper import Callback_Selector
from SNN2.src.model.replayMemory.mem import ReplayMemory
from SNN2.src.model.reward.accuracyMap import find_zone, weightAccuracy
from SNN2.src.util.dataManger import DataManager

from tensorflow.keras import optimizers
from tensorflow.keras import Model

from SNN2.src.util.strings import s
from SNN2.src.io.logger import LogHandler as LH

from SNN2.src.decorators.decorators import RLModelHandlerWrapper
from SNN2.src.model.reinforcement.reinforcement import RLModelHandler

@RLModelHandlerWrapper
class RL_AC_manager_intermidiate(RLModelHandler):

    def __define_replay_mem(self) -> ReplayMemory:
        data_spec: Tuple[tf.TensorSpec, ...] = (
                    tf.TensorSpec(shape=[self.observation_dim], dtype=tf.float64, name='Observation'),
                    tf.TensorSpec(shape=[1], dtype=tf.float16, name='Action'),
                    tf.TensorSpec(shape=[1], dtype=tf.float32, name='Reward')
                )
        output_pth = self.mem_output_dir.path if not self.fresh_mem_only else None

        memory = ReplayMemory(data_spec,
                              ast.literal_eval(self.params["memory_max_dimension"]),
                              utilization=ast.literal_eval(self.params["memory_utilization"]),
                              output_file=output_pth,
                              logger=self.logger)
        return memory

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.qoe_threshold = self.params["qoe_threshold", True]
        self.gamma = self.params["gamma", True]
        self.entropy_beta = self.params["entropy_beta", True]
        self.training_epochs = self.params["training_epochs", True]
        self.exploration_p = self.params["exploration_p", True]
        self.exploration_decay = self.params["exploration_decay", True]
        self.train_batch_size = self.params["train_batch_size", True]
        self.observation_dim = self.params["state_dim", True]
        self.actor_csv: str = self.params["actor_csv_output"]
        self.fresh_mem_only = self.params["fresh_mem_only", True]
        self.use_stop_thresholds = self.params["stop_thresholds", True]
        self.action_interleave = self.params["action_interleave", True]
        self.adam_strength = self.params["ReinforcementAdamStrength", True]
        self.adam_epsilon = self.params["ReinforcementAdamEpsilon", True]
        self.adam_global_clipnorm = self.params["ReinforcementAdamClipNorm", True]
        self.interruption_penalty = self.params["InterruptionPenalty", True]
        self.model_definition: Model = self.define_model(None, self.emb)
        self.load()
        self.write_msg("Reinforcement model initialized", level=LH.INFO)

        # Compile the network
        self.model = self.compile(self.params["ReinforceCustomModel"],
                                  self.gamma, self.entropy_beta,
                                  optimizer=optimizers.Adam(self.adam_strength,
                                                            epsilon=self.adam_epsilon,
                                                            global_clipnorm=self.adam_global_clipnorm),
                                  weighted_metrics=[])
        self.write_msg("Reinforcement model compiled")

        self.memory = self.__define_replay_mem()

        self.df: DataManager = DataManager(["Episode", "Step", "State", "Statistic", "Value"])
        self.current_training_epoch = 0
        self.training = True
        self.previous_state: Optional[tf.Tensor] = None
        self.current_step: int = 0
        self.current_episode: int = 0
        self.expected_qoe = self.pp.validation_triplets.dft("Targets")
        self.exp_flags = self.pp.validation_triplets.dft("ExpectedLabel")
        self.i = 0
        self.action_policy = self.actPolicyH.get_handler(self.params["ActionPolicy"])(
                    model=self.model, register=self.register, logger=self.logger)
        # self.exp_flags = tf.cast(tf.reshape(tf.where(self.expected_qoe > self.qoe_threshold, 0, 1), [-1]), tf.int8)

    def register(self, stat: str, value: Any, step: Optional[int] = None) -> None:
        if step is None:
            step = self.current_step
        np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)
        update = {"Episode": self.current_episode,
                  "Step": step,
                  "State": f"{self.previous_state.numpy()}",
                  "Statistic": stat,
                  "Value": value}
        self.write_msg(f"Update the df with {update}", level=LH.DEBUG)
        np.set_printoptions()
        self.df.append(update)

    def update_memory(self) -> None:
        self.write_msg("Updating the memory!")
        self.log_memory_state()
        self.log_history_state()

        trajectories = (self.history['state'].history,
                        self.history['action'].history,
                        self.history['reward'].history)
        self.write_msg(f"Trajectories: {trajectories}", level=LH.DEBUG)

        self.memory.append(trajectories)
        self.memory.dump(tf.constant(self.current_episode*self.MAX_STEPS + self.current_step))
        self.write_msg(f"The memory has been updated with the last steps, current size of the memory: {len(self.memory)}")

    @tf.function
    def aggregate(self, agg, x):
        return self.gamma*agg + x

    def discounted_sum(self, x: tf.Tensor) -> tf.Tensor:
        self.write_msg(f"Requiring discounted sum for {x}", level=LH.DEBUG)
        return tf.scan(self.aggregate, x, reverse=True)

    def calculate_confusion_matrix(self, labels: tf.Tensor, exp_l: Optional[tf.Tensor] = None) -> Dict[str, int]:
        exp_l = self.exp_flags if exp_l is None else exp_l
        correct_l, wrong_l = self.get_correct_wrong(labels, exp_l)
        tp=  tf.size(tf.where(correct_l == 0)).numpy()
        fp = tf.size(tf.where(wrong_l == 0)).numpy()
        tn = tf.size(tf.where(correct_l == 1)).numpy()
        fn = tf.size(tf.where(wrong_l == 1)).numpy()
        u = tf.size(tf.where(wrong_l == 2)).numpy()
        matrix = {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "U": u}
        self.write_msg(f"Confusion matrix computed: {matrix}")
        return matrix

    def update_reward(self, labels: tf.Tensor,
                      conf_matrix: Optional[Dict[str, int]] = None,
                      current_params: Optional[tf.Tensor] = None,
                      previous_params: Optional[tf.Tensor] = None,) -> None:
        self.write_msg(f"Reward function")
        cf_matrix = self.calculate_confusion_matrix(labels) if conf_matrix is None else conf_matrix

        self.reward.update(cf_matrix)
        current_reward_state = self.reward.state()
        self.write_msg(f"Reward state: {current_reward_state}")

    def compute_reward(self, game_over: bool = False,
                       interrupted: bool = False) -> tf.Tensor:
        self.write_msg(f"Reward function")

        previous_comAccuracy = 0.0
        if self.history["previous_accuracy"].history is not None:
            previous_comAccuracy = round(float(self.history["previous_accuracy"].history[-1].numpy()), 4)
            self.write_msg(f"Previous reward: {previous_comAccuracy}")

        reward, comulativeAccuracy = self.reward.compute(previous_comAccuracy=previous_comAccuracy)
        self.write_msg(f"Reward obtained: {reward}")
        if game_over:
            self.write_msg(f"Game over state, reward must be 0")
            reward = 0.0
            if interrupted:
                self.write_msg(f"Interrupted, the reward will be equal to the penalty {self.interruption_penalty}")
                reward = self.interruption_penalty

        self.register("RewardComAccuracy", comulativeAccuracy, step=self.current_step-1)
        self.register("PreviousComAccuracy", previous_comAccuracy, step=self.current_step-1)
        self.register("Reward", reward, step=self.current_step-1)
        # TODO register also all the accuracy accumulated and the confusion matrix
        self.write_msg(f"Reward after comparison: {reward}")
        self.write_msg(f"Comulative accuracy: {comulativeAccuracy}")

        tf_reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        tf_accuracy = tf.convert_to_tensor([comulativeAccuracy], dtype=tf.float32)
        self.history["previous_accuracy"].update(tf.expand_dims(tf_accuracy, 0))
        self.history["reward"].update(tf.expand_dims(tf_reward, 0))
        self.write_msg(f"Current reward history: {self.history['reward'].history.numpy().flatten()}")
        self.write_msg(f"len reward history: {len(self.history['reward'].history.numpy().flatten())}")

        self.write_msg(f"Reward obtained: {reward}")
        self.write_msg(f"Tensor reward: {tf_reward}", level=LH.DEBUG)
        for key, item in self.reward.state().items():
            self.register(key, item, step=self.current_step-1)
        # self.register("Accuracy", (cf_matrix["TP"] + cf_matrix["TN"])/sum(cf_matrix.values()), step=self.current_step-1)
        # self.register("TP", cf_matrix["TP"], step=self.current_step-1)
        # self.register("FP", cf_matrix["FP"], step=self.current_step-1)
        # self.register("TN", cf_matrix["TN"], step=self.current_step-1)
        # self.register("FN", cf_matrix["FN"], step=self.current_step-1)
        # self.register("U", cf_matrix["U"], step=self.current_step-1)
        self.reward.reset()
        return tf_reward

    def step(self, observation: tf.Tensor,
             labels: tf.Tensor,
             game_over: bool,
             cycle: Optional[int] = None,
             interrupted: bool = False) -> Optional[int]:
        self.write_msg(f"Step function, step: {self.current_step}")
        cycle = self.current_step if cycle is None else cycle
        observation, cm_matrix = self.observation_preproc(observation,
                                                          self.get_correct_wrong(labels, self.exp_flags),
                                                          current_step=self.current_step,
                                                          logger=self.logger)

        if self.previous_state is not None:
            self.update_reward(labels, conf_matrix=cm_matrix, current_params=observation[0], previous_params=self.previous_state[0])
            if cycle % self.action_interleave != 0 and not game_over:
                self.write_msg(f"In between actions")
                return None

            self.write_msg(f"The previous state {self.previous_state} is not None")
            reward = self.compute_reward(game_over=game_over, interrupted=interrupted)
            # reward = self.calculate_reward(labels, current_params=observation[0], previous_params=self.previous_state[0])

        if game_over:
            self.evaluate_performances()
            self.write_msg(f"reward history: {self.history['reward'].history.numpy()}", level=LH.DEBUG)
            self.history["reward"].history = self.discounted_sum(self.history["reward"].history)
            self.write_msg(f"Discounted sum of reward: {self.history['reward'].history.numpy()}", level=LH.DEBUG)
            self.df.save(self.actor_csv)
            self.update_memory()
            self.train()
            self.reset()
            self.current_episode += 1
            self.write_msg(f"Reset applied, moving to episode {self.current_episode} step {self.current_step}")
            return None

        self.previous_state = observation

        np_val, tf_val = self.action_policy(observation, current_step=self.current_step)

        action = self.np_rng.choice(self.actions, p=np_val[0])
        tf_action = tf.expand_dims(tf.convert_to_tensor([float(action)], dtype=tf.float16), 0)

        self.history["state"].update(tf.expand_dims(observation, 0))
        self.history["action"].update(tf_action)
        self.history["action_probs"].update(tf_val[0])

        self.write_msg(f"Action probs: {np_val[0]}")
        self.write_msg(f"Critic value: {np_val[1]}")
        self.write_msg(f"Action chosen: {action}")
        self.log_memory_state()

        self.register("Action", action)
        self.register("ActionProbs", np_val[0])
        self.register("CriticValue", np_val[1])

        self.current_step += 1
        return action

    def execute_train(self, *args, **kwargs) -> None:
        self.write_msg(f"Executing a training cycle on the RL Network, mem dimension: {len(self.memory)}")
        if len(self.memory) == 0:
            self.write_msg("The memory is empty, no training is possible")
            return

        examples = self.memory.take(batch_size=self.train_batch_size)
        csv_logger = Callback_Selector("csvLogger", self.csv_output, append=True)
        self.model.fit(examples, *args, callbacks=[csv_logger], verbose=0, **kwargs)

        if self.fresh_mem_only:
            self.memory.clear()

    def get_margin_values(self, *args, **kwargs) -> tf.Tensor:
        margin = self.history["state"].history[:, 0]
        return margin

    def get_probabilities_values(self, *args, **kwargs) -> tf.Tensor:
        probabilities = self.history["action_probs"].history
        return probabilities

    def evaluate_performances(self, *args, **kwargs) -> None:
        if not self.training or not self.use_stop_thresholds:
            return

        if self.perfEval_function(self.get_margin_values(),
                                  self.get_probabilities_values(),
                                  current_episode=self.current_episode,
                                  reward_history=self.history["reward"].history,
                                  logger=self.logger):
            self.training = False

    def train(self, *args, **kwargs) -> None:
        if not self.training:
            self.write_msg("The training option is deactivated")
            return
        if self.trained:
            self.write_msg("The model has already been previously trained and loaded")
            return

        self.execute_train(*args, epochs=self.current_training_epoch+self.training_epochs,
                           initial_epoch=self.current_training_epoch,
                           **kwargs)
        self.current_training_epoch += self.training_epochs

    def reset(self) -> None:
        self.previous_state = None
        self.current_step = 0
        [h.reset() for h in self.history.values()]
        self.observation_preproc.reset()
