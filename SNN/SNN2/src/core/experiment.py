# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
import os
from logging import exception
import numpy as np
import tensorflow as tf
import pandas as pd
from SNN2.src.decorators.decorators import c_logger
from SNN2.src.model.callbacks.CallbacksHelper import CallbackHelper as CBH
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

from tensorflow.keras.callbacks import Callback

from SNN2.src.util.strings import s
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.files import FileHandler as FH
from SNN2.src.params.paramHandler import ParamHandler as PH
from SNN2.src.io.pickleHandler import PickleHandler as PkH
from SNN2.src.core.data.PreProcessing import PreProcessing as PP
from SNN2.src.model.modelHandler import DefaultModelHandler as MH
from SNN2.src.core.gray.grayWrapper import flag_windows
from SNN2.src.util.helper import dst2tensor
from SNN2.src.actions.actionWrapper import action_selector as AS
from SNN2.src.model.reinforcement.reinfoceModelHandler import ReinforceModelHandler as RLMH
from SNN2.src.io.progressBar import pb

from typing import Callable, List, Optional, Tuple, Union


@c_logger
class ExperimentEnv:

    def __init__(self,
                 pp: PP,
                 model: MH,
                 params: PH,
                 environment: PH,
                 callbacks: PH,
                 fitMethods: PH,
                 actions: PH,
                 grayEvolution: PH,
                 numpyRNG: PH,
                 EnvExitFunctionH: PH,
                 ph: PkH,
                 rl_model: Optional[RLMH] = None,
                 model_extension_name: Optional[str] = None):
        # Basic objects to generate an experiment
        self.ph: PkH = ph
        self.params: PH = params
        self.data: PP = pp
        self.model: MH = model
        self.env: PH = environment
        self.callbacksH: PH = callbacks
        self.fitMethods: PH = fitMethods
        self.actions: PH = actions
        self.grayEvolution: PH = grayEvolution
        self.rl_model: RLMH = rl_model
        self.EnvExitFunctionH = EnvExitFunctionH
        self.numpyRNG = numpyRNG["numpy_rng"]
        self.required_callbacks = ast.literal_eval(self.params["callbacks"])
        self.categorical = self.params["categorical", True]
        self.model_extension_name = model_extension_name

        self.contrastive = False

        # self.write_msg(f"Available keys: {self.data.training_triplets.keys()}")

        if self.categorical:
            self.train_data = self.data.training["SamplesDst"]["TfDataset"]
            self.validation_data = self.data.validation["SamplesDst"]["TfDataset"]
            self.test_data = self.data.test["SamplesDst"]["TfDataset"]
        else:
            self.write_msg(type(self.data))
            self.train_data = self.data.training_triplets['TripletDst']['TfDataset']
            self.validation_data = self.data.validation_triplets["TripletDst"]["TfDataset"]
            self.test_data = self.data.test_triplets["TripletDst"]["TfDataset"]
            self.grays_data = self.data.gray_triplets["TripletDst"]["TfDataset"]


        self.prepare_data()

        self.cbh = CBH(self.required_callbacks,
                       self.callbacksH,
                       self.data,
                       logger=self.logger,
                       test_data = self.test_data,
                       validation_data = self.validation_data,
                       snn_model = self.model,
                       ph = self.ph,
                       rl_model = self.rl_model,
                       EnvExitFunctionH = self.EnvExitFunctionH,
                       model_extra_label=self.model_extension_name)

    def prepare_data(self) -> None:
        if not self.categorical and not self.contrastive:
            self.train_data = self.train_data.shuffle(50000)
            self.grays_data = self.grays_data.batch(int(self.params[s.batch_size_key]))
        self.train_data = self.train_data.batch(int(self.params[s.batch_size_key]))
        self.train_data = self.train_data.prefetch(tf.data.AUTOTUNE)
        self.validation_data = self.validation_data.batch(int(self.params[s.batch_size_key]))
        self.validation_data = self.validation_data.prefetch(tf.data.AUTOTUNE)
        self.test_data = self.test_data.batch(int(self.params[s.batch_size_key]))
        self.test_data = self.test_data.prefetch(tf.data.AUTOTUNE)

    def get_difficult_args(self) -> Union[Callable, None]:
        if self.params[s.fitMethod] not in s.all_grays:
            return None

        self.write_msg("A gray method has been defined")
        self.data.gray_triplets.log_dump("Difficult triplets for the SL cycle")
        return self.grayEvolution.get_handler(self.params[s.grayEvMethod])(
                            self.model,
                            self.data.gray_triplets,
                            self.data.goods_prop,
                            self.data.bads_prop,
                            self.data.training_triplets,
                            self.data.gray_out_train,
                            self.actions.get_handler("GenerateGrayTriplets"),
                            ph=self.ph,
                            logger = self.logger)

    def fit(self, *args, **kwargs) -> None:
        # print(f"Model trained: {self.model.trained}")
        # print(f"Model extension name: {self.model_extension_name}")
        if self.model.trained and self.model_extension_name is None:
            print("Model already trained")
            # raise Exception("Model already trained")
            return
        elif self.model.trained and self.model_extension_name is not None:
            print(f"\nMODEL ALREADY TRAINED CHECK DEACTIVATED\nModel" \
                   "extension {self.model_extension_name}\n")
        # raise Exception("Retraining?")
        self.model.trained = False

        callbacks = self.cbh.get_callbacks()

        self.write_msg("Model not trained")
        fit_args = [self.model, callbacks, self.train_data]

        difficult_args = self.get_difficult_args()
        if difficult_args is not None:
            fit_args.append(difficult_args)

        if self.params[s.fitMethod] in s.rl_training:
            self.write_msg("A RL training method has been defined")
            fit_args.append(self.rl_model)

        self.write_msg("Starting training")
        # with tf.device("gpu:0"):
        # with tf.device("cpu:0"):

        # history = self.model.model.fit(self.train_data,
        #                          validation_data=self.validation_data,
        #                          batch_size=16,
        #                          epochs=1)
        history = self.fitMethods.get_handler(self.params[s.fitMethod])(
                        *fit_args,
                        epochs=int(self.params[s.epochs_key]),
                        validation_data=self.validation_data,
                        logger=self.logger
                     )

        self.model.trained = True
        self.model.save(extend=self.model_extension_name)
        if self.rl_model is not None and self.rl_model.trained == False:
            self.rl_model.trained = True
            # print(self.rl_model.emb.summary())
            self.rl_model.save(obj=self.rl_model.emb)

    @classmethod
    def emb_inference(cls, model,
                      dataset,
                      label,
                      ph,
                      batch_size: int = 5000) -> None:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # embs = [model.emb(s) for s in dataset]
        embs = None
        pbar = pb.bar(total=dataset.cardinality().numpy())
        for s in dataset:
            model.emb(s)
            # if embs is None:
            #     embs = model.emb(s)
            # else:
            #     embs = tf.concat([embs, model.emb(s)], axis=0)
            pbar.update(1)
        pbar.close()
        # ph.save(embs, f"embedding_{label}", unix_time=True)

    @classmethod
    def predict(cls, model, dataset, batch_size:int = 5000, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        ap, an = model.model.predict(dataset)
        return ap, an

    def evaluate(self, *args, save_output: bool = False, **kwargs):
        callbacks = self.cbh.get_callbacks(caller="Evaluate")
        result = self.model.model.evaluate(self.test_data, callbacks=callbacks, return_dict=True)
        self.write_msg(f"evaluation result: {result}")
        if save_output:
            test_df = pd.DataFrame({
                    "Statistic": result.keys(),
                    "Value": result.values()
                })
            AS("write", self.params["EvaluationOutput"], df=test_df, index=False)
        return result
