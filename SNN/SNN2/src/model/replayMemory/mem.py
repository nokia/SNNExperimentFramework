# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
ReplayMemory module
===================

Class used to manage a replay memory
"""

import math
import numpy as np
from typing import Generator, Optional, Tuple, Union
import tensorflow as tf

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from SNN2.src.io.files import FileHandler
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.pickleHandler import PickleHandler as PkH
from SNN2.src.util.helper import dst2tensor

class ReplayMemory:

    def write_msg(self, msg: str, level: int = LH.INFO) -> None:
        """__write_msg.
        write a message into the log file with a defined log level

        parameters
        ----------
        msg : str
            msg to print
        level : int
            level default info

        returns
        -------
        none

        """
        if self.logger is not None:
            self.logger(f"{self.__class__.__name__}", f"{msg}", level=level)


    def __init__(self, data_spec: Tuple[tf.TensorSpec, ...],
                 max_length: int,
                 utilization: Optional[int] = None,
                 output_file: Optional[str] = None,
                 logger: Optional[LH] = None,
                 batch_size: int = 1):
        self.data_spec = data_spec
        self.max_length :int = max_length
        self.disable_unique_id = True
        self.utilization = utilization
        self.output_file = output_file
        self.logger = logger
        self.batch_size = batch_size

        self.memory = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                    self.data_spec,
                    self.batch_size,
                    max_length=self.max_length
                )

        self.rb_checkpointer: Optional[common.Checkpointer] = None
        if output_file is not None:
            self.rb_checkpointer: common.Checkpointer = common.Checkpointer(
                ckpt_dir=output_file,
                max_to_keep=1,
                replay_buffer=self.memory)
            self.rb_checkpointer.initialize_or_restore()

    def clear(self) -> None:
        self.write_msg(f"Memory clear!")
        self.memory.clear()

    def take(self, n: Optional[int] = None,
             batch_size: Optional[int] = 1):
        if len(self) == 0:
            return None

        n = self.utilization if n is None else n

        if n is None:
            raise Exception("A take dimension must be specified!")

        mem_dst = self.memory.as_dataset(sample_batch_size=batch_size)
        mem_shuffle = mem_dst.shuffle(len(self))
        n = len(self)/batch_size if len(self)/batch_size < n else n
        return mem_shuffle.take(math.ceil(n))

    def append(self, trajectories: Tuple[tf.Tensor, ...]):
        trajectory_len = trajectories[0].shape[0]
        assert all([trajectory_len == t.shape[0] for t in trajectories])
        self.write_msg(f"All the objects have a length of {trajectory_len}")

        for i in range(trajectory_len):
            trajectory = tf.nest.map_structure(lambda x: tf.convert_to_tensor(tf.gather(x, i).numpy()),
                                                trajectories)
            batch_trajectory = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0),
                                                     trajectory)
            self.memory.add_batch(batch_trajectory)
        self.write_msg(f"All trajectories has been appended to the memory")

    def dump(self, *args, **kwargs):
        if len(self) == 0:
            raise Exception("Cannot dump an empty memory")

        if self.rb_checkpointer is not None:
            self.rb_checkpointer.save(*args, **kwargs)
            self.write_msg(f"Memory saved on {self.output_file}")
        else:
            self.write_msg(f"Memory not save, no Checkpointer available")

    def __len__(self) -> int:
        if self.memory.num_frames() == -1:
            return 0
        return self.memory.num_frames()

    def __repr__(self) -> str:
        if self.memory is None:
            return "The memory is empty"
        return "Not implemented yet how to print the memory buffer with TF"
