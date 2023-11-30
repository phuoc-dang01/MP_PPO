# add queuing delay into halo
import os
import numpy as np
import core as abrenv
import load_trace

from collections.abc import Iterable
from const import *


class ABREnv:
    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        self.net_env = abrenv.Environment(
            all_cooked_time=all_cooked_time,
            all_cooked_bw=all_cooked_bw,
            random_seed=random_seed,
        )

        # self.last_action = DEFAULT_ACTION
        self.last_action = DEFAULT_ACTION
        self.buffer_size = 0.0
        self.state = np.zeros(S_DIM)

    def seed(self, num):
        np.random.seed(num)

    @staticmethod
    def replace_last_n_elements(arr, row_index, new_elements):
        # Check if new_elements is iterable (like a list or array), if not, make it a one-element list
        if not isinstance(new_elements, Iterable) or isinstance(new_elements, str):
            new_elements = [new_elements]
        n = len(new_elements)

        arr[row_index, -n:] = new_elements
        return arr

    @staticmethod
    def get_last_video_chunk_size(video_chunk_size, delay):
        return [float(_i) / float(delay) / M_IN_K for _i in video_chunk_size]

    def get_new_state_from_action(
        self, action, video_chunk_size, buffer, delay, video_chunk_remain
    ):
        state = self.state
        # Get the action details
        action_detail, _last_bit_rate = self.get_action_detail_bitrate(action)
        _last_video_chunk_size = self.get_last_video_chunk_size(video_chunk_size, delay)
        _last_video_chunk_remain = np.minimum(
            video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP
        ) / float(CHUNK_TIL_VIDEO_END_CAP)

        # Add on new information to the state
        state = self.replace_last_n_elements(state, 1, _last_bit_rate)  # last quality
        state = self.replace_last_n_elements(state, 0, buffer / BUFFER_NORM_FACTOR)  # 10 sec
        state = self.replace_last_n_elements(state, 2, _last_video_chunk_size)  # kilo byte / ms
        state = self.replace_last_n_elements(state, 3, float(delay) / M_IN_K / BUFFER_NORM_FACTOR)  # 10 sec
        state = self.replace_last_n_elements(state, 4, _last_video_chunk_remain)  # 10 sec
        
        self.state = state
        return self.state

    def reset(self):
        # self.net_env.reset_ptr()
        self.time_stamp = 0
        self.last_action = DEFAULT_ACTION
        self.state = np.zeros(S_DIM)
        self.buffer_size = 0.0
        action = self.last_action
        (
            delay,
            sleep_time,
            self.buffer_size,
            rebuf,
            video_chunk_size,
            end_of_video,
            video_chunk_remain,
        ) = self.net_env.get_video_chunk(action)

        # get the new state
        state = self.get_new_state_from_action(
            action, video_chunk_size, self.buffer_size, delay, video_chunk_remain
        )

        return state

    def render(self):
        return

    @staticmethod
    def get_action_detail_bitrate(action):
        # Get the action details
        action_detail = ACTION_TABLE[action]
        _last_bit_rate = [
            VIDEO_BIT_RATE[_i] / float(np.max(VIDEO_BIT_RATE))
            for _i in (action_detail[1], action_detail[3])
        ]
        return action_detail, _last_bit_rate

    @staticmethod
    def reward_bitrate(_last_bitrate):
        return sum([np.log(_i + 1) for _i in _last_bitrate])

    def penelty_smoothness(self, action, last_action):
        _, bit_rate = self.get_action_detail_bitrate(action)
        _, _last_bit_rate = self.get_action_detail_bitrate(last_action)
        return sum([i[0] - i[1] for i in zip(bit_rate, _last_bit_rate)])

    def step(self, action):
        # the action is from the last decision
        # this is to make the framework similar to the real
        (
            delay,
            sleep_time,
            self.buffer_size,
            rebuf,
            video_chunk_size,
            end_of_video,
            video_chunk_remain,
        ) = self.net_env.get_video_chunk(action)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        action_detail, _last_bit_rate = self.get_action_detail_bitrate(action)

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = (
            self.reward_bitrate(_last_bit_rate)
            - REBUF_PENALTY * rebuf
            - SMOOTH_PENALTY * self.penelty_smoothness(action, self.last_action)
        )

        self.last_action = action

        # get the new state
        state = self.get_new_state_from_action(
            action, video_chunk_size, self.buffer_size, delay, video_chunk_remain
        )
        return (
            state,
            reward,
            end_of_video,
            {"action": action, "rebuffer": rebuf},
        )
