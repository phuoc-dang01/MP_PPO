# add queuing delay into halo
import os
import numpy as np
import core as abrenv
import load_trace

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

        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        # TODO reset bit_rate to action
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(
            np.max(VIDEO_BIT_RATE)
        )  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = (
            np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        )  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
            CHUNK_TIL_VIDEO_END_CAP
        )
        self.state = state
        return state

    def render(self):
        return

    def step(self, action):
        # bit_rate = int(action) # remove, use the action instead

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

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = (
            VIDEO_BIT_RATE[bit_rate] / M_IN_K
            - REBUF_PENALTY * rebuf
            - SMOOTH_PENALTY
            * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[self.last_action])
            / M_IN_K
        )

        self.last_action = bit_rate
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(
            np.max(VIDEO_BIT_RATE)
        )  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = (
            np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        )  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
            CHUNK_TIL_VIDEO_END_CAP
        )

        self.state = state
        # observation, reward, done, info = env.step(action)
        return (
            state,
            reward,
            end_of_video,
            {"bitrate": VIDEO_BIT_RATE[bit_rate], "rebuffer": rebuf},
        )
