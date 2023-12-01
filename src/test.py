import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow.compat.v1 as tf
import load_trace

# add queuing delay into halo
from collections.abc import Iterable

# import a2c as network
import ppo2 as network
import fixed_env as env

from const import *

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")


# tf.disable_v2_behavior()

# os.chdir("./src/")

# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = sys.argv[1]
# NN_MODEL = "./pretrain/nn_model_ep_151200.ckpt"


def get_action_detail_bitrate(action):
    # Get the action details
    action_detail = ACTION_TABLE[action]
    _last_bit_rate = [
        VIDEO_BIT_RATE[_i] / float(np.max(VIDEO_BIT_RATE))
        for _i in (action_detail[1], action_detail[3])
    ]
    return action_detail, _last_bit_rate

def reward_bitrate(_last_bitrate):
    return sum([np.log(_i) for _i in _last_bitrate])

def penelty_smoothness(action, last_action):
    _, bit_rate = get_action_detail_bitrate(action)
    _, _last_bit_rate = get_action_detail_bitrate(last_action)
    return abs(sum([i[0] - i[1] for i in zip(bit_rate, _last_bit_rate)]))

def get_last_video_chunk_size(video_chunk_size, delay):
    return [float(_i) / float(delay) / M_IN_K for _i in video_chunk_size]

def replace_last_n_elements(arr, row_index, new_elements):
    # Check if new_elements is iterable (like a list or array), if not, make it a one-element list
    if not isinstance(new_elements, Iterable) or isinstance(new_elements, str):
        new_elements = [new_elements]
    n = len(new_elements)

    arr[row_index, -n:] = new_elements
    return arr

def get_new_state_from_action(
    input_state, action, buffer, video_chunk_size, delay, video_chunk_remain
):
    state = input_state
    # Get the action details
    action_detail, _last_bit_rate = get_action_detail_bitrate(action)

    _last_video_chunk_size = get_last_video_chunk_size(video_chunk_size, delay)
    _last_video_chunk_remain = np.minimum(
        video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP
    ) / float(CHUNK_TIL_VIDEO_END_CAP)

    # Add on new information to the state
    state = replace_last_n_elements(state, 1, _last_bit_rate)  # last quality
    state = replace_last_n_elements(state, 0, buffer / BUFFER_NORM_FACTOR)  # 10 sec
    state = replace_last_n_elements(state, 2, _last_video_chunk_size)  # kilo byte / ms
    state = replace_last_n_elements(
        state, 3, float(delay) / M_IN_K / BUFFER_NORM_FACTOR
    )  # 10 sec
    state = replace_last_n_elements(state, 4, _last_video_chunk_remain)  # 10 sec

    return state

def main():
    NN_MODEL = sys.argv[1]
    np.random.seed(RANDOM_SEED)

    # assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(
        all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw
    )

    log_path = TEST_LOG_FILE + "_" + all_file_names[net_env.trace_idx]
    log_file = open(log_path, "w")

    with tf.Session() as sess:
        actor = network.Network(
            sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE
        )

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        time_stamp = 0

        last_action = DEFAULT_ACTION
        action = DEFAULT_ACTION

        action_vec = np.zeros(A_DIM)
        action_vec[action] = DEFAULT_ACTION

        s_batch = [np.zeros(S_DIM)]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        entropy_ = 0.5
        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            (
                delay,
                sleep_time,
                buffer_size,
                rebuf,
                video_chunk_size,
                end_of_video,
                video_chunk_remain,
            ) = net_env.get_video_chunk(action)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            action_detail, _last_bit_rate = get_action_detail_bitrate(action)

            # reward is video quality - rebuffer penalty - smoothness
            reward = (
                reward_bitrate(_last_bit_rate)
                - REBUF_PENALTY * rebuf
                - SMOOTH_PENALTY * penelty_smoothness(action, last_action)
            )

            r_batch.append(reward)

            last_action = action

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(
                str(time_stamp / M_IN_K)
                + "\t"
                + str(action)
                + "\t"
                + str(buffer_size)
                + "\t"
                + str(rebuf)
                + "\t"
                + str(video_chunk_size)
                + "\t"
                + str(delay)
                + "\t"
                + str(entropy_)
                + "\t"
                + str(reward)
                + "\n"
            )
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = np.zeros(S_DIM)
            else:
                state = np.array(s_batch[-1], copy=True)

            # get the new state
            state = get_new_state_from_action(
                state, action, buffer_size, video_chunk_size, delay, video_chunk_remain
            )

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            noise = np.random.gumbel(size=len(action_prob))
            bit_rate = np.argmax(np.log(action_prob) + noise)

            s_batch.append(state)
            entropy_ = -np.dot(action_prob, np.log(action_prob))
            entropy_record.append(entropy_)

            if end_of_video:
                log_file.write("\n")
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                action = DEFAULT_ACTION  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[action] = 1

                s_batch.append(np.zeros(S_DIM))
                a_batch.append(action_vec)
                # print(np.mean(entropy_record))
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = TEST_LOG_FILE + "_" + all_file_names[net_env.trace_idx]
                log_file = open(log_path, "w")


if __name__ == "__main__":
    main()
