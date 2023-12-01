import numpy as np
import random

from const import *


class Environment:
    def get_random_coocked_time_bw(self):
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        cooked_time = self.all_cooked_time[self.trace_idx]
        cooked_bw = self.all_cooked_bw[self.trace_idx]
        return cooked_time, cooked_bw

    def get_double_channels_cooked_time_bw(self):
        # Get 2 time, bw set
        _time_1, _bw_1 = self.get_random_coocked_time_bw()
        _time_2, _bw_2 = self.get_random_coocked_time_bw()

        # get min length
        _min_length = min(len(_time_1), len(_time_2))

        # stack, trimmed and assign to class
        # cooked_time = np.vstack((_time_1[:_min_length], _time_1[:_min_length])) # duplication of time_1

        cooked_time = random.choice([_time_1[:_min_length], _time_1[:_min_length]])
        cooked_bw = np.vstack((_bw_1[:_min_length], _bw_2[:_min_length]))
        return cooked_time, cooked_bw

    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        self.cooked_time, self.cooked_bw = self.get_double_channels_cooked_time_bw()

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw[0]) - 1)
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def get_video_quality_from_action(self, quality, chunk):
        if chunk <= TOTAL_VIDEO_CHUNCK:
            return self.video_size[quality][chunk]
        else:
            return 0

    def download_video_chunk_over_mahimahi(
        self, cooked_b, cooked_time, video_chunk_size
    ):
        _last_mahimahi_time = self.last_mahimahi_time
        _mahimahi_ptr = self.mahimahi_ptr
        _video_chunk_counter_sent = 0
        _delay = 0
        _over_mahimahi_ptr_flag = False

        if _mahimahi_ptr >= len(cooked_b):
            _over_mahimahi_ptr_flag = True
            _mahimahi_ptr, self.mahimahi_ptr = 1, 1
            _last_mahimahi_time, self.last_mahimahi_time = 0, 0

        while True:  # download video chunk over mahimahi
            throughput = cooked_b[_mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE
            duration = cooked_time[_mahimahi_ptr] - _last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if _video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (
                    (video_chunk_size - _video_chunk_counter_sent)
                    / throughput
                    / PACKET_PAYLOAD_PORTION
                )
                _delay += fractional_time
                _last_mahimahi_time += fractional_time
                break

            _video_chunk_counter_sent += packet_payload
            _delay += duration
            _last_mahimahi_time = cooked_time[_mahimahi_ptr - 1]
            _mahimahi_ptr += 1

            if _mahimahi_ptr >= len(cooked_b):
                _over_mahimahi_ptr_flag = True
                _mahimahi_ptr, self.mahimahi_ptr = 1, 1
                _last_mahimahi_time, self.last_mahimahi_time = 0, 0
                # loop back in the beginning
                # note: trace file starts with time 0

        _delay *= MILLISECONDS_IN_SECOND
        _delay += LINK_RTT

        # add a multiplicative noise to the delay
        _delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(_delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - _delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = (
                np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME)
                * DRAIN_BUFFER_SLEEP_TIME
            )
            self.buffer_size -= sleep_time

            while True:
                if _mahimahi_ptr >= len(cooked_time):
                    _over_mahimahi_ptr_flag = True
                    _mahimahi_ptr, self.mahimahi_ptr = 1, 1
                    _last_mahimahi_time, self.last_mahimahi_time = 0, 0

                duration = cooked_time[_mahimahi_ptr] - _last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    _last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                _last_mahimahi_time = cooked_time[_mahimahi_ptr]
                _mahimahi_ptr += 1

                if _mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    _over_mahimahi_ptr_flag = True
                    _mahimahi_ptr = 1
                    _last_mahimahi_time = 0

        # Workting with the buffer
        return_buffer_size = self.buffer_size

        # Working with the chunk counter
        self.video_chunk_counter += 1
        # If same path then +=1
        if self.same_path_flag:
            self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            self.cooked_time, self.cooked_bw = self.get_double_channels_cooked_time_bw()

            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw[0]) - 1)
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        _return = [
            _last_mahimahi_time,
            self.video_chunk_counter,
            _delay,
            sleep_time,
            rebuf / MILLISECONDS_IN_SECOND,
            video_chunk_size,
            _over_mahimahi_ptr_flag,
            end_of_video,
        ]

        return _return

    def global_processing_tracked_mahi(self, track_mahi):
        (
            _last_mahimahi_time,
            _video_chunk_counter,
            _delay,
            sleep_time,
            rebuf,
            video_chunk_size,
            _over_mahimahi_ptr_flag,
            end_of_video,
        ) = list(zip(*track_mahi))

        # From 2 path into 1
        self.last_mahimahi_time = max(_last_mahimahi_time)
        self.video_chunk_counter = max(_video_chunk_counter)
        self.delay = max(_delay)
        self.sleep_time = max(sleep_time)
        self.rebuf = max(rebuf)

        if sum(_over_mahimahi_ptr_flag):
            self.mahimahi_ptr = 1
            self.last_mahimahi_time = 0

        if sum(end_of_video):
            self.buffer_size = 0
            self.video_chunk_counter = 0

            self.cooked_time, self.cooked_bw = self.get_double_channels_cooked_time_bw()
            self.mahimahi_ptr = np.random.randint(1, self.cooked_bw.shape[1])
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        # Get the next video chunk size - NOT USE FOR NOW
        # next_video_chunk_sizes = [
        #     self.video_size[i][self.video_chunk_counter] for i in range(BITRATE_LEVELS)
        # ]

        _return_infor = [
            self.delay,
            self.sleep_time,
            self.buffer_size / MILLISECONDS_IN_SECOND,
            self.rebuf,
            video_chunk_size,
            bool(sum(end_of_video)),
            TOTAL_VIDEO_CHUNCK - self.video_chunk_counter,
        ]
        return _return_infor

    def same_path_processing(self, _p1, _p2, _k1, _k2):
        if _p1 == _p2:
            self.same_path_flag = True
            return _p1, _p2, _k1 + _k2, 0
        return _p1, _p2, _k1, _k2

    def sending_2_chunk(self, _p1, _k1, _p2, _k2):
        # need a global variable to track 2 mahimahi then moving together
        self.video_chunk_counter_sent = 0
        self.delay = 0
        self.sleep_time = 0
        self.rebuf = 0
        self.same_path_flag = False

        # tracking the return from the mahimahi simulation
        track_mahi = []

        _p1, _p2, _k1, _k2 = self.same_path_processing(_p1, _p2, _k1, _k2)
        for _patch, _quality in zip([_p1, _p2], [_k1, _k2]):
            track_mahi.append(
                self.download_video_chunk_over_mahimahi(
                    self.cooked_bw[_patch], self.cooked_time, _quality
                )
            )

        # processing 2 track_mahi - combine into 1
        return self.global_processing_tracked_mahi(track_mahi)

    def get_video_chunk(self, action):
        _p1, _q1, _p2, _q2 = ACTION_TABLE[
            action
        ]  # path 1, quality 1, path 2, quality 2

        video_chunk_size_1 = self.get_video_quality_from_action(
            _q1, self.video_chunk_counter
        )
        video_chunk_size_2 = self.get_video_quality_from_action(
            _q2, self.video_chunk_counter + 1
        )

        # use the delivery opportunity in mahimahi
        self.delay = 0.0  # in ms
        self.video_chunk_counter_sent = 0  # in bytes

        # all the information is return here
        infor = self.sending_2_chunk(_p1, video_chunk_size_1, _p2, video_chunk_size_2)

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return infor
