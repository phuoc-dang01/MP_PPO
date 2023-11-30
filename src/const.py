# VIDEO_BIT_RATE = np.array([300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0])  # Kbps
VIDEO_BIT_RATE = [300.0, 1850.0, 4300.0]  # Kbps
BITRATE_LEVELS = 3


S_INFO = (
    5  # last quality, buffer_size, last video chunks size, delay, chunk_til_video_end
)
S_LEN = 8  # take how many frames in the past
S_DIM = (S_INFO, S_LEN)
A_DIM = (2 * BITRATE_LEVELS) ** 2

NUM_AGENTS = 2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality - select the lowest quality
DEFAULT_ACTION = 3  # Package 1 through channel 0 low, package 2 through channel 1 low


ACTION_TABLE = {
    0: [0, 0, 0, 0],
    1: [0, 0, 0, 1],
    2: [0, 0, 0, 2],
    3: [0, 0, 1, 0],
    4: [0, 0, 1, 1],
    5: [0, 0, 1, 2],
    6: [0, 1, 0, 0],
    7: [0, 1, 0, 1],
    8: [0, 1, 0, 2],
    9: [0, 1, 1, 0],
    10: [0, 1, 1, 1],
    11: [0, 1, 1, 2],
    12: [0, 2, 0, 0],
    13: [0, 2, 0, 1],
    14: [0, 2, 0, 2],
    15: [0, 2, 1, 0],
    16: [0, 2, 1, 1],
    17: [0, 2, 1, 2],
    18: [1, 0, 0, 0],
    19: [1, 0, 0, 1],
    20: [1, 0, 0, 2],
    21: [1, 0, 1, 0],
    22: [1, 0, 1, 1],
    23: [1, 0, 1, 2],
    24: [1, 1, 0, 0],
    25: [1, 1, 0, 1],
    26: [1, 1, 0, 2],
    27: [1, 1, 1, 0],
    28: [1, 1, 1, 1],
    29: [1, 1, 1, 2],
    30: [1, 2, 0, 0],
    31: [1, 2, 0, 1],
    32: [1, 2, 0, 2],
    33: [1, 2, 1, 0],
    34: [1, 2, 1, 1],
    35: [1, 2, 1, 2],
}  # channel of package 1, quaylity 1, channel of package 2, quality 2

RANDOM_SEED = 42


TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 500000
MODEL_SAVE_INTERVAL = 300

LOG_FILE = "./test_results/log_sim_ppo"
TEST_TRACES = "./test/"
# TEST_TRACES = "./SAM_test_trace/"
SUMMARY_DIR = "./ppo"
MODEL_DIR = "./models"
TRAIN_TRACES = "./train/"
# TRAIN_TRACES = "./SAM_processed_trace/"
TEST_LOG_FOLDER = "./test_results/"
LOG_FILE = SUMMARY_DIR + "/log"
PPO_TRAINING_EPO = 5
VIDEO_SIZE_FILE = "./envivio/video_size_"

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer

TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
# PPO2
EPS = 0.2

RAND_RANGE = 1000
