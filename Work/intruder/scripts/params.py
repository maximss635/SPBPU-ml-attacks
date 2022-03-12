PATH_ATTACKED_MODEL = '../../ml_model/model.h5'
PATH_MIRROR_MODEL = '../res/mirror_model.h5'

PATH_ATTACKED_DATA = '../../ml_model/data/'
PATH_INFECTED_DATA = '../data_infected/'

BAD_TARGET = 0

NUM_SRC_SAMPLES = 50
NUM_BAD_SAMPLES = 10_000

BAD_PIXEL_X, BAD_PIXEL_Y = 5, 5

__epsilons = [
    0.0,
    0.0002,
    0.0005,
    0.0008,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.01,
    0.1,
    0.3,
    0.5,
    1.0,
]

DEFAULT_EPSILON = __epsilons[-2]
