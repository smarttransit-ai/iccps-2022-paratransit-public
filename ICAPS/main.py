import os
import sys
import random
import numpy as np
import tensorflow as tf
from manager import Manager


def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print("Set seed: ", seed_value)


if __name__ == "__main__":
    seed = 111 if len(sys.argv) == 1 else int(sys.argv[1])
    set_seed(seed)
    base_path = "seed_" + str(seed)

    manager = Manager(base_path)
    manager.manage()






