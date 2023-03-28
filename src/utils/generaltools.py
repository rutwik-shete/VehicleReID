# Copyright (c) EEEM071, University of Surrey

import random

import numpy as np
import torch
import torch.mps as mps

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.has_mps:
        mps.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)
