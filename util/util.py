import glob
import math
import os
import random
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm


def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file):
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found
