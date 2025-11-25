import os
import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.nn import MSELoss
from torcheval.metrics import MeanSquaredError, R2Score
from torcheval.metrics.functional import r2_score
from collections import defaultdict
import torchvision
import torch.nn.functional as F

# dataset and unet (could also add to this file, hmm!
from Dataset_test_many_metrics import LandscapeMetricsDataset
from unet_num_metr import unet, Decoder

#import tqdm
try:
    get_ipython().__class_._name__
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

