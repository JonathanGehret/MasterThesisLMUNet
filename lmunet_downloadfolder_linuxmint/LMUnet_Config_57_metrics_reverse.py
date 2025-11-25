#import os
import torch
#import numpy as np
import pandas as pd
#import cv2
from pathlib import Path
from Dataset_test_many_metrics import LandscapeMetricsDataset
from torch.utils.data import DataLoader, random_split


DATA_DIR         = Path(f'../notebooks/data')
TRAIN_IMG_DIR    = DATA_DIR/'100k_landscapes_5_class' # 5 class 100k
TRAIN_DF         = pd.read_csv('data/100k_joinent_5_class_metric_list.csv', header=None) #100k 5 class

# USE THIS IF TRAINING SINGLE CHANNEL METRIC
#METRIC_NAME = 'ai'
METRIC_NAME = None

if METRIC_NAME:
    metr_norm = [(0, 'lsm_l_' + METRIC_NAME)]
    TRAIN_METRIC     = f'100k_{METRIC_NAME}_5_class_metr'

# USE THIS IF TRAINING MULTI-CHANNEL-METRICS

if not METRIC_NAME:
    METRIC_NAME = 'multi_metrics_66'
    TRAIN_METRIC     = f'100k_5cl_{METRIC_NAME}'
    METRIC_NAMES = pd.read_csv(f'data/{METRIC_NAME}.csv', header=None)[0]   

    #METRIC_NAMES         = pd.read_csv(f'data/all_metrics_ordered_short.csv', header=None)
    #METRIC_NAMES         = pd.read_csv(f'data/all_metrics_ordered.csv', header=None)    
    #METRIC_NAMES = ['lsm_l_dcad'] *50
    
    metr_norm = []
    for idx, metr_name in enumerate(METRIC_NAMES):
        metr_norm.append((idx, metr_name))
        #print(metr_norm)
print(metr_norm)

TRAIN_LBL_DIR    = DATA_DIR/TRAIN_METRIC # 100k 5 class
OUTPUT_DIR       = Path(f'outputs/landscape_testiing/{TRAIN_METRIC}')
TRAIN_LOG        = OUTPUT_DIR/'output_log.txt'

print(TRAIN_LBL_DIR)
print(OUTPUT_DIR)

TRAIN_MODEL      = False
EVALUATE         = False
PRETRAINED       = True

REVERSE          = True

TEST_MODEL       = False

GRID_SEARCH      = False
GRID_SEARCH_OPTIM= False

if TEST_MODEL or PRETRAINED or REVERSE:
    #PRETRAINED_DIR   = Path(f'../notebooks/outputs/saved_models')
    #PRETRAINED_PATH  = 'PRETRAINED_DIR/100k_frac_mn_5_class_metr/bst_model128_fold8_0.9017.bin'
    #PRETRAINED_PATH  = 'outputs/saved_models/100k_5cl_multi_metrics_66/bst_model_100k_5cl_multi_metrics_66_bs:16_nw:0_nm:57_lf:HL_NAN_ep:13_0.8937_lr:2.0e-05.bin'
    PRETRAINED_PATH  = 'outputs/saved_models/100k_5cl_multi_metrics_66_pad_zeroes/bst_model_100k_5cl_multi_metrics_66_bs:16_nw:0_nm:57_lf:HL_NAN_ep:50_0.9537_lr:2.0e-05.bin'

IMG_SIZE         = 128
DATASET_LENGHTS  = [80000,15000,5000] # example split for 100k
#DATASET_LENGHTS  = [16,15000,84984] # example split for 100k
#DATASET_LENGHTS  = [1,15000,84999] # example split for 100k
BATCH_SIZE       = 16 #1 for just one landscape
TEST_BATCH_SIZE  = BATCH_SIZE
TRAIN_BATCH_SIZE = BATCH_SIZE
VALID_BATCH_SIZE = BATCH_SIZE
DEVICE           = 'cuda'
LEARNING_RATE    = 2e-5
EPOCHS           = 100
LOSS_FN          = 'HL_NAN' #'MSE_NAN' #'HL_NAN' #'HL' # HuberLoss ['MSE','RMSE','MAE' 'HL is combined mae and ... rmse
EVAL_FN          = 'R2' # ['MSE']
USE_CRIT         = True
NUM_WORKERS      = 0 # was 12, but much stronger with 0; also try 4 (GS?)
#NUM_METRICS      = 1 # automaticalyl detect channel amount later on




#DATA_DIR         = Path(f'../notebooks/data')
#TRAIN_DF         = pd.read_csv('data/100k_joinent_5_class_metric_list.csv', header=None) #100k 5 class
#TRAIN_IMG_DIR    = DATA_DIR/'100k_landscapes_5_class' # 5 class 100k


### f) Dataset

mean_std = 'mean_std.csv'
mean_std = pd.read_csv(mean_std, header=None)
mean_std = {mean_std.iloc[row][0]:[mean_std.iloc[row][1], mean_std.iloc[row][2]] for row in range(len(mean_std))}
#mean_std

#full_dataset = LandscapeMetricsDataset(TRAIN_DF, TRAIN_IMG_DIR, TRAIN_LBL_DIR, metr_norm=metr_norm)
INPUT_LS_DIR = 'data/100k_random_landscape_noises'
full_dataset = LandscapeMetricsDataset(TRAIN_DF, TRAIN_IMG_DIR, TRAIN_LBL_DIR, metr_norm=metr_norm, mean_std=mean_std)

#full_dataset = LandscapeMetricsDataset(TRAIN_DF, TRAIN_IMG_DIR, TRAIN_LBL_DIR, input_ls_dir=INPUT_LS_DIR, metr_norm=metr_norm, mean_std=mean_std)
#full_dataset = LandscapeMetricsDataset(TRAIN_DF, TRAIN_IMG_DIR, TRAIN_LBL_DIR, metr_norm=None)

len(full_dataset)

LS_NUMBER = 95011 #95001 # 11
DATASET_LENGHTS = [LS_NUMBER, 1, len(full_dataset)-LS_NUMBER-1]
DATASET_LENGHTS

# random_split conventinoally a 10k dataset
#from torch.utils.data import random_split

rest1, reverse_dataset, rest2 = random_split(dataset=full_dataset, 
                                                        lengths=DATASET_LENGHTS,
                                                        generator=torch.Generator().manual_seed(42))

print(len(rest1), len(reverse_dataset), len(rest2))

# sanity check
landscape, metric = reverse_dataset[0]['landscape'], reverse_dataset[0]['metric']
#landscape = train_dataset[1]['landscape']

landscape.shape, metric.shape
# (torch.Size([1, 128, 128]), torch.Size([1, 128, 128]))



### h) DataLoaders

REVERSE_BATCH_SIZE = 1

# dataloaders
# shuffle = True means random sampler, which is fine.
reverse_dataloader = DataLoader(reverse_dataset, REVERSE_BATCH_SIZE, 
                              shuffle=True, num_workers=NUM_WORKERS)


# sanity check
landscapes, metrics = next(iter(reverse_dataloader))['landscape'], next(iter(reverse_dataloader))['metric']
landscapes.shape, metrics.shape
# should look like this: (torch.Size([14, 3, 512, 512]), torch.Size([14, 1, 512, 512]))