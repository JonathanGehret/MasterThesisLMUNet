import os
import torch
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# class for changing images totensor. really necessary? ...
# according to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
# Class level dataset: Landscapes as arrays, metrics as single values or arrays!!

class LandscapeMetricsDataset(Dataset):
    """Dataset for landscapes and their metrics data"""
    def __init__(self, ls_list, landscape_dir, metrics_dir=None, metr_norm=None, mean_std=None, to_binary=None, padding=None, metr_denorm=None, add_noise=None, apply_nan=None,ls_norm=None):
        """Initialising dataset, including locations of landscapes and metrics
        if landscape_lvl: enter metrics_file.csv;
        in metrics_file the second column is landscape lvl metrics
        Enter metric name + index as tuple?. hmm, how to do it in the best way...?
        """
        
        # landscape metric names list
        self.ls_list = ls_list 
        self.metrics_dir = metrics_dir
        self.landscape_dir = landscape_dir

        self.metr_norm = metr_norm
              
        self.meanstd = mean_std
        self.metr_denorm = metr_denorm
        
        self.to_binary = to_binary
        self.padding = padding
        #self.too_big = too_big
        self.apply_nan = apply_nan
        
        self.add_noise = add_noise       
        
        self.ls_norm = ls_norm

    def __len__(self):
        """get length of dataset"""

        return len(self.ls_list)

    def __getitem__(self, i):
        """retrieve one item from dataset
        #landscape = torch.Tensor(landscape) # array to tensor; moved to class ToTensor
        # landscape = landscape[None, :] # add channel dimension; moved to class ToTensor
        """
        
        # landscape id passed when calling dataset or by the dataloader
        ls_id  = self.ls_list.iloc[i]
 
        # set landscape path to corresponding id, load that landscape and set to float32
        landscape_path = os.path.join(self.landscape_dir, ls_id[0])
        landscape = np.load(landscape_path) # import .npy landscapes to numpy array 
            
        # for the wrong approach binary
        if self.to_binary: # necessary because I accidentaly ceated edges up to int 255 
            landscape = (landscape!=0).astype(int)
            
            
        # Convert to float32 if not already
        if landscape.dtype != 'float32':
            landscape = landscape.astype('float32')    
            
        # set metric path to corresponding id, load that metric and set to float32
        metric_path = os.path.join(self.metrics_dir, ls_id[1])      
        metric = np.load(metric_path)            
    
        if metric.dtype != 'float32':
            metric = metric.astype('float32')        
       
        # Convert landscape and metric to tensor
        landscape = ToTensor()(landscape)
        metric = ToTensor()(metric)
                
        if self.metr_norm:
            for metr_id, norm_metric in self.metr_norm:
                
                # Access mean [0] and std [1]
                # the [1:] should be removed and replaced with better approach.
                NormalizeMetr = Normalize(mean = self.meanstd[norm_metric][0], 
                                          std = self.meanstd[norm_metric][1]) 
                
                # Apply normalization at layer     
                metric[metr_id] = NormalizeMetr(metric[metr_id].unsqueeze(0)).squeeze(0) 
    
        # ChatGPT about padding:
        #https://chat.openai.com/c/3e330cc4-4c52-4b09-88c6-060a079c6368
        # REmove padding from here!?
        if self.padding:
            # Calculate the amount of padding needed on each side
            #pad_amount = (target_size - input_size) / 2 #input_image.size(2)

            pad_amount = 49 #(128 - 30) / 2 #input_image.size(2)

            # Apply padding to the input image
            #padded_input = F.pad(input_image, (0, pad_width, 0, pad_height), mode='constant', value=0)

            landscape = F.pad(landscape, (pad_amount, pad_amount, pad_amount, pad_amount), mode='constant', value=0)
            #value=float('nan')
            # alternatively: 'replicate' or 'reflect'
            metric = F.pad(metric, (pad_amount, pad_amount, pad_amount, pad_amount), mode='constant', value=0)
            
            
            
        # https://chat.openai.com/share/9323b3d3-9e33-4765-8978-2d9c19db1837
        #if self.too_big:          
            
                        
        if self.add_noise:
            # Randomly add values between -.5 and .5
            landscape += torch.rand((128,128)) - 0.5
            
            
        if self.ls_norm:
            landscape = (landscape - 1) / 4
        
        return {
            'landscape': landscape, 
            'metric' : metric
        }
    
    
    
