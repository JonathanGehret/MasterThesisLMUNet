# Denormalize the output metrics.
import numpy as np
import torch
from torchvision.transforms import ToTensor, Normalize, Compose

# Input tensor/array of metric map stacks of format [n_metrics, 128, 128]
def DenormNameApplyNan(predicted_metrics, mean_std, metr_norm, nan_metrics=None):  
    ''' 
    Function to add the predicted metrics to a dictionary, 
    name them, denormalize them, round the binary mask 
    and apply the binary mask if necessary.
    nan_metrics should be a list of strings of the names of all
    metrics to add nans to.
    '''
    
    metric_denorm_dict = {}
    for metr_id, metric_name in metr_norm:

        # Access mean (0) and std (1) for metric_name
        try:
            metr_mean = mean_std[metric_name]['mean'] # [0]
        except:
            metr_mean = mean_std[metric_name][0] # [0]
            
        try:
            metr_std  = mean_std[metric_name]['std'] # [1]
        except:
            metr_std  = mean_std[metric_name][1] # [1]

        # Squeeze if shape [1,n_metrics, 128, 128]
        metric = predicted_metrics.squeeze(0)[metr_id]

        # Compose Denormalize class
        DenormalizeMetr = Compose([Normalize(mean = 0, std = 1/metr_std),
            Normalize(mean = -1 * metr_mean, std = 1),])

        # Simply round to 0 and 1 if nan mask
        if metric_name == 'lsm_l_nan_mask':
            metric_denorm_dict[metric_name] = torch.round(metric)         
        
        # Aply NaN mask if applicable
        elif nan_metrics:
            if metric_name in nan_metrics:
                denormed_metric = DenormalizeMetr(metric.unsqueeze(0)).squeeze(0) 
                denormed_metric[metric_denorm_dict['lsm_l_nan_mask'] == 0] = float('nan')

                metric_denorm_dict[metric_name] = denormed_metric

        else:
            # Apply denorm class into dict with correct metric names.          
            metric_denorm_dict[metric_name] = DenormalizeMetr(metric.unsqueeze(0)).squeeze(0) 

    return(metric_denorm_dict)

def ReplaceHighValues(arr):
    """Fucntion to replace all values > len(arr) with values <len(arr)."""

    # Step 1: Count all individual numbers in the array
    unique_values = np.unique(arr)
    print(f'{unique_values=}')

    # Step 2: Determine the number of unique values
    n = len(unique_values)
    print(f'{n=}')

    # Step 2.5: create a list with all values to replace them
    replacement_list = [i for i in range(n) if i not in unique_values]

    # Step 3: Create a random mapping for values > n
    replacement_mapping = {}
    for value in unique_values:
        if value > n:
            #print(f'{value=}')
            replacement_value = replacement_list.pop(0)
            #print(f'{replacement_value=}')
            replacement_mapping[value] = replacement_value
            #print(f'{replacement_value=}')

    # Replace values > n using the mapping
    for old_value, new_value in replacement_mapping.items():
        arr[arr == old_value] = new_value
    
    return arr