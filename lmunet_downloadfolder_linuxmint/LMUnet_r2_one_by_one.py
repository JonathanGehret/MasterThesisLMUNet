import csv
import torch
import numpy as np
from torcheval.metrics.functional import r2_score

def r2_one_by_one_old(target_metric, output_metric, file_path=None, number_metrics=57, skip_metrics=True):
    """Returns R2 score for every metric channel."""
    
    # Initialize helper values
    cum_r2 = 0
    r2_list = []
    
    # Iterate over every metric
    for metric_num in range(number_metrics):
        
        # Skip metrics as needed
        if skip_metrics:
            #if metric_num in skip_metrics:
            if (metric_num == 36) or (metric_num == 45) or (metric_num == 50): 
                r2_list.append('nan')
                continue
        
        # Select target and output metrics for metric_num
        target_metric_num = target_metric[metric_num]
        output_metric_num = output_metric[metric_num]

        # Create nan mask and apply it
        nan_mask = torch.isnan(target_metric_num) | torch.isnan(output_metric_num)
        valid_mask = ~nan_mask     
        target_metric_nan = target_metric_num[valid_mask]
        output_metric_nan = output_metric_num[valid_mask]        
      
        # Calculate R2 score for current metric
        R2_metric_num = r2_score(target_metric_nan, output_metric_nan)
       
        # Increment cumulative metric counter
        cum_r2 += R2_metric_num
        
        # Access calculated R2 value and append it to R2 list
        R2_metric_num_item = R2_metric_num.item()
        r2_list.append(R2_metric_num_item)

        # Optionally directly save R2 values to csv
        if file_path:
            with open(file_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([metric_num, R2_metric_num_item])
        
    # Calculate total average R2
    r2_average = cum_r2 / number_metrics
    
    return r2_list, r2_average


def r2_one_by_one_dict(target_metric, ls_opt_dict, ls_number, number_metrics=57):
    
    cum_r2 = 0
    r2_dict = ls_opt_dict.copy()
    r2_dict[ls_number]['R2'] = {}

    for metric_num in range(number_metrics):
        if (metric_num == 36) or (metric_num == 45) or (metric_num == 50): 
            print(1)
            continue

        #target_metric = reverse_dataset[0]['metric'][metric_num].float()
        target_metric_num = target_metric['metric'][metric_num].float()

        #output_metric = ls_opt_dict[ls_number]
        output_metric = torch.tensor(ls_opt_dict[ls_number]['output'][0,metric_num,:,:])
            # dealing with NaNs:
        nan_mask = torch.isnan(target_metric_num)
        valid_mask = ~nan_mask
        target_metric_nan = target_metric_num[valid_mask]
        out_metric_nan = output_metric[valid_mask]

        #R2_metric_num = r2_score(reverse_dataset[0]['metric'][metric_num].float(), out_test[0][metric_num].detach().cpu().float())
        R2_metric_num = r2_score(target_metric_nan, out_metric_nan)

        print(f'{metric_num}: {R2_metric_num}')
        cum_r2 += R2_metric_num
        
        r2_dict[ls_number]['R2'][metric_num] = R2_metric_num.detach().cpu().numpy()
    
    print(f'{cum_r2=}')
    r2_average = cum_r2 / number_metrics
    print(f'{r2_average=}')
    
    return r2_dict, r2_average


    
def loss_one_by_one(target_metric, output_metric, criterion, number_metrics=57):    
    cum_loss = 0    
    loss_list = []
    for metric_num in range(number_metrics):        
        if (metric_num == 36) or (metric_num == 45) or (metric_num == 50): 
            print(1)
            loss_list.append('nan')
            continue
            
        HL_metric_num = criterion(torch.tensor(target_metric[metric_num,:,:]), 
                                  torch.tensor(output_metric[metric_num,:,:]))
        
        loss_list.append(HL_metric_num)
        
        #print(f'{metric_num}: {HL_metric_num}')
        cum_loss += HL_metric_num
        
    #print(f'{cum_loss=}')
    average_loss = cum_loss / 55 
    #print(f'{average_loss=}')

    return loss_list, average_loss



def r2_one_by_one(target_metric, output_metric, file_path=None, number_metrics=57, skip_metrics=True):
    """Returns R2 score for every metric channel."""
    
    # Initialize helper values
    cum_r2 = 0
    r2_list = []
    
    # Iterate over every metric
    for metric_num in range(number_metrics):
        
        # Skip metrics as needed
        if skip_metrics:
            #if metric_num in skip_metrics:
            if (metric_num == 36) or (metric_num == 45) or (metric_num == 50): 
                r2_list.append('nan')
                continue
        
        # Select target and output metrics for metric_num
        target_metric_num = target_metric[metric_num]
        output_metric_num = output_metric[metric_num]
        
        # Calculate R2 score for current metric
        R2_metric_num = r2_nan(target_metric_num, output_metric_num)
       
        # Increment cumulative metric counter
        cum_r2 += R2_metric_num
        
        # Access calculated R2 value and append it to R2 list
        R2_metric_num_item = R2_metric_num.item()
        r2_list.append(R2_metric_num_item)

        # Optionally directly save R2 values to csv
        if file_path:
            with open(file_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([metric_num, R2_metric_num_item])
        
    # Calculate total average R2
    r2_average = cum_r2 / number_metrics
    
    return r2_list, r2_average



def r2_nan(target_metric, output_metric):

    # Create nan mask and apply it
    nan_mask = torch.isnan(target_metric) | torch.isnan(output_metric)
    valid_mask = ~nan_mask     
    target_metric_nan = target_metric[valid_mask]
    output_metric_nan = output_metric[valid_mask]        

    # Calculate R2 score for current metric
    R2_metric_num = r2_score(target_metric_nan, output_metric_nan)
    
    return R2_metric_num