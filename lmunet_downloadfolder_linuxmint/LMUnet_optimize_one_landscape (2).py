import torch
from tqdm import tqdm
import numpy as np

# Conventional landscape optimization
def optimize_one_landscape(target_metric, input_data, model, optimizer, loss_fn, device='cuda'):
    '''Function to optimize the input landscape'''
    
    # Model, target metr and input ls to GPU 
    model = model.to(device)
    input_data = input_data.to(device)
    target_metric = target_metric.to(device)
    
    # Model to evaluation mode
    model.eval()
    
    # Run input landscape through model
    out  = model(input_data)           
    
    # Calculate loss 
    loss = loss_fn(out, target_metric)  
    
    # Loss backward through model and optimize + clear gradients
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()                      

    return loss, input_data, out

# Optimization with fixed digitization    
def optimize_one_landscape_fixed_digitization(target_metric, input_data, model, optimizer, loss_fn, device='cuda', epoch=0):
    '''Function to optimize the input landscape and digitize it by fixed values'''
    
    # Model, target metr and input ls to GPU 
    model = model.to(device)
    input_data = input_data.to(device)
    target_metric = target_metric.to(device)
    
    # Model to evaluation mode
    model.eval()
            
    # Fixed digitization (roughly round), plus outliers are put to 3
    if epoch%5 == 1:
        print('100')
        input_data = input_data.detach().cpu().numpy()
        input_data[input_data >= 5.5] = 3
        input_data[np.logical_and(input_data >= 4.5, input_data < 5.5)] = 5
        input_data[np.logical_and(input_data >= 3.5, input_data < 4.5)] = 4
        input_data[np.logical_and(input_data >= 2.5, input_data < 3.5)] = 3
        input_data[np.logical_and(input_data >= 1.5, input_data < 2.5)] = 2
        input_data[np.logical_and(input_data >= 0.5, input_data < 1.5)] = 1
        input_data[input_data < 0.5] = 3
        input_data_digitized = input_data.copy()
        input_data = torch.from_numpy(input_data)
        input_data.requires_grad_()
        input_data = input_data.to(device)
            
    # Run (digitized) input landscape through model
    out = model(input_data)           
    
    # Calculate loss 
    loss = loss_fn(out, target_metric)  
    
    # Loss backward through model and optimize + clear gradients
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()                        

    return loss, input_data, out

# With digitization based on threshold values
# Simlar to previous, just different values
def optimize_one_landscape_digitize(train_loader, input_data, model, optimizer, loss_fn, device='cuda', epoch=0):
    '''Function to optimize the input landscape and digitize it by threshold values'''
    
    # Model, target metr and input ls to GPU 
    model = model.to(device)
    input_data = input_data.to(device)
    target_metric = target_metric.to(device)
    
    # Model to evaluation mode
    model.eval()         
    
    # Run input landscape through model
    out = model(input_data)           
    
    # Calculate loss 
    loss = loss_fn(out, target_metric)  
   
    # If digitize:
    q_all = np.percentile(input_data.detach().cpu().numpy(), all_p_cs100)
    inp_digi = np.digitize(input_data.detach().cpu().numpy(), q_all).astype('float32') +1.
    input_data = torch.from_numpy(inp_digi)
    input_data.requires_grad_()

    # Loss backward through model and optimize + clear gradients
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()                       

    return loss, input_data, out
    

def optimize_one_landscape_digitize_one_class(train_loader, input_data, model, optimizer, loss_fn, device='cuda', epoch=0):
    model = model.to(device)
    input_data = input_data.to(device)
    model.eval()

    target_metric = target_metric.to(device)          
    
    loss = loss_fn(out, target_metric) 
    if epoch%100 == 1:
        print("lolo")
    # hier digitization step? without gradient calc.
    #q_all = np.percentile(input_data.detach().cpu().numpy(), all_p_cs100)
    #inp_digi = np.digitize(input_data.detach().cpu().numpy(), q_all).astype('float32') +1.

        input_detached = input_data.detach().cpu().numpy()

        hist_bin_edges_input_data = np.histogram_bin_edges(input_detached, bins=6)
        #input_digitized = np.digitize(input_detached, hist_bin_edges_input_data[:-1]).astype('float32')
        input_digitized = np.digitize(input_detached, hist_bin_edges_input_data, right=True).astype('float32')
        print(np.unique(input_digitized))
        input_digitized[input_digitized == 0] = np.random.randint(1, 5)#.astype('float32')
        input_digitized[input_digitized == 6] = np.random.randint(1, 5)#.astype('float32')


        #input_data_digitized = input_digitized.copy()

        #print(np.unique(inp_digi))
        input_data = torch.from_numpy(input_digitized)#.astype('float32')
        input_data.requires_grad_()

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()                      

    return loss, input_data, out
    


# https://chat.openai.com/share/b97629f3-db87-4f35-99c9-2b49ded55c3b
# https://chat.openai.com/c/432360ff-095c-4d07-8293-217c51983eb1

#import torch

def apply_integer_constraints(landscape_values, min_class, max_class):
    rounded_values = torch.round(landscape_values)
    clamped_values = torch.clamp(rounded_values, min_class, max_class)
    return clamped_values

#https://chat.openai.com/share/b97629f3-db87-4f35-99c9-2b49ded55c3b
# https://chat.openai.com/c/432360ff-095c-4d07-8293-217c51983eb1


def apply_integer_constraints_frac(landscape_values, min_class, max_class):
    # Calculate the fractional part of the values
    fractional_part = torch.frac(landscape_values)
    
    # Set values n % 1 > 0.5 to floor, and n % 1 < 0.5 to ceil
    # If still close: reset  
    floor_mask = ((fractional_part >= 0.5) & (fractional_part <= 0.9)) | ((fractional_part >= 0) & (fractional_part <= 0.1))
    ceil_mask = ~floor_mask
    landscape_values[floor_mask] = torch.floor(landscape_values[floor_mask])
    landscape_values[ceil_mask] = torch.ceil(landscape_values[ceil_mask])
    
    # Apply integer constraints for values outside the desired range
    #values_outside_range = (landscape_values < min_class) | (landscape_values > max_class)
    
    # Replace values outside the range with random values between 1 and 5
    #random_values = torch.rand_like(landscape_values) * (max_class - min_class) + min_class
    #landscape_values[values_outside_range] = random_values[values_outside_range]
    
    # Clamp the values to the desired integer range
    clamped_values = torch.clamp(landscape_values, min_class, max_class)
    return clamped_values


def apply_integer_constraints_hmm(landscape_values, min_class, max_class):
    fractional_part = torch.round(landscape_values)
    
    # Apply integer constraints for values outside the desired range
    values_outside_range = (landscape_values < min_class) | (landscape_values > max_class)
      
    # Replace values outside the range with random values between 1 and 5
    random_values = torch.rand_like(landscape_values) * (max_class - min_class) + min_class
    landscape_values[values_outside_range] = random_values[values_outside_range]
 
    # Clamp the values to the desired integer range
    clamped_values = torch.clamp(landscape_values, min_class, max_class)
    return clamped_values

def apply_integer_constraints_clamp_only(landscape_values, min_class, max_class):

    # Apply to higher and lower
    values_below_range = landscape_values < min_class
    values_above_range = landscape_values > max_class
   
    # Replace higher and lower values
    landscape_values[values_below_range] = max_class
    landscape_values[values_above_range] = min_class

    # Clamp the values to the desired integer range
    #clamped_values = torch.clamp(landscape_values, min_class, max_class)
    return landscape_values


def optimize_one_landscape_int_constraints(target_metric, input_data, model, optimizer, loss_fn, epoch, device='cuda'):
    '''Function to optimize the input landscape with integer constraints'''
    
    # Model, target metr and input ls to GPU 
    model = model.to(device)
    input_data = input_data.to(device)
    target_metric = target_metric.to(device)
    
    # Model to evaluation mode
    model.eval()
    
    # Run input landscape through model
    out  = model(input_data)           
    
    # Calculate loss 
    loss = loss_fn(out, target_metric)  
    
    # Apply integer contstraints
    input_data.data = apply_integer_constraints(input_data.data, 0, 1)
    
    # Loss backward through model and optimize + clear gradients
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()                            

    return loss, input_data, out



