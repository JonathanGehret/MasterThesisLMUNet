import torch
from tqdm import tqdm
import numpy as np

# Conventional landscape optimization
def optimize_one_landscape_conv(target_metric, input_data, model, optimizer, loss_fn, device='cuda'):
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

# With digitization based on threshold values (remove this?)
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


# https://chat.openai.com/share/b97629f3-db87-4f35-99c9-2b49ded55c3b
# https://chat.openai.com/c/432360ff-095c-4d07-8293-217c51983eb1

def apply_integer_constraints(landscape_values, min_class, max_class, mode='clamp'):
    '''Apply integer constraints to landscape to be optimized
    landscape_values -- the landscape to be optimized
    min_class        -- minimum class value
    max_class        -- maximum class value
    mode             -- 'clamp' for rounding and clamping the data, 
                        'rand' to also disperse values outside the range randomly
    '''
    if mode == 'rand':
        # Identify all values outside the min class max class range
        values_outside_range = (landscape_values < min_class) | (landscape_values > max_class)

        # Round all values inside range
        rounded_values = torch.where(~values_outside_range, torch.round(landscape_values), landscape_values)

        # Replace values outside the range with random values between min_class and max_class
        random_values = torch.rand_like(landscape_values) * (max_class - min_class) + min_class
        rounded_values[values_outside_range] = random_values[values_outside_range]
 
    elif mode = 'clamp':        
        # Clamp and round the values to the desired integer range
        clamped_values = torch.clamp(landscape_values, min_class, max_class)
        rounded_values = torch.round(clamped_values)

    return rounded_values

#https://chat.openai.com/share/b97629f3-db87-4f35-99c9-2b49ded55c3b
# https://chat.openai.com/c/432360ff-095c-4d07-8293-217c51983eb1

# Conventional landscape optimization
def optimize_one_landscape(target_metric, input_data, model, optimizer, loss_fn, device='cuda', int_constraints=None):
    '''Function to optimize the input landscape
    target_metric   -- Desired output metrics
    input_data      -- landscapt to be optimized
    model           -- pretrained frozen model
    optimized       -- optimizer optimizing the input_data
    loss_fn         -- loss function to calculate loss between target_metric and model output of input_data
    device          -- run on GPU if 'cuda'; 'cpu' for CPU
    int_constraints -- enter 'rand' or 'clamp' (default) if wanting to constrain.
    '''
    
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
    
    if int_constraints:
        # Apply integer contstraints
        input_data.data = apply_integer_constraints(input_data.data, min_class=1, max_class=5, mode=int_constraints)
    
    # Loss backward through model and optimize + clear gradients
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()                      

    return loss, input_data, out
