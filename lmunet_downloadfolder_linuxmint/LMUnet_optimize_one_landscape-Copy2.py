import torch
from tqdm import tqdm
import numpy as np
from LMUnet_connected_component_labeling import connected_component_labeling

# Averagemeter plays a role in taking care of the losses.

# It's initialized in train_one_epoch, having the values set at zero.
# Reset appears to be unused.
# It gets updated with the current loss, which is handed as "val", and batch_size as n.
# val is therefore the loss score.
# sum is previous sum loss score plus val multiplied by batch size.
# count is previous batches + new one
# avg is sum of all loss scores divided by count of all batches.

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#with original landscape
def optimize_one_landscape_with_tqdm(train_loader, input_data, model, optimizer, loss_fn, accumulation_steps=1, device='cuda'):

    losses = AverageMeter()
    model = model.to(device)        
    input_data = input_data.to(device)
    model.eval()

    if accumulation_steps > 1: 
        optimizer.zero_grad()
    tk0 = tqdm(train_loader, total=1)
    for b_idx, data in enumerate(tk0):   

        if accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()

        data['metric'] = data['metric'].to(device)
        
        # Fake Quant
        #input_data = torch.fake_quantize_per_tensor_affine(input_data, scale=1, zero_point=1, quant_min=1, quant_max=6)
        
        out  = model(input_data)            
        #loss = loss_fn(out, data['metric'])  
        # skip certain metrics, i.e. 45 and 50 for loss calculation
        out_cat = torch.cat((out[:36], out[37:45], out[46:50], out[51:]), dim=0)
        target_cat = torch.cat((data['metric'][:36], data['metric'][37:45], data['metric'][46:50], data['metric'][51:]), dim=0)
        loss = loss_fn(out_cat, target_cat)    
        #loss = loss_fn(out_cat[:][:][49:79, 49:79], target_cat[:][:][49:79, 49:79])    
        #torch.cat((input_tensor[:45], input_tensor[46:50], input_tensor[51:]), dim=0)
        loss.backward()

        if (b_idx + 1) % accumulation_steps == 0:               
            optimizer.step()
            optimizer.zero_grad()                      

        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])
        #tk0.set_postfix(loss=loss, learning_rate=optimizer.param_groups[0]['lr'])

        if b_idx%100 == 1:
            cpu_input_data = input_data.clone().detach()
            cpu_out = out.clone().detach()
            ls_num = 15
            metric_num = 1
            visualize(test_dataset[ls_num]['landscape'].squeeze(),
                      cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
              left_title= f"Target Landscape", right_title = f"Predicted Landscape")
            visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                      cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
              left_title= f"Target Metric", right_title = f"Predicted Metric")

    return losses.avg, input_data, out
    #return loss, input_data, out

#with original landscape
def optimize_one_landscape(target_metric, input_data, model, optimizer, loss_fn, device='cuda'):

    #losses = AverageMeter()
    model = model.to(device)        
    input_data = input_data.to(device)
    model.eval()

    optimizer.zero_grad()


    
    target_metric = target_metric.to(device)
    
    out  = model(input_data)            
    #loss = loss_fn(out, data['metric'])  
    
    # skip certain metrics, i.e. 45 and 50 for loss calculation
    #out_cat = torch.cat((out[:36], out[37:45], out[46:50], out[51:]), dim=0)
    #target_cat = torch.cat((target_metric[:36], target_metric[37:45], target_metric[46:50], target_metric[51:]), dim=0)
    #loss = loss_fn(out_cat, target_cat)    
    
    
    loss = loss_fn(out, target_metric)  
    
    loss.backward()

    #if (b_idx + 1) % accumulation_steps == 0:               
    optimizer.step()
    optimizer.zero_grad()                      


    '''
    if b_idx%100 == 1:
        cpu_input_data = input_data.clone().detach()
        cpu_out = out.clone().detach()
        ls_num = 15
        metric_num = 1
        visualize(test_dataset[ls_num]['landscape'].squeeze(),
                  cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
          left_title= f"Target Landscape", right_title = f"Predicted Landscape")
        visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                  cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
          left_title= f"Target Metric", right_title = f"Predicted Metric")
    '''
    
    #return losses.avg, input_data, out
    return loss, input_data, out
    
def optimize_one_landscape_patch_by_patch(train_loader, patch_input_test_CCL, region_map, model, optimizer, loss_fn, 
                           unique_dict, accumulation_steps=1, device='cuda'):

    #losses = AverageMeter()
    model = model.to(device)             
    model.eval()


    #print(unique_dict)
    if accumulation_steps > 1:
        optimizer.zero_grad()

    best_unique_ls = patch_input_test_CCL.copy()
    #best_regions = unique_regions.copy()
    best_loss = 1000 #set this to the besot loss of the input landscapes's metrics.

    for i in unique_dict:
        #unique_dicts = []
        #best_dict = {}
        #if i % 1 == 0:                
         #   i = 0
        print(f'{i=}')
        for n in [1, 2, 3, 4, 5]:
            
            
            best_ls_n = best_unique_ls.copy()
            best_ls_n[region_map == i] = n 
            
            #print(f'{best_ls_n=}')            
            #print((best_ls_n == best_unique_ls).any == False)
            
            
            # visualize(best_unique_ls,
            #                   best_ls_n,
            #                   left_title= f"best_unique_ls", 
            #                   right_title = f"best_loss")   

            #print(f'{np.unique(input_data)=}')

            #input_data = torch.tensor(input_data)
            input_data = torch.tensor(best_ls_n).to(torch.float32)
            input_data = input_data.unsqueeze(0).unsqueeze(0).to(device) 
            #print(f'{input_data.shape=}')


            #tk0 = tqdm(train_loader, total=len(train_loader))
            tk0 = tqdm(train_loader, total=1)
            for b_idx, data in enumerate(tk0):   

                if accumulation_steps == 1 and b_idx == 0:
                    optimizer.zero_grad()

                data['metric'] = data['metric'].to(device)
                #print(data['metric'].shape)
                #print(data['metric'].shape)
                out  = model(input_data)     
                #print(f'{out.shape=}')
                loss = loss_fn(out, data['metric']) 

                #loss = loss +  auxiliary_loss_function(percentages_optimize)

                loss.backward()

                if (b_idx + 1) % accumulation_steps == 0:               
                    optimizer.step()
                    optimizer.zero_grad()                      

                #losses.update(loss.item(), train_loader.batch_size)


                tk0.set_postfix(loss=loss, learning_rate=optimizer.param_groups[0]['lr'])


                print(f'{loss=}')
                print(f'{best_loss=}')

                if loss < best_loss:
                    #best_dict = unique_dict  
                    #best_input = input_data.squeeze(0).squeeze(0).detach().cpu().numpy()
                    best_ls = best_ls_n.copy()
                    #best_unique_ls = best_ls_n
                    best_loss = loss


                #print(f'{divisor_optimize=}')
                #print(f'{percentages_optimize=}')

                '''
                if b_idx%100 == 0:
                    visualize(best_unique_ls,
                              best_ls_n,
                              left_title= f"best_unique_ls", 
                              right_title = f"best_loss")       
                '''

        # update best ls with best one
        best_unique_ls = best_ls.copy()

        # after iterating through all 5 numbers
        #unique_dict = best_dict


    return loss, input_data, out, best_loss


def optimize_one_landscape_nodes_by_edges(train_loader, patch_input_test_CCL, region_map, model, optimizer, loss_fn, 
                           unique_dict, accumulation_steps=1, device='cuda'):

    #losses = AverageMeter()
    model = model.to(device)             
    model.eval()


    #print(unique_dict)
    if accumulation_steps > 1:
        optimizer.zero_grad()

    best_unique_ls = patch_input_test_CCL.copy()
    #best_regions = unique_regions.copy()
    best_loss = 1000 #set this to the besot loss of the input landscapes's metrics.

    for i in unique_dict:
        #unique_dicts = []
        #best_dict = {}
        #if i % 1 == 0:                
         #   i = 0
        print(f'{i=}')
        for n in [1, 2, 3, 4, 5]:
            
            
            best_ls_n = best_unique_ls.copy()
            best_ls_n[region_map == i] = n 
            
            #print(f'{best_ls_n=}')            
            #print((best_ls_n == best_unique_ls).any == False)
            
            
            # visualize(best_unique_ls,
            #                   best_ls_n,
            #                   left_title= f"best_unique_ls", 
            #                   right_title = f"best_loss")   

            #print(f'{np.unique(input_data)=}')

            #input_data = torch.tensor(input_data)
            input_data = torch.tensor(best_ls_n).to(torch.float32)
            input_data = input_data.unsqueeze(0).unsqueeze(0).to(device) 
            #print(f'{input_data.shape=}')


            #tk0 = tqdm(train_loader, total=len(train_loader))
            tk0 = tqdm(train_loader, total=1)
            for b_idx, data in enumerate(tk0):   

                if accumulation_steps == 1 and b_idx == 0:
                    optimizer.zero_grad()

                data['metric'] = data['metric'].to(device)
                #print(data['metric'].shape)
                #print(data['metric'].shape)
                out  = model(input_data)     
                #print(f'{out.shape=}')
                loss = loss_fn(out, data['metric']) 

                #loss = loss +  auxiliary_loss_function(percentages_optimize)

                loss.backward()

                if (b_idx + 1) % accumulation_steps == 0:               
                    optimizer.step()
                    optimizer.zero_grad()                      

                #losses.update(loss.item(), train_loader.batch_size)


                tk0.set_postfix(loss=loss, learning_rate=optimizer.param_groups[0]['lr'])


                print(f'{loss=}')
                print(f'{best_loss=}')

                if loss < best_loss:
                    #best_dict = unique_dict  
                    #best_input = input_data.squeeze(0).squeeze(0).detach().cpu().numpy()
                    best_ls = best_ls_n.copy()
                    #best_unique_ls = best_ls_n
                    best_loss = loss


                #print(f'{divisor_optimize=}')
                #print(f'{percentages_optimize=}')

                '''
                if b_idx%100 == 0:
                    visualize(best_unique_ls,
                              best_ls_n,
                              left_title= f"best_unique_ls", 
                              right_title = f"best_loss")       
                '''

        # update best ls with best one
        best_unique_ls = best_ls.copy()

        # after iterating through all 5 numbers
        #unique_dict = best_dict


    return loss, input_data, out, best_loss


# with fixed digitization
    
def optimize_one_landscape_fixed_digitization(train_loader, input_data, model, optimizer, loss_fn, accumulation_steps=1, device='cuda', epoch=0):
    losses = AverageMeter()
    model = model.to(device)
    input_data_digitized = input_data.detach().cpu().numpy()
    input_data = input_data.to(device)
    
    model.eval()
    if accumulation_steps > 1: 
        optimizer.zero_grad()
    tk0 = tqdm(train_loader, total=1)
    for b_idx, data in enumerate(tk0):   
        if accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()
        data['metric'] = data['metric'].to(device)
        
        print(epoch)
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
            
        out  = model(input_data)
        loss = loss_fn(out, data['metric'])
        
        loss.backward()
        if (b_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])

    return losses.avg, input_data, out, input_data_digitized



# With digitization "hack"
    

def optimize_one_landscape_digitize(train_loader, input_data, model, optimizer, loss_fn, accumulation_steps=1, device='cuda', epoch=0):
#def optimize_one_landscape(train_loader, model, loss_fn, accumulation_steps=1, device='cuda'):
    losses = AverageMeter()
    model = model.to(device)

    #input_data = torch.randn((16,1,128,128), requires_grad=True)  # Random initialization        
    #optimizer = torch.optim.Adam([input_data], lr=0.01)  # input_data will be optimized

    input_data = input_data.to(device)

    model.eval()
    if accumulation_steps > 1: 
        optimizer.zero_grad()
    #tk0 = tqdm(train_loader, total=len(train_loader))
    tk0 = tqdm(train_loader, total=1)
    for b_idx, data in enumerate(tk0):   
        #print(b_idx)
        if accumulation_steps == 1 and b_idx == 0:
            #print('sdf')
            optimizer.zero_grad()
        data['metric'] = data['metric'].to(device)
        #print(data['landscape'].size())
        #input_data = ReLU(input_data)
        #input_data = torch.max(0, torch.min(5, input_data))
        #input_data = 5 * torch.relu(input_data)
         #5 * relu_output  # Scale the output by 5
        #input_data = torch.clamp(input_data, min=0, max=5)
        out  = model(input_data)
        #print(out.size())
        #print(data['metric'].size())
        loss = loss_fn(out, data['metric'])
        #print(f'{loss.requires_grad=}')
        #with torch.set_grad_enabled(True):
        #with loss.set_grad_enabled(True): #error
            #print(f'{loss.requires_grad=}')

        #print(epoch)
        #print(epoch%3)
        #if epoch%3 == 1:
        #print("lolo")
        # hier digitization step? without gradient calc.
        q_all = np.percentile(input_data.detach().cpu().numpy(), all_p_cs100)
        inp_digi = np.digitize(input_data.detach().cpu().numpy(), q_all).astype('float32') +1.
        #print(np.unique(inp_digi))
        input_data = torch.from_numpy(inp_digi)
        input_data.requires_grad_()

        #input_data = inp_digi
        #optimizer = torch.optim.Adam([input_data], lr=lr)  # input_data will be optimized
        #input_data = input_data.to(device)

        loss.backward()
        #print(f'{loss.requires_grad=}')
        if (b_idx + 1) % accumulation_steps == 0:
            #print('sdf')
            optimizer.step()
            optimizer.zero_grad()


        #optimizer.step()
        #print(f'{loss.requires_grad=}')

        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])

        # experiment with putting a filter ??





        if b_idx%100 == 1:
            cpu_input_data = input_data.clone().detach()
            cpu_out = out.clone().detach()
            ls_num = 15
            metric_num = 1
            #visualize(predicted_metric[2],predicted_metric[2])
            visualize(test_dataset[ls_num]['landscape'].squeeze(),
                      cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
              left_title= f"Target Landscape", right_title = f"Predicted Landscape")
            visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                      cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
              left_title= f"Target Metric", right_title = f"Predicted Metric")

    return losses.avg, input_data, out


def optimize_one_landscape_digitize_one_class(train_loader, input_data, model, optimizer, loss_fn, accumulation_steps=1, device='cuda', epoch=0):
#def optimize_one_landscape(train_loader, model, loss_fn, accumulation_steps=1, device='cuda'):
    losses = AverageMeter()
    model = model.to(device)

    #input_data = torch.randn((16,1,128,128), requires_grad=True)  # Random initialization        
    #optimizer = torch.optim.Adam([input_data], lr=0.01)  # input_data will be optimized

    #input_data_digitized = input_data.clone().detach().cpu().numpy()
    
    input_data = input_data.to(device)

    model.eval()
    if accumulation_steps > 1: 
        optimizer.zero_grad()
    #tk0 = tqdm(train_loader, total=len(train_loader))
    tk0 = tqdm(train_loader, total=1)
    for b_idx, data in enumerate(tk0):   
        #print(b_idx)
        if accumulation_steps == 1 and b_idx == 0:
            #print('sdf')
            optimizer.zero_grad()
        data['metric'] = data['metric'].to(device)
        #print(data['landscape'].size())
        #input_data = ReLU(input_data)
        #input_data = torch.max(0, torch.min(5, input_data))
        #input_data = 5 * torch.relu(input_data)
         #5 * relu_output  # Scale the output by 5
        #input_data = torch.clamp(input_data, min=0, max=5)
        out  = model(input_data)
        #print(out.size())
        #print(data['metric'].size())
        loss = loss_fn(out, data['metric'])
        #print(f'{loss.requires_grad=}')
        #with torch.set_grad_enabled(True):
        #with loss.set_grad_enabled(True): #error
            #print(f'{loss.requires_grad=}')

        #print(epoch)
        #print(epoch%3)
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

        #input_data = inp_digi
        #optimizer = torch.optim.Adam([input_data], lr=lr)  # input_data will be optimized
        #input_data = input_data.to(device)

        loss.backward()
        #print(f'{loss.requires_grad=}')
        if (b_idx + 1) % accumulation_steps == 0:
            #print('sdf')
            optimizer.step()
            optimizer.zero_grad()


        #optimizer.step()
        #print(f'{loss.requires_grad=}')

        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])

        # experiment with putting a filter ??





        if b_idx%100 == 1:
            cpu_input_data = input_data.clone().detach()
            cpu_out = out.clone().detach()
            ls_num = 15
            metric_num = 1
            #visualize(predicted_metric[2],predicted_metric[2])
            visualize(test_dataset[ls_num]['landscape'].squeeze(),
                      cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
              left_title= f"Target Landscape", right_title = f"Predicted Landscape")
            visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                      cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
              left_title= f"Target Metric", right_title = f"Predicted Metric")

    return losses.avg, input_data, out#, input_data_digitized

# graphs
def optimize_one_landscape_graph(train_loader, input_data, model, optimizer, loss_fn, 
                           divisor_optimize, percentages_optimize, accumulation_steps=1, device='cuda'):

    losses = AverageMeter()
    model = model.to(device)             
    model.eval()

    if accumulation_steps > 1:
        optimizer.zero_grad()

    #input_data = 

    # 1. identify patches        
    input_data, num_patches = connected_component_labeling(input_data[0,0,:,:].detach().cpu().numpy(), 
                                                           divisor_optimize)

    # 2. Create an empty graph
    map_graph = nx.Graph()

    # 3. Add nodes representing regions
    unique_regions = np.unique(input_data)
    map_graph.add_nodes_from(unique_regions)

    # 4. label nodes ##
    node_labels_ordered = node_label_preparator(num_patches, percentages_optimize)

    # 5. Give nodes random numbers    
    numbered_graph = digitize_nodes(map_graph, node_labels_ordered)

    # print(numbered_graph.nodes[5.]['number'])

    #print(numbered_graph.nodes)
    # 6. return nodes to array
    #print(f'{np.unique(input_data)=}')

    input_data = graph_to_array(input_data, numbered_graph.nodes)
    #print(f'{np.unique(input_data)=}')

    input_data = input_data.to(device)        


    #tk0 = tqdm(train_loader, total=len(train_loader))
    tk0 = tqdm(train_loader, total=16)
    for b_idx, data in enumerate(tk0):   

        if accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()

        data['metric'] = data['metric'].to(device)
        out  = model(input_data)            
        loss = loss_fn(out, data['metric']) 

        #loss = loss +  auxiliary_loss_function(percentages_optimize)

        loss.backward()

        if (b_idx + 1) % accumulation_steps == 0:               
            optimizer.step()
            optimizer.zero_grad()                      

        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])

        print(f'{divisor_optimize=}')
        print(f'{percentages_optimize=}')

        if b_idx%100 == 1:
            cpu_input_data = input_data.clone().detach()
            cpu_out = out.clone().detach()
            ls_num = 15
            metric_num = 1
            #visualize(predicted_metric[2],predicted_metric[2])
            visualize(test_dataset[ls_num]['landscape'].squeeze(),
                      cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
              left_title= f"Target Landscape", right_title = f"Predicted Landscape")
            visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                      cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
              left_title= f"Target Metric", right_title = f"Predicted Metric")

    return losses.avg, input_data, out


#with original landscape testing without graphs

#from LMUnet_graph_operations import patch_label_preparat
from LMUnet_connected_component_labeling import connected_component_labeling

def optimize_one_landscape_CCL(train_loader, input_data, model, optimizer, loss_fn, 
                               accumulation_steps=1, device='cuda', epoch=0):
        
    losses = AverageMeter()
    model = model.to(device)             
    model.eval()
    input_data_CCL = input_data.detach().cpu().numpy()
    
    divisor_optimize = 7

    if accumulation_steps > 1:
        optimizer.zero_grad()

    #input_data = 

    # 1. identify patches        
    #patch_input_test_CCL, num_patches = connected_component_labeling(input_opt_ls[0,0,:,:].detach().cpu().numpy(), 
    #                                                       divisor_optimize)

    #region_map = (patch_input_test_CCL / num_patches)# * 4 + 1.0000003
    # 4. label nodes ##
    #patch_labels_ordered = patch_label_preparator(num_patches, percentages_optimize)

    #input_data_CCL = torch.tensor(input_opt_ls).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)        
    #input_data = torch.tensor(input_data).to(torch.float32).to(device)        
    input_data = input_data.to(device)        


    #tk0 = tqdm(train_loader, total=len(train_loader))
    tk0 = tqdm(train_loader, total=1)
    for b_idx, data in enumerate(tk0):   

        if accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()

        data['metric'] = data['metric'].to(device)
        
        if epoch%5 == 1:
            print('lul')
            patch_input_test_CCL, num_patches = connected_component_labeling(input_data[0,0,:,:].detach().cpu().numpy(), 
                                                           divisor=1)
            region_map = (patch_input_test_CCL / num_patches) * 4 + 1.0000003
            
            input_data_CCL = region_map.copy()
            input_data = torch.from_numpy(region_map).unsqueeze(0).unsqueeze(0)
            input_data.requires_grad_()
            input_data = input_data.to(device)
        
        
        out  = model(input_data)            
        loss = loss_fn(out, data['metric']) 

        #loss = loss +  auxiliary_loss_function(percentages_optimize)

        loss.backward()

        if (b_idx + 1) % accumulation_steps == 0:               
            optimizer.step()
            optimizer.zero_grad()                      

        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])

        #print(f'{divisor_optimize=}')
        #print(f'{percentages_optimize=}')

        if b_idx%100 == 1:
            cpu_input_data = input_data.clone().detach()
            cpu_out = out.clone().detach()
            ls_num = 15
            metric_num = 1
            #visualize(predicted_metric[2],predicted_metric[2])
            visualize(test_dataset[ls_num]['landscape'].squeeze(),
                      cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
              left_title= f"Target Landscape", right_title = f"Predicted Landscape")
            visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                      cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
              left_title= f"Target Metric", right_title = f"Predicted Metric")

    return losses.avg, input_data, out, input_data_CCL


# run through all graph steps, try to optimize divisor/percentages (not possible bc not differentiable)
def optimize_one_landscape_full_graph_DEFUNCT(train_loader, input_data, model, optimizer, loss_fn, 
                           divisor_optimize, dif_scale, percentages_optimize, accumulation_steps=1, device='cuda'):

    losses = AverageMeter()
    model = model.to(device)             
    model.eval()

    if accumulation_steps > 1:
        optimizer.zero_grad()

    #divisor_optimize = random.randrange(50, 200, 1) / 10
    #divisor_optimize = np.random.normal(divisor_optimize, dif_scale)
    divisor_optimize = torch.normal(divisor_optimize, dif_scale)
    #percentages_optimize_prep = percentages_optimize + np.random.normal([0., 0., 0. ,0. ,0.], dif_scale) #np.repeat(random.random(), n_classes)
    percentages_optimize_prep = percentages_optimize + torch.normal(torch.tensor([0., 0., 0. ,0. ,0.]), dif_scale*30) #np.repeat(random.random(), n_classes)
    #percentages_optimize_prep = np.repeat(random.random(), n_classes)
    #percentages_optimize = percentages_optimize_prep / np.sum(percentages_optimize_prep)
    percentages_optimize = percentages_optimize_prep / torch.sum(percentages_optimize_prep)

    # 1. identify patches        
    input_data, num_patches = connected_component_labeling(input_data[0,0,:,:].detach().cpu().numpy(), 
                                                           divisor_optimize)


    # 2. Create an empty graph
    map_graph = nx.Graph()

    # 3. Add nodes representing regions
    unique_regions = np.unique(input_data)
    map_graph.add_nodes_from(unique_regions)

    # 4. label nodes ##
    node_labels_ordered = node_label_preparator(num_patches, percentages_optimize.detach().cpu().numpy())

    # 5. Give nodes random numbers    
    numbered_graph = digitize_nodes(map_graph, node_labels_ordered)

    # print(numbered_graph.nodes[5.]['number'])

    #print(numbered_graph.nodes)
    # 6. return nodes to array
    #print(f'{np.unique(input_data)=}')

    input_data = graph_to_array(input_data, numbered_graph.nodes)
    #print(f'{np.unique(input_data)=}')

    input_data = torch.tensor(input_data)
    input_data = input_data.unsqueeze(0).unsqueeze(0).to(device) 
    print(f'{input_data.shape=}')


    #tk0 = tqdm(train_loader, total=len(train_loader))
    tk0 = tqdm(train_loader, total=1)
    for b_idx, data in enumerate(tk0):   

        if accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()

        data['metric'] = data['metric'].to(device)
        print(data['metric'].shape)
        print(data['metric'].shape)
        out  = model(input_data)     
        print(f'{out.shape=}')
        loss = loss_fn(out, data['metric']) 

        #loss = loss +  auxiliary_loss_function(percentages_optimize)

        loss.backward()

        if (b_idx + 1) % accumulation_steps == 0:               
            optimizer.step()
            optimizer.zero_grad()                      

        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])

        print(f'{divisor_optimize=}')
        print(f'{percentages_optimize=}')

        if b_idx%100 == 1:
            cpu_input_data = input_data.clone().detach()
            cpu_out = out.clone().detach()
            ls_num = 15
            metric_num = 1
            #visualize(predicted_metric[2],predicted_metric[2])
            visualize(test_dataset[ls_num]['landscape'].squeeze(),
                      cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
              left_title= f"Target Landscape", right_title = f"Predicted Landscape")
            visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                      cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
              left_title= f"Target Metric", right_title = f"Predicted Metric")

    return losses.avg, input_data, out, divisor_optimize, percentages_optimize, dif_scale

# reversal otpimization only utilising some of the metric output channels, i.e. the first 35.
def optimize_one_landscape_n_metrics(train_loader, input_data, model, optimizer, loss_fn, accumulation_steps=1, device='cuda'):

    losses = AverageMeter()
    model = model.to(device)        
    input_data = input_data.to(device)
    model.eval()

    if accumulation_steps > 1: 
        optimizer.zero_grad()
    #tk0 = tqdm(train_loader, total=len(train_loader))
    tk0 = tqdm(train_loader, total=1)
    for b_idx, data in enumerate(tk0):   

        if accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()

        #print(f'{input_data.shape=}')
        data['metric'] = data['metric'].to(device)
        out  = model(input_data)            
        #print(data['metric'].shape)
        #loss = loss_fn(out, data['metric'])            
        loss = loss_fn(out[0:35,3,:,:], data['metric'][0:35,3,:,:])           
        loss.backward()
        #print(f'{out.shape=}')        

        if (b_idx + 1) % accumulation_steps == 0:               
            optimizer.step()
            optimizer.zero_grad()                      

        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])

        if b_idx%100 == 1:
            cpu_input_data = input_data.clone().detach()
            cpu_out = out.clone().detach()
            ls_num = 15
            metric_num = 1
            #visualize(predicted_metric[2],predicted_metric[2])
            visualize(test_dataset[ls_num]['landscape'].squeeze(),
                      cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
              left_title= f"Target Landscape", right_title = f"Predicted Landscape")
            visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                      cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
              left_title= f"Target Metric", right_title = f"Predicted Metric")

    return losses.avg, input_data, out


# https://chat.openai.com/share/b97629f3-db87-4f35-99c9-2b49ded55c3b
# https://chat.openai.com/c/432360ff-095c-4d07-8293-217c51983eb1

#import torch

def apply_integer_constraints(landscape_values, min_class, max_class):
    rounded_values = torch.round(landscape_values)
    clamped_values = torch.clamp(rounded_values, min_class, max_class)
    return clamped_values

#https://chat.openai.com/share/b97629f3-db87-4f35-99c9-2b49ded55c3b
# https://chat.openai.com/c/432360ff-095c-4d07-8293-217c51983eb1

#äimport torch

#import torch

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




def optimize_one_landscape_int_constraints(train_loader, input_data, model, optimizer, loss_fn, epoch, accumulation_steps=1, device='cuda'):

    #losses = AverageMeter()
    model = model.to(device)        
    input_data = input_data.to(device)
    model.eval()
    input_data_constrained = input_data.clone()
    
    if accumulation_steps > 1: 
        optimizer.zero_grad()
    tk0 = tqdm(train_loader, total=1)
    for b_idx, data in enumerate(tk0):   

        if accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()

        data['metric'] = data['metric'].to(device)
        out  = model(input_data)            
        
        # skip some metrics
        out_cat = torch.cat((out[:36], out[37:45], out[46:50], out[51:]), dim=0)
        target_cat = torch.cat((data['metric'][:36], data['metric'][37:45], data['metric'][46:50], data['metric'][51:]), dim=0)
        loss = loss_fn(out_cat, target_cat) 
        
        #loss = loss_fn(out, data['metric'])           
        loss.backward()
        
        #if epoch%50==1:
        #if epoch%5==1:
        input_data.data = apply_integer_constraints(input_data.data, 1, 5)
        #input_data_constrained = input_data.clone()
        

        if (b_idx + 1) % accumulation_steps == 0:               
            optimizer.step()
            optimizer.zero_grad()    
            
        #torch.round(input_data)


        #losses.update(loss.item(), train_loader.batch_size)
        #tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])
        tk0.set_postfix(loss=loss, learning_rate=optimizer.param_groups[0]['lr'])
        '''
        if b_idx%100 == 1:
            cpu_input_data = input_data.clone().detach()
            cpu_out = out.clone().detach()
            ls_num = 15
            metric_num = 1
            visualize(test_dataset[ls_num]['landscape'].squeeze(),
                      cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
              left_title= f"Target Landscape", right_title = f"Predicted Landscape")
            visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                      cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
              left_title= f"Target Metric", right_title = f"Predicted Metric")
              
        '''
        
    #return losses.avg, input_data, out#, input_data_constrained
    return loss.item(), input_data, out#, input_data_constrained

    #return loss, input_data, out
    
    
# outside of the loop:
#target_metric = train_loader/train_dataset[0]['metric'].to(device)
    
#with original landscape
def optimize_one_landscape_exactly(target_metric, input_data, model, optimizer, loss_fn, device='cuda'):

    #losses = AverageMeter()
    model = model.to(device)        
    input_data = input_data.to(device)
    model.eval()

    #if accumulation_steps > 1: 
    
    # Here needed? probably not.
    optimizer.zero_grad()
    #tk0 = tqdm(train_loader, total=1)
    #for b_idx, data in enumerate(tk0):   

    #if accumulation_steps == 1 and b_idx == 0:
    #    optimizer.zero_grad()

    #data['metric'] = data['metric'].to(device)
    out  = model(input_data)            
    #loss = loss_fn(out, data['metric'])  
    # skip certain metrics, i.e. 45 and 50 for loss calculation
    # do this before the thing?
    out_cat = torch.cat((out[:36], out[37:45], out[46:50], out[51:]), dim=0)
    target_cat = torch.cat((target_metric[:36], target_metric[37:45], target_metric[46:50], target_metric[51:]), dim=0)
    
    #target_cat = torch.cat((data['metric'][:36], data['metric'][37:45], data['metric'][46:50], data['metric'][51:]), dim=0)
    loss = loss_fn(out_cat, target_cat)  
    #loss = loss_fn(out, target_metric)    

    #loss = loss_fn(out_cat[:][:][49:79, 49:79], target_cat[:][:][49:79, 49:79])    
    #torch.cat((input_tensor[:45], input_tensor[46:50], input_tensor[51:]), dim=0)
    loss.backward()

    #if (b_idx + 1) % accumulation_steps == 0:               
    optimizer.step()
    optimizer.zero_grad()                      

    #losses.update(loss.item(), train_loader.batch_size)
    #tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])
    #tk0.set_postfix(loss=loss, learning_rate=optimizer.param_groups[0]['lr'])

    if b_idx%100 == 1:
        cpu_input_data = input_data.clone().detach()
        cpu_out = out.clone().detach()
        ls_num = 15
        metric_num = 1
        visualize(test_dataset[ls_num]['landscape'].squeeze(),
                  cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
          left_title= f"Target Landscape", right_title = f"Predicted Landscape")
        visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                  cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
          left_title= f"Target Metric", right_title = f"Predicted Metric")

    #return losses.avg, input_data, out
    return loss, input_data, out




#with original landscape
def optimize_one_landscape_quant(target_metric, input_data, model, optimizer, loss_fn, accumulation_steps=1, device='cpu'):

    #losses = AverageMeter()
    model = model.to(device)        
    input_data = input_data.to(device)
    model.eval()

    #if accumulation_steps > 1: 
    #    optimizer.zero_grad()
    #tk0 = tqdm(train_loader, total=1)
    #for b_idx, data in enumerate(tk0):   

    #if accumulation_steps == 1 and b_idx == 0:
    optimizer.zero_grad()

    target_metric = target_metric.to(device)
       
        
    out  = model(input_data)            
    #loss = loss_fn(out, data['metric'])  
    # skip certain metrics, i.e. 45 and 50 for loss calculation
    #out_cat = torch.cat((out[:36], out[37:45], out[46:50], out[51:]), dim=0)
    #target_cat = torch.cat((data['metric'][:36], data['metric'][37:45], data['metric'][46:50], data['metric'][51:]), dim=0)
    loss = loss_fn(out, target_metric)    
    #loss = loss_fn(out_cat[:][:][49:79, 49:79], target_cat[:][:][49:79, 49:79])    
    #torch.cat((input_tensor[:45], input_tensor[46:50], input_tensor[51:]), dim=0)
    loss.backward()

    #if (b_idx + 1) % accumulation_steps == 0:               
    optimizer.step()
    optimizer.zero_grad()                      

    #losses.update(loss.item(), train_loader.batch_size)
    #tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])
    #tk0.set_postfix(loss=loss, learning_rate=optimizer.param_groups[0]['lr'])

    if b_idx%100 == 1:
        cpu_input_data = input_data.clone().detach()
        cpu_out = out.clone().detach()
        ls_num = 15
        metric_num = 1
        visualize(test_dataset[ls_num]['landscape'].squeeze(),
                  cpu_input_data.cpu().numpy()[ls_num,0,:,:], 
          left_title= f"Target Landscape", right_title = f"Predicted Landscape")
        visualize(test_dataset[ls_num]['metric'][metric_num].squeeze(),
                  cpu_out.cpu().numpy()[ls_num,metric_num,:,:], 
          left_title= f"Target Metric", right_title = f"Predicted Metric")

    #return losses.avg, input_data, out
    return loss, input_data, out