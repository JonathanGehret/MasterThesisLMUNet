# generate intial percentages for n calsses and total_nodes
#n_classes = 5
#num_nodes = 10

#percentages_optimize = np.repeat(1/n_classes, n_classes)


def node_label_preparator(num_nodes, percentages_optimize): 
    
    # percentages_optimize = np.repeat(1/n_classes, n_classes)

    nodes_per_number = percentages_optimize * num_nodes

    print(nodes_per_number)
    
    node_labels_ordered = np.repeat([i + 1 for i in range(len(percentages_optimize))], nodes_per_number.astype(int))

    np.random.shuffle(node_labels_ordered)
    #print(node_labels_ordered)
    
    return(node_labels_ordered)


import torch
from torch.distributions import Categorical

def patch_label_preparator(num_patches, percentages_optimize):
    # Create a categorical distribution based on the percentages_optimize tensor
    #dist = Categorical(percentages_optimize)

    # Sample the number of nodes per class
    patches_per_number = torch.tensor(num_patches* percentages_optimize) # dist.sample(torch.Size([num_patches]))
    
    print(patches_per_number)
    # Generate node labels ordered by the sampled number of nodes per class
    patch_labels_ordered = torch.cat([torch.full((n,), i+1, dtype=torch.long) for i, n in enumerate(patches_per_number)])

    # Shuffle the node labels
    patch_labels_ordered = patch_labels_ordered[torch.randperm(num_patches)]

    return patch_labels_ordered


import torch
from torch.distributions import Categorical

def patch_label_preparator(num_patches, percentages_optimize):
    # Create a categorical distribution based on the percentages_optimize tensor
    dist = Categorical(percentages_optimize)

    # Sample the number of nodes per class
    patches_per_number = dist.sample(torch.Size([num_patches]))
    print(patches_per_number)

    # Generate node labels ordered by the sampled number of nodes per class
    patch_labels_ordered = torch.cat([torch.full((n,), i+1, dtype=torch.long) for i, n in enumerate(patches_per_number)])

    # Shuffle the node labels
    patch_labels_ordered = patch_labels_ordered[torch.randperm(num_patches)]

    return patch_labels_ordered


# https://chat.openai.com/c/a2e876c6-b15b-40f1-aff6-bd2ef1dbb1f7
# https://chat.openai.com/share/dedff0d9-6300-43df-9a45-a9aaf88ee4ae

import networkx as nx
import matplotlib.pyplot as plt

def digitize_nodes(graph, numbers):
    total_nodes = graph.number_of_nodes()
    nodes = list(graph.nodes())
    #num_numbers = len(numbers)
    #print(total_nodes)
    #print(len(numbers))

    for i in range(total_nodes-5):
    #for i in range(len(numbers)):
        #print(i)
        node = nodes[i]
        #number_index = int(i / (total_nodes / num_numbers))
        number = numbers[i]
        graph.nodes[node]['number'] = number

    return graph

import torch
#import torch_geometric as torch_geom
from torch_geometric.data import Data

def digitize_nodes(graph, numbers):
    node_labels = torch.tensor(numbers, dtype=torch.float32)
    graph.y = node_labels
    return graph

def digitize_nodes(graph, numbers):
    graph.ndata['number'] = numbers
    return graph


def graph_to_array(region_map, numbered_graph):
    for node in numbered_graph:
        region_map[region_map == node] = numbered_graph.nodes[node]['number']
    return(region_map)


def graph_to_array(region_map, numbered_graph_nodes):
    #print(numbered_graph_nodes[5]['number'])

    for node in range(len(numbered_graph_nodes)-5):
        #print(node)
        #print(numbered_graph_nodes[node])
        #example = numbered_graph_nodes[node]#['number']
        region_map[region_map == node] = numbered_graph_nodes[node]['number']
        #region_map[region_map == node] = example['number']
    region_map[region_map > 5] = 3 # helper line for the length problem
    return(region_map)

def dict_to_array(region_map, numbered_graph_nodes):
    #print(numbered_graph_nodes[5]['number'])

    for node in numbered_graph_nodes:
        #print(node)
        #print(numbered_graph_nodes[node])
        #example = numbered_graph_nodes[node]#['number']
        region_map[region_map == node] = numbered_graph_nodes[node]
        #region_map[region_map == node] = example['number']
    #region_map[region_map > 5] = 3 # helper line for the length problem
    return(region_map)

def dict_to_array(region_map, unique_dict):
    #print(numbered_graph_nodes[5]['number'])

    for i in unique_dict:
        #print(node)
        #print(numbered_graph_nodes[node])
        #example = numbered_graph_nodes[node]#['number']
        #print(i)
        #print(unique_dict[i])
        region_map[region_map == i] = unique_dict[i]
       # print(np.unique(region_map))
        #region_map[region_map == node] = example['number']
    #region_map[region_map > 5] = 3 # helper line for the length problem
    return(region_map)


def add_edges(region_map, map_graph):
    
    # Convert the numpy array into a binary mask
    binary_mask = (region_map != 0).astype(int)
    # Iterate through each pixel in the binary mask
    height, width = binary_mask.shape
    for i in range(height):
        for j in range(width):
            if binary_mask[i, j] == 1:
                current_region = region_map[i, j]

                # Check neighboring pixels
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj

                    # Add edge if the neighboring pixel is within the bounds and belongs to a different region
                    if 0 <= ni < height and 0 <= nj < width and binary_mask[ni, nj] == 1 and region_map[ni, nj] != current_region:
                        neighbor_region = region_map[ni, nj]
                        map_graph.add_edge(current_region, neighbor_region)
    
    return map_graph


import numpy as np

# Label nodes with numbers of certain percentage
def label_nodes_percentages(graph, label_array, percentages):
    n = len(graph)  # Total number of nodes
    target_counts = np.round(np.array(percentages) * n).astype(int)  # Calculate target count for each label
    print(f'{target_counts=}')
    current_counts = {label: 0 for label in label_array}  # Dictionary to track current count of each label
    print(current_counts)
    labeled_nodes = {}  # Dictionary to store labeled nodes

    while len(labeled_nodes) < n:
        for node in graph:
            if node in labeled_nodes:
                continue

            count_ratios = []
            for label in label_array:
                count_ratio = current_counts[label] / target_counts[label-1] if target_counts[label-1] > 0 else float('inf')
                count_ratios.append(count_ratio)

            min_ratio = min(count_ratios)
            min_labels = [label for label, ratio in zip(label_array, count_ratios) if ratio == min_ratio]

            if len(min_labels) == 1:
                selected_label = min_labels[0]
            else:
                selected_label = min(min_labels, key=lambda label: current_counts[label])

            labeled_nodes[node] = selected_label
            current_counts[selected_label] += 1

    return labeled_nodes


import numpy as np

# no two neighboring nods have the same color
def label_nodes_no_touch(graph, label_array, percentages):
    n = len(graph)  # Total number of nodes
    target_counts = np.round(np.array(percentages) * n).astype(int)  # Calculate target count for each label
    #print(f'{target_counts=}')
    current_counts = {label: 0 for label in label_array}  # Dictionary to track current count of each label
    #print(current_counts)
    labeled_nodes = {}  # Dictionary to store labeled nodes

    while len(labeled_nodes) < n:
        for node in graph:
            if node in labeled_nodes:
                continue

            neighbor_labels = [labeled_nodes[neighbor] for neighbor in graph[node] if neighbor in labeled_nodes]
            available_labels = [label for label in label_array if label not in neighbor_labels]

            if len(available_labels) == 0:
                continue

            selected_label = min(available_labels, key=lambda label: current_counts[label])

            labeled_nodes[node] = selected_label
            current_counts[selected_label] += 1

    return labeled_nodes


import numpy as np
from scipy.ndimage import label

# count how many patches there are per class
def count_connected_regions(array):
    unique_values = np.unique(array)
    result = {}

    for value in unique_values:
        binary_array = np.where(array == value, 1, 0)
        labeled_array, num_regions = label(binary_array)
        result[value] = num_regions

    return result



def get_largest_neighbor(edged_graph, current_patch_number, patches_counts_dict):
    largest_patch_size = 0
    #largest_patch_num = 0
    #labeled_patch_numbers.append(current_patch_number)
        
    for neighbor in edged_graph.neighbors(current_patch_number):
        #print(f'{neighbor=}')
        if neighbor not in patches_counts_dict:
            #print(neighbor)
            continue
        #if neighbor == 0:
        #    continue
        patch_size = patches_counts_dict[neighbor]
        #print(neighbor, patch_size)
        if (patch_size > largest_patch_size) & (current_patch_number != 0):
            largest_patch_size = patch_size
            largest_patch_num = neighbor
            
    del patches_counts_dict[current_patch_number]
    #unlabeled_patch_numbers.remove(current_patch_number)
    
    # Ugly, rewrite better?
    try:
        print(patches_counts_dict[largest_patch_num])
    except NameError:       
        try:
            largest_patch_num = max(patches_counts_dict, key=patches_counts_dict.get)
            #del patches_counts_dict[largest_patch_num]
        except:
            largest_patch_num = current_patch_number
            largest_patch_size = patch_size
            
    #if largest_patch_num not in locals():
    #    largest_patch_num = max(patches_counts_dict, key=patches_counts_dict.get)
    
    #del patches_counts_dict[current_patch_number]
    #unlabeled_patch_numbers.remove(current_patch_number)

    return largest_patch_num, largest_patch_size, patches_counts_dict # could also only return the list, but I think it's fine


def get_all_neighbors(edged_graph, current_neighbors, patches_counts_dict):
    largest_patch_size = 0
    #largest_patch_num = 0
    #labeled_patch_numbers.append(current_patch_number)
    
    new_neighbors = []
    
    for previous_patch in current_neighbors:
        
        '''
        if previous_patch not in patches_counts_dict:
            print('not in dict')
            #current_patch_number = max(patches_counts_dict_copy, key=patches_counts_dict_copy.get)

            continue
        '''
        
        all_new_neighbors = edged_graph.neighbors(previous_patch)
        #print(all_new_neighbors)
        for each_new_neighbor in all_new_neighbors: 
            #print(each_new_neighbor)

            if each_new_neighbor not in patches_counts_dict:
                #print(neighbor)
                continue            
            else:
                new_neighbors.append(each_new_neighbor)
                del patches_counts_dict[each_new_neighbor]
        
        #print(f'{neighbor=}')
    
    #unlabeled_patch_numbers.remove(current_patch_number)
    '''
    try:
        print(patches_counts_dict[largest_patch_num])
    except NameError:        
        largest_patch_num = max(patches_counts_dict, key=patches_counts_dict.get)
        #del patches_counts_dict[largest_patch_num]
    '''       
    #if largest_patch_num not in locals():
    #    largest_patch_num = max(patches_counts_dict, key=patches_counts_dict.get)
    
    #del patches_counts_dict[current_patch_number]
    #unlabeled_patch_numbers.remove(current_patch_number)

    return new_neighbors, patches_counts_dict # could also only return the list, but I think it's fine