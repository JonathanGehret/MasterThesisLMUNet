# Need these?
# https://chat.openai.com/c/a2e876c6-b15b-40f1-aff6-bd2ef1dbb1f7
# https://chat.openai.com/share/dedff0d9-6300-43df-9a45-a9aaf88ee4ae


def add_edges(region_map, map_graph):
    ''' Connect adjacent patches by graph "edges"'''
    
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



def get_largest_neighbor(edged_graph, current_patch_number, patches_counts_dict):
    ''' Function to find the largest neighbor based on the edged graph'''
    
    # Initialize largest patch size
    largest_patch_size = 0
    
    # Find largest neighbor
    for neighbor in edged_graph.neighbors(current_patch_number):
        if neighbor not in patches_counts_dict:
            continue
        patch_size = patches_counts_dict[neighbor]
    
        # If patch is bigger, it becomes the largest patch
        if (patch_size > largest_patch_size) & (current_patch_number != 0):
            largest_patch_size = patch_size
            largest_patch_num = neighbor
    
    # Remove that patch from the input dict
    del patches_counts_dict[current_patch_number]
    
    # For some unforeseen errors
    try:
        print(patches_counts_dict[largest_patch_num])
    except NameError:       
        try:
            largest_patch_num = max(patches_counts_dict, key=patches_counts_dict.get)
        except:
            try:
                largest_patch_num = current_patch_number
                patch_size = patches_counts_dict[largest_patch_num]
                largest_patch_size = patch_size
            except:
                largest_patch_num = 9999
                largest_patch_size = 9999
                patches_counts_dict = {}
    
    
    return largest_patch_num, largest_patch_size, patches_counts_dict