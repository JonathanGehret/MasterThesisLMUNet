# Includes connected component algorithm and helper functions

import numpy as np
import torch
from scipy.optimize import minimize


# https://chat.openai.com/share/b9e5ba7c-b93e-4ba3-a456-0c3ff8337c40
# label exactly the same only if the same value, so for already digitized landscape
def connected_component_labeling(image, divisor):

    # Create a label tensor with the same shape as the input image, initially filled with zeros
    labels = np.zeros_like(image)

    # Define the current label counter
    current_label = 1

    # Compute the dynamic threshold based on the pixel value distribution
    # Adjust the factor according to your specific application 
    threshold = np.std(image) / divisor  
    
    # Traverse through the image and perform connected-component labeling
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # If the pixel is foreground and not yet labeled
            if labels[i, j] == 0:                
                # Assign the current label to the pixel
                labels[i, j] = current_label

                # Apply DFS to propagate the label to connected neighbors
                stack = [(i, j)]
                while len(stack) > 0:
                    row, col = stack.pop()
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            if abs(dx) + abs(dy) != 1:  # Skip diagonals (https://chat.openai.com/share/2968679f-cbfe-4b48-8c7f-cc48ce16b2db)
                                continue
                            new_row, new_col = row + dx, col + dy
                            if (
                                0 <= new_row < image.shape[0] and
                                0 <= new_col < image.shape[1] and
                                image[new_row, new_col] > 0 and
                                labels[new_row, new_col] == 0 and
                                abs(image[new_row, new_col] - image[row, col]) <= threshold
                            ):
                                labels[new_row, new_col] = current_label
                                stack.append((new_row, new_col))

                # Increment the current label for the next connected component
                #print(current_label)
                current_label += 1

    # Convert the labels array to a PyTorch tensor
    #labels_tensor = torch.from_numpy(labels)

    return labels, current_label - 1

# https://chat.openai.com/share/b608170a-6398-4ac7-b358-d877ddd14a29

# Find the value of the largest patch and create a binary mask with that patch
def find_largest_patch(CCL_landscape):
    unique_labels, label_counts = np.unique(CCL_landscape, return_counts=True)

    # Find the largest area and its corresponding label
    largest_area = np.max(label_counts)
    largest_area_label = unique_labels[np.argmax(label_counts)]
    print(largest_area_label)

    binary_mask = np.where(CCL_landscape == largest_area_label, 1, 0)
    
    return binary_mask, largest_area_label, largest_area
