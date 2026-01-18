import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

# for discrete colormap
#https://chat.openai.com/share/57351dcf-784b-45e4-a1fe-3bd6e1c1cad9

def matplotlib_imshow(img, one_channel=False):
    """Display an image using Matplotlib."""
    fig,ax = plt.subplots(figsize=(8,8))
    ax.imshow(img.permute(1,2,0).numpy())

def visualize(landscape, metric, left_title="Landscape", right_title="Metric", suptitle=None, savefilepath=None, left_disc=None, right_disc=None):
    """PLot two 2D numpy arrays or torch tensors next to each other.
    
    Parameters:
    - landscape: The left 2D array (numpy or torch tensor).
    - metric: The right 2D array (numpy or torch tensor).
    - left_title: Title for the left plot.
    - right_title: Title for the right plot.
    - suptitle: Title for the entire figure.
    - savefilepath: If provided, save the figure to this path.
    - left_disc: If True, use a discrete color map for the left plot.
    - right_disc: If True, use a discrete color map for the right plot.
    """
    
    columns = 2
    rows = 1
    fig = plt.figure(figsize=(8, 8))
    #fig, axes = plt.subplots(1, 2, figsize=(8, 8))

    plt.suptitle(suptitle)
    
    # Convert to numpy array if necessary
    if torch.is_tensor(landscape):
        landscape = landscape.detach().cpu().numpy()# force=True doesn't work for some reason

    fig.add_subplot(rows, columns, 1)
    plt.title(left_title)
   
    # Modify the colormap for discrete colors
    if left_disc:       
        discrete_colorbar(landscape)        
    else:        
        plt.imshow(landscape, cmap='gray')
        plt.colorbar(fraction=0.046, pad=0.04)
        
    #if metric:

    fig.add_subplot(rows, columns, 2)    
    plt.title(right_title)

    if torch.is_tensor(metric):
        metric = metric.detach().cpu().numpy() #force=True doesn't work for some reason

    # Modify the colormap for discrete colors
    if right_disc:       
        discrete_colorbar(metric)        
    else:        
        plt.imshow(metric, cmap='gray')
        plt.colorbar(fraction=0.046, pad=0.04)


        
    # Create some space between the two images
    plt.subplots_adjust(wspace=0.25)  # Adjust the values as needed

    if savefilepath:
        plt.savefig(fname=savefilepath)
            
    plt.show()    
    
    
# "This combination (and values near to these) seems to "magically" work for me to keep the colorbar scaled to the plot, no matter what size the display.
# plt.colorbar(im,fraction=0.046, pad=0.04)
# It also does not require sharing the axis which can get the plot out of square.
# https://stackoverflow.com/a/26720422


def discrete_colorbar(image):
    """ Create a discrete Colorbar for an image and plot it."""    
    # Make the ticklabels appear in the middle
    num_classes = image.max().astype(np.int64)# + np.abs(image.min().astype(np.int64) )
    color_values = np.linspace(0, 1, num_classes)
    tick_labels = np.arange(1 , num_classes + 1)
    boundaries = np.arange(1, num_classes + 2)
 
    grayscale_values_str = [str(i) for i in color_values]
    
    cmap = ListedColormap(grayscale_values_str)
    norm = BoundaryNorm(boundaries, cmap.N)

    plt.imshow(image, cmap=cmap, norm=norm)
    
    # Start with 1.5 because 0 is skipped.
    cbar = plt.colorbar(fraction=0.046, pad=0.04, cmap=cmap, ticks=np.arange(1.5, num_classes +1 , 1))
    cbar.set_ticklabels(tick_labels)
