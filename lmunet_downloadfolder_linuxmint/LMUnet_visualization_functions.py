import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
import numpy as np



def matplotlib_imshow(img, one_channel=False):
    fig,ax = plt.subplots(figsize=(8,8))
    ax.imshow(img.permute(1,2,0).numpy())

def visualize(landscape, metric, left_title = "Landscape", right_title = "Metric", suptitle = None, vmin=None, vmax=None, savefilepath=None):
    """PLot two 2D numpy arrays next to each other."""
    
    columns = 2
    rows = 1
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle(suptitle)
    
    # Modify the colormap for discrete colors
    #num_colors = 5  # Adjust the number of desired colors
    if torch.is_tensor(landscape):
        landscape = landscape.detach().cpu().numpy()# force=True doesn't work for some reason
    #num_colors = int(landscape.max().item())  # Adjust the number of desired colors
    #colormap = plt.cm.get_cmap('gray', num_colors)
    #new_cmap = ListedColormap(colormap(np.linspace(0, 1, num_colors)))
    
    fig.add_subplot(rows, columns, 1)
    plt.title(left_title)
    #plt.imshow(landscape, cmap=new_cmap, vmin=vmin, vmax=vmax)
   
    
    

    #landscape, metric = landscapes['landscape'], landscapes['metric']
    #plt.imshow(landscape.transpose(1,2,0), vmin=0, vmax=1)
    plt.imshow(landscape, cmap='gray', vmin=vmin, vmax=vmax)
    #plt.colorbar(ticks=np.arange(num_colors+1), fraction=0.046, pad=0.04)

    plt.colorbar(fraction=0.046, pad=0.04)
    fig.add_subplot(rows, columns, 2)
    plt.title(right_title)
    #plt.suptitle(right_st)
    #plt.title("Metric")
    if torch.is_tensor(metric):
        metric = metric.detach().cpu().numpy() #force=True doesn't work for some reason
    plt.imshow(metric, cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    if savefilepath:
        plt.savefig(fname=savefilepath)
    
    plt.show()    
    
    
# This combination (and values near to these) seems to "magically" work for me to keep the colorbar scaled to the plot, no matter what size the display.
# plt.colorbar(im,fraction=0.046, pad=0.04)
# It also does not require sharing the axis which can get the plot out of square.
# https://stackoverflow.com/a/26720422
