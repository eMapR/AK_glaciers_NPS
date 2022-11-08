import numpy as np
import rasterio
import os
import torch
import matplotlib.pyplot as plt

def load_tensor(input_path):

    # Load the tensor
    tensor_2d = torch.load(input_path)
        
    # create the input tensor and the target labels
    x = tensor_2d[0:-1,:,:]
    y = tensor_2d[-1,:,:]
    
    
    # Sanity check on the labels
    y = torch.where(y == 3, 0, 1)
    
    return x.numpy(), y.long().numpy()

if __name__ == "__main__":
    
    # Load in a random geotiff
    path = "D://glacier_tensors//training_data//grid_tile_8116_train_chip_row304.0_col16.0_additional.pt"
    
    # Open it and return it in GDAL
    x, y = load_tensor(path)
    
    # Run a script to visual each of the channels
    for i in range(0, x.shape[0]):
        
        # Plot each of the bands
        fig = plt.figure()
        plt.imshow(x[i,:,:])
        
        print('i', x[i,:,:].min(), x[i,:,:].max())
        
    # Plot the label
    fig = plt.figure()
    plt.imshow(y)
    
    