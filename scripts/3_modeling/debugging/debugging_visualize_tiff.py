import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt

def load_image(input_path):

    # Load in the image
    tiff_path = os.path.join(input_path)
    
    #Read the Multi-band raster values into a 3D array
    #Array structure ['band', 'row number', 'column num']
    with rasterio.open(tiff_path, 'r') as ds:
        image_data = ds.read()
        
    image_data = image_data[0:-1, :, :]
    
    return image_data

def apply_normalization(input_array):
    
    # Define the tensors
    means = np.array([[[5.7279108e+02,  2.8725705e+02,  1.3425389e+02,  7.1344717e+03,
        1.5746520e+03,  3.0957052e+01,  3.0969534e+01, -5.7596784e+00,
        1.1013014e+03, 0]]])
    stds = np.array([[[387.84143,  524.1624 ,  378.68375, 4335.9365 , 2173.4917 ,
        228.33615,  228.33505,  127.43118,  489.14343, 1]]])
    
    means = np.moveaxis(means, [0,1,2], [2,1,0])
    stds = np.moveaxis(stds, [0,1,2], [2,1,0])
    
    return (input_array - means) / stds

if __name__ == "__main__":
    
    # Load in a random geotiff
    path = "E://glacier_data//128x128_full_study_area//grid_tile_8116_train_chip_row304.0_col16.0.tif" 
    
    # Open it and return it in GDAL
    array = load_image(path)
    
    # Apply normalization
    array = apply_normalization(array)
    
    # Run a script to visual each of the channels
    for i in range(0, array.shape[0]):
        
        # Plot each of the bands
        fig = plt.figure()
        plt.imshow(array[i,:,:])
        
        print('i', array[i,:,:].min(), array[i,:,:].max())
    
    
    