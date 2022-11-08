import torch
import rasterio
from glob import glob
from tqdm import tqdm

def load_geotiff_as_array(geotiff_path, chip_size):
    '''
    Returns an array with the structure: [row number, column number, band, year]
    '''
    #Read the Multi-band raster values into a 3D array
    #Array structure ['band', 'row number', 'column num']
    with rasterio.open(geotiff_path, 'r') as ds:
        image_data = ds.read()
    
    # This corresponds to [channel, row number, column number]
    reshaped = image_data[:-1, 0:chip_size, 0:chip_size]
    
    return torch.Tensor(reshaped).reshape(10, chip_size*chip_size).double()

if __name__ == '__main__':
    
    # Define the image chip size
    image_chip_size = 128
    
    # Defien a path to a dataset that needs to be laoded
    files = glob("C:\\temp_ssd_space\\cnn_train_set\\*tif")
    
    # Loop through the files
    num_records = len(files)
    mean = torch.Tensor([0,0,0,0,0,0,0,0,0,0]).double()
    std = torch.Tensor([0,0,0,0,0,0,0,0,0,0]).double()
    
    print("\nInitiating export:\n")
    for i, file in tqdm(enumerate(files)):

        if i % 10000 == 0:
            print('Currently at {i} of {tot}'.format(i=i, tot=num_records))

        # Load the image as a tensor
        pt_tensor = load_geotiff_as_array(file, image_chip_size)
        
        # Compute the mean and the standard deviations
        mean += pt_tensor.mean(1)
        std += pt_tensor.std(1)
    
    # Scale em
    mean = mean / num_records
    std = std / num_records
    
    print("\nProgram Complete.")