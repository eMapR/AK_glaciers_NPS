import torch
import rasterio
from barbar import Bar
import os
import numpy as np
from glob import glob

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, num_classes, transform=None):
        '''
        Args:
            root_dir (string): Directory with all of the images
            transform (calllable, optional): Optional transforms that can be 
                applied to the images. 
        '''
        # Initalize the class attributes
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.paths = None
        
        # Get the file names
        self.__get_file_list()
        
        return
    
    def __get_file_list(self):
        '''Get a list of the tensors in the target directory.'''
        
        # Glob all of the tiffs in the root dir
        self.paths = glob(self.root_dir + "//*.tif")
        
        if len(self.paths) == 0:
            raise ValueError("Dataset found no `.pt` files in specified directory.")
        
        return None

    def __getitem__(self, idx):
        '''
        Args:
            idx (int): Index of tensor to retrieve
        
        Returns:
            torch.tensor: DL Tensor [row, col, band]
        
        '''  
        # if the idx provided is a tensor convert it to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load in the image
        tiff_path = os.path.join(self.root_dir, self.paths[idx])
        
        #Read the Multi-band raster values into a 3D array
        #Array structure ['band', 'row number', 'column num']
        with rasterio.open(tiff_path, 'r') as ds:
            image_data = ds.read()
    
        # This corresponds to [channel, row number, column number]
        tensor_2d = torch.Tensor(image_data)
            
        # Apply the transform if needed 
        if self.transform is not None: 
            tensor_2d = self.transform(tensor_2d)
            
        #
            
        # create the input tensor and the target labels
        x = tensor_2d[0:-1,:,:]
        y = tensor_2d[-1:,:,:]
        
        return x, y.squeeze(0).long()

    def __len__(self):
        return len(self.paths)
    
if __name__ == "__main__":
    
    # Path to the tiffs
    tiff_path = "E:\\glacier_data\\train_128x128_chips"
    
    # Define some stuff
    num_channels = 9
    
    # Load teh dataset
    dataset = Dataset(tiff_path, 3)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=16,
        shuffle=False
    )

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    print("Initiating processing...")
    for i, data in enumerate(Bar(loader)):

        # get the inputs
        inputs, labels = data
        
        # shape (batch_size, 3, height, width)
        numpy_image = inputs.numpy()
        
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)
        
    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    np.set_printoptions(suppress=True)
    pop_mean_out = np.array(pop_mean).mean(axis=0).round(decimals=4)
    pop_std0_out = np.array(pop_std0).mean(axis=0).round(decimals=4)
    pop_std1_out = np.array(pop_std1).mean(axis=0).round(decimals=4)
    
    print("\nProgram Completed\n")