import torch
import os
from pathlib import Path
import torch.nn as nn
import rasterio
from glob import glob
from rasterio.windows import Window
import torchvision.transforms as transforms
from math import ceil
import numpy as np
from tqdm import tqdm

from crfseg import CRF

import utils.unet_models as UNets

class ProcessBigImages():

    def __init__(self, image_path, output_dir, output_name, chunk_size):
        
        # Set the class attributes
        self.image_path  = image_path
        self.output_dir = output_dir
        self.output_name = output_name
        self.chunk_size = chunk_size
        
        # Open the source image
        source = rasterio.open(self.image_path) 
        
        # Get and save the hight, width, and profile
        self.source_cols = source.width
        self.source_rows = source.height
        self.source_channels = source.count
        self.source_profile = source.profile
        
        # Get the number of elements to pad
        self.n_pad_cols  = self.chunk_size - (source.width % chunk_size)
        self.n_pad_rows  = self.chunk_size - (source.height % chunk_size)
        
        # Get the number of chunks to slice out
        self.n_chunks_x = ceil(source.width / float(self.chunk_size))
        self.n_chunks_y = ceil(source.height / float(self.chunk_size))
        
         # Save the number of patches in total generated
        self.num_patches = self.n_chunks_x * self.n_chunks_y

        return None
    
    def print_processing_info(self):
        print('Processing information:')
        print('    # source rows: ', self.source_rows)
        print('    # source cols: ', self.source_cols)
        print('       chunk size: ', self.chunk_size)
        print('       # x chunks: ', self.n_chunks_x)
        print('       # y chunks: ', self.n_chunks_y)
        print('# padding columns: ', self.n_pad_cols)
        print('   # padding rows: ', self.n_pad_rows)
        print('')
        return None
    
    def generate_prediction(self, network):
        
        # Turn off the autograd 
        with  torch.no_grad():
            
            # Format the image into patches that can be processed
            pred_patches = self.__generate_prediction_tiles(network)
            
            # Assemble the patches
            assembled = self.__assemble_chips(pred_patches)
               
            # Write out the layer
            self.__write_image(assembled)
            
            # Clear GPU memory cache
            torch.cuda.empty_cache()
       
        return None     
    
    def __generate_prediction_tiles(self, network):
        
        # Split the image into image chips
        patches = []

        # Loop through the rows
        for y in tqdm(range(0, int(self.n_chunks_y))):  
            
            # Loop through the columns
            for x in range(0, int(self.n_chunks_x)):  

                # Identify the start point for the current chunk
                x_offset = x * self.chunk_size
                y_offset = y * self.chunk_size
                    
                # Read in the raster
                with rasterio.open(self.image_path) as src:
                    image_array = src.read(window=Window(x_offset, y_offset, self.chunk_size, self.chunk_size))
                
                # Cast the input as a torch tensor
                # The indexing is to re-order the slope and elevation bands
                image_array  = torch.Tensor(image_array[:-1,:,:])
                
                 # Deal with row padding
                if image_array.shape[1] != self.chunk_size:
                    row_padding = torch.zeros(image_array.shape[0], self.n_pad_rows, image_array.shape[2])                    
                    image_array = torch.cat([image_array, row_padding], dim=1)
                
                # Deal with column padding
                if image_array.shape[2] != self.chunk_size:
                    column_padding = torch.zeros(image_array.shape[0], image_array.shape[1], self.n_pad_cols)
                    image_array = torch.cat([image_array, column_padding], dim=2)   
                
                # Create the prediction 
                prediction = self.__generate_prediction(image_array, network)

                # Append the prediction tile to the patches
                patches.append(prediction)  
                
        return patches
    
    def __generate_prediction(self, patch, network):
        
        # Normalize the tensor values
        means = torch.tensor([5.7279108e+02,  2.8725705e+02,  1.3425389e+02,  7.1344717e+03,
            1.5746520e+03,  3.0957052e+01,  3.0969534e+01, -5.7596784e+00,
            1.1013014e+03]).to("cuda:1")
        stds = torch.tensor([387.84143,  524.1624 ,  378.68375, 4335.9365 , 2173.4917 ,
            228.33615,  228.33505,  127.43118,  489.14343]).to("cuda:1")
        norm_transform = transforms.Normalize(mean=means, std=stds)
        
        # Create the predictor
        input_tensor = patch.to("cuda:1")
        
        # Normalize the values
        norm = norm_transform(input_tensor)
        
        # Get the prediction
        scores = network.forward(norm.unsqueeze(0)).squeeze(0)
        
        # Get the index of the mostly class (which corresponds to the class label)
        prediction = torch.argmax(scores, 0)
        
        # Format the image as a NumPy array
        prediction = prediction.cpu().numpy().astype(np.int8)
                
        return prediction
                    
    def __assemble_chips(self, input_list):
        
        # Iterate through the rows
        temp_list = []
        outer_list = []
        for i in range(0, self.n_chunks_y * self.n_chunks_x):
            
            # Append the image chip to the inner list
            temp_list.append(input_list[i].astype(np.int8))
            
            # Append the temporary list to the outer_list
            if ((i+1) % self.n_chunks_x) == 0:
                outer_list.append(temp_list)
                temp_list = []            
                
        # Combine the predictions into a single matrix
        # Indexing removes the padding
        combined = np.block(outer_list)

        return combined[0:self.source_rows, 0:self.source_cols]

    def __write_image(self, out_array):
        
        # Register GDAL format drivers and configuration options with a
        # context manager.
        with rasterio.Env():
        
            # And then change the band count to 1, set the
            # dtype to uint8, and specify LZW compression.
            self.source_profile.update(
                dtype=rasterio.uint8,
                count = 1,
                compress = 'lzw',
                nodata = 255
                )
        
            out_path = self.output_dir +'\\'+ self.output_name
            with rasterio.open(out_path, 'w', **self.source_profile) as dst:
                dst.write(out_array.astype(rasterio.uint8), 1)
            
        return None

if __name__ == '__main__':
    
    # Define the parameters
    time_series_dir = 'E:\\glacier_data\\demo_time_series'
    output_path_base = 'E:\\glacier_data\\predictions'
    chunk_size = 1024 + 512 
    
    print('Loading model...')
    net = torch.nn.Sequential(
        UNets.AttU_Net(9, 2),
#        CRF(n_spatial_dims=2)
        ).to("cuda:1")
    
    net.load_state_dict(torch.load('E:\\glacier_data\\model_weights\\AttUNet_binary_crf__0_0001.pth'))
    net.eval()
    print('...model loaded.')
    
    # Get all of the folders with a time-series of tiffs
    folders_containing_ts = glob(time_series_dir+"\\*")
    
    # Loop through all of the folders with time-series
    print('\nProcessing inputs...')
    for ts_dir in folders_containing_ts:
        
        # Generate the output directory
        new_dir_name = ts_dir.split('\\')[-1]
        current_output_dir = output_path_base+'\\'+new_dir_name
        
        # Create the directory if it doesn't exist
        path = Path(current_output_dir)
        if path.is_dir():
            pass
        else:
            os.mkdir(current_output_dir)
        
        # Loop through the fikes in the directory
        files = glob(ts_dir + '\\*.tif')
        for i, file in enumerate(files):
                
            # Get information from the file path
            output_name = file.split('\\')[-1]
            
            # Create the objects
            processor = ProcessBigImages(file, current_output_dir, output_name, chunk_size)
            
            # Run the processing
            processor.generate_prediction(net)
        
    print('\nScript completed.')
#    
#    
#    
#    
#    
#    
    
    
    