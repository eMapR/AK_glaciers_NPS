import os
import sys
import glob
import pandas as pd 
import numpy as np 
import json
from osgeo import gdal
import rasterio 
import subprocess
import multiprocessing
import random
import time
import geopandas as gpd
import csv
import numpy as np
# from skimage.morphology import rectangle   # for Structuring Elements (e.g. disk, rectangle)
# from skimage.filters.rank import modal     # the puppy we want

# def create_binary_maps_and_filter(input_raster): 
# 	ds = gdal.Open(input_raster)
# 	arr = ds.GetRasterBand(1) #hardcoded for the first band- assume it only has one band if binary 
# 	print(arr)
# 	# # Same seed for directly comparable results
# 	# np.random.seed(329)

# 	# # Sample array/image
# 	# arr = np.random.randint(low=0, high=10, size=(6, 8), dtype=np.uint8)

# 	# Run the filter with a 5x5 rectangular Structuring Element
# 	result = modal(arr,rectangle(5,5))
# 	print(result)
# 	# print(result)

# 	# array([[9, 2, 0, 0, 0, 2, 4, 5],
# 	#        [1, 1, 0, 0, 0, 2, 5, 2],
# 	#        [1, 1, 0, 5, 5, 5, 5, 5],
# 	#        [1, 1, 1, 5, 5, 5, 4, 5],
# 	#        [1, 1, 1, 1, 5, 5, 4, 5],
# 	#        [1, 1, 5, 5, 5, 5, 4, 4]], dtype=uint8)
def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')
def create_class_label(input_raster,static_raster,output_dir,thresh=3): 
	"""Read in a STEM classification, binarize it and then combine with RGI."""
	with rasterio.open(static_raster) as src1: 
		arr1 = src1.read(1) #get the last band as that is the class band 

		with rasterio.open(input_raster) as src2: 
			arr2 = src2.read(1) #get the last band as that is the class band 

			#binarize
			arr2=(arr2 > thresh).astype(int)
			#arr2[arr2>thresh]= 1 
			print(arr2)
			try: 
				boundary = arr1 * arr2

			except ValueError as e: 
				#set to pad arr 2 to arr 1 but this should be more flexible 
				arr2 = to_shape(arr2,arr1.shape)
				boundary = arr1*arr2
				print(boundary)
				output_arr = arr1+boundary
				print(output_arr.shape)
		output_raster = os.path.join(output_dir,'2019_test_revised_raster_minus_RGI.tif')
	
		profile=src1.profile
		profile.update(
			driver='GTiff',
			compress='lzw',
			dtype='uint8'
			) 
		
	with rasterio.open(output_raster, 'w',**profile) as dst:
		print('writing')
		try: 
			dst.write(output_arr)

		except Exception as e: 
			dst.write(output_arr,indexes=1)
def main(output_dir):#input_grid,output_dir,full_study_area_vrt,**kwargs): 
	
		#run commands
		#perform a spatial join to get places where 512x512 grid intersects RGI bounds 
		#spatial_join(glacier_boundaries,train_grid)

		#add a bounding box that is used to snip out the partition 
		# grid_coords = add_bbox_col(gpd.read_file(grid),sort_col='label') #sort_col is an optional arg. Default is 'new_id'
		
		# start_time = time.time()
		
		# pool = multiprocessing.Pool(processes=20)
		# pool.map(run_cmd, create_grid_image_cmds(grid_coords,output_dir,full_study_area_vrt,reject_ids=kwargs.get('reject_ids')))  
		# pool.close()

		# print(f'Time elapsed for image chip extraction is: {((time.time() - start_time))/60} minutes')
	    create_class_label('/vol/v3/ben_ak/param_files_rgi/southern_region/output_files/models_2019_year_glacier_probabilities_no_low_class_vote.tif',
	    	"/vol/v3/ben_ak/raster_files/rgi/rgi_epsg_3338_clipped_US_bounds.tif",output_dir)

		
		
if __name__ == '__main__':

	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		output_dir = variables["output_dir"]

	main(output_dir)