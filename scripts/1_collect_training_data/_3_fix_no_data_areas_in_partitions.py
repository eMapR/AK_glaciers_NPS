import os
import sys
import glob
import pandas as pd 
import numpy as np 
import json
import rasterio 
import subprocess
import multiprocessing
import random
import time
from rasterio.windows import Window
import re
from functools import partial
import progressbar

def cast_dtype(x,dtype): 
	"""
	Helper function.
	"""
	if (dtype == 'float32') | (dtype == 'float64'): 
		return float(x)
	elif (dtype == 'int16') | (dtype == 'int32'): 
		return int(x)
	else: 
		print('Looks like your dtype is not float or int, please double check')

def make_masks(arr,band,ndata): 
	"""
	Helper function.
	"""
	mask1 = arr[band]<0

def read_raster_to_np_and_write_to_tif(input_raster,output_dir,topo_nodata= -9999,optical_nodata= -32768,class_nodata=None,fill_value=0,topo_band=-3,optical_band=0,class_band=9,optical_start=0,topo_start=5,**kwargs): #note that it will be a numpy array which is zero indexed 
	"""Read a vrt raster into np array."""
	output_filename = os.path.join(output_dir,f'{os.path.split(input_raster)[1][:-4]}_corrected.tif')
	print(output_filename)
	if not os.path.exists(output_filename): 
		with rasterio.open(input_raster) as src: #window works like Window(col_off, row_off, width, height)
			#this will be a numpy arr of the shape (num_bands,partition_size,partition_size)
			w = src.read()
			#make sure the nodata values used for constructing masks are the right dtype 
			topo_nodata = cast_dtype(topo_nodata,w.dtype)
			optical_nodata = cast_dtype(optical_nodata,w.dtype)

			if class_nodata: 
				class_nodata = cast_dtype(class_nodata,w.dtype)
			else: 
				class_nodata = optical_nodata

			#check if the defaults disagree with the input raster (now numpy) type 
			if (str(w.dtype).startswith('float')) & (str(type(fill_value)).startswith('int')): 
				print(f'There is a disagreement between the input raster type {w.dtype} and the default int value for ndata and fill. Stopping.')
				return None 
			#create the masks 

			#first do the class mask- this one will be applied last 
			class_mask = w[class_band] ==class_nodata
			
			#get the topographic data mask- there are some weird things happening with masking here at the edges so this is a bit messy
			topo_mask1 = w[topo_band]<0
			if kwargs.get('alt_topo_band'): 
				topo2 = kwargs.get('alt_topo_band')
				topo_mask2 = w[topo2]==(class_nodata*-1)-1 
				topo_mask3 = w[topo2]==topo_nodata
				topo_mask4 = w[topo2]==class_nodata

				topo_mask=np.logical_or.reduce((topo_mask1, topo_mask2, topo_mask3, topo_mask4))
			else: 
				print('processing with one topo mask')
				topo_mask=topo_mask1
			#get the optical mask 
			optical_mask = w[optical_band]==optical_nodata
			
			#apply the masks 
			#first mask the optical bands
			optical_arr=np.ma.array(w[optical_start:topo_start], mask=w[optical_start:topo_start]*optical_mask[np.newaxis,:,:]).filled(fill_value=fill_value)
			topo_arr=np.ma.array(w[topo_start:class_band], mask=w[topo_start:class_band]*topo_mask[np.newaxis,:,:]).filled(fill_value=fill_value) 
			
			#stack the masked pieces of the image back together 	
			out_arr=np.concatenate((optical_arr,topo_arr,w[class_band:]),axis=0)#[np.newaxis,:,:]),axis=0)

			#anywhere the class label (last band) is nodata make everything zero 
			out_arr=np.ma.array(out_arr,mask=out_arr*class_mask[np.newaxis,:,:]).filled(fill_value=fill_value)
			#get the metadata for writing 
			profile=src.profile
			
			#defaults to making a vrt- specify this arg to write to tif
			
			#make a few updates to the metadata before writing
			profile.update(
			driver='GTiff',
			compress='lzw') 
			
		with rasterio.open(output_filename, 'w',**profile) as dst:
			dst.write(out_arr)
	else: 
		print('That file already exists')
		return None 

def main(vrt_dir,**kwargs): 
		###########################################################
		#use this section for testing
		test_file = '/vol/v3/ben_ak/raster_files/neural_net/train_dev_multi_annual_revised_class_label/southern_region/train_256x256_partitions/train_pt_id_79945217_partition_year_2006_grid.vrt'
		output_dir = "/vol/v3/ben_ak/raster_files/neural_net/testing/noData/"
		read_raster_to_np_and_write_to_tif(test_file,output_dir)
		###########################################################
		#working
		# time0 = time.time()
		# file_list = glob.glob(os.path.join(vrt_dir,'*.vrt'))
		
		# bar = progressbar.ProgressBar(maxval=len(file_list), \
		#     widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		# bar.start()
		# pool = multiprocessing.Pool(processes=20)

		# part_converter=partial(read_raster_to_np_and_write_to_tif, output_dir=vrt_dir,alt_topo_band=-4)
		# result_list = pool.map(part_converter, file_list)


		# bar.finish()
		# # for file in file_list: 
		# # 	output_filename = os.path.join(vrt_dir,f'{os.path.split(file)[1][:-4]}_corrected.tif')
		# # 	read_raster_to_np_and_write_to_tif(file,output_filename,alt_topo_band=-4)
		# # 	count +=1 
		# time1=time.time()
		# print(f'Processed {len(file_list)} files in {(time1-time0)/60} minutes')

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		train_dir = variables["train_partitions"]
		dev_dir = variables["dev_partitions"]
		output_dir = variables["output_dir"]
	#this assumes that the previous _2_prep_train_locations has been run and has created these dirs
	main(train_dir)

