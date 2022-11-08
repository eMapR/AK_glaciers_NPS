

#Import modules

import pandas as pd
import os
import sys


def featable (workspace, basefile, filename):
	print ('Function starting')

	#future arguments

	path= workspace
	primeraw = basefile
	newfilename = filename

	print(workspace)
	# #Creating base for new file

	# prime1=pd.read_csv(primeraw,usecols = [i for i in range(3)])
	# prime=pd.DataFrame(prime1)
	# print("Base file started")

	# #Appending Raw Data to Prime
	# files = os.listdir(path)

	# for f in files:
	#     os.chdir(path)
	#     #get time stamp
	#     time=str(f[0:5]).replace("-", ":")
	#     tempfile= pd.read_csv(f,usecols = [i for i in range(4)])
	#     #append rew data col to to prime
	#     df=pd.DataFrame(tempfile[' Raw Data (365-15/450-20)'])
	#     prime[time]=df
	#     print("Column added")

	# #Creating new File
	# prime.to_csv(r"%s/%s.csv" % (path,newfilename),index = False, header=True)

	print("Done")

if __name__ == "__main__":
	workspace, basefile, filename = sys.argv[1:]

	print(workspace)
	featable (workspace, basefile, filename)







# import yaml

# with open("/vol/v3/ben_ak/glacier_neural_net/dl_glacier_mapping_2020/params/config.yml", 'rb') as f:
#     conf = yaml.safe_load(f.read())    # load the config file

# def process(**params):    # pass in variable numbers of args
#     for key, value in params.items():
#         print('%s: %s' % (key, value))

# process(**conf)    # pass in your keyword args



#depreceated: 
	#split train/test 512x512 chips 
		#get list lengths for a 70/30 split 
		
		# x_train ,x_test = train_test_split(np.array(vrt_grid_list),test_size=0.3,random_state=10)       #test_size=0.5(whole_data)
		
		# #just copy and rename the test 512x512 chips to a new dir 
		# for file in x_test: 
		# 	vrt_file=rename_copy_file(file,test_dir)
		# 	tif_file = vrt_file[:-4]+'.tif' #input for next function where we convert vrt to tif 
		# 	#convert the vrt file to a tif file with gdal_translate
		# 	tif=convert_to_tif.convert_single_file(vrt_file,tif_file)
		# 	if not tif: 
		# 		break 
			#read_raster_to_np_and_write_to_tif(vrt_file,tif_file,None,None,None,None) 


# def read_raster_to_np_and_write_to_tif(input_raster,filename,col_off,row_off,image_chip_size,class_nodata,topo_nodata,toss_threshold): 
# 	"""Read a vrt raster into np array."""

# 	subset_window=Window(col_off, row_off, image_chip_size, image_chip_size)
	
# 	with rasterio.open(input_raster) as src: #window works like Window(col_off, row_off, width, height)

# 		w = src.read(window=subset_window,masked=True)
		
# 		#make sure the noData values match the image dtype 
# 		if w.dtype == 'float32': 
# 			topo_nodata = float(topo_nodata)
# 			class_nodata = float(class_nodata) 
# 		elif (w.dtype == 'int16') | (w.dtype == 'int32'): 
# 			topo_nodata = int(topo_nodata)
# 			class_nodata = int(class_nodata) 

# 		#first deal with chips that we want to toss. These have >50% class noData and/or they have >50% missing topo data 
# 		class_vals = np.count_nonzero(np.logical_and(w[-1]<=7, w[-1]>=3)) #count the number of valid class values 
# 		topo_nd = np.count_nonzero(w[-2]==topo_nodata) #count the number of topo no data values 

# 		if (class_vals < (image_chip_size**2)*toss_threshold) | (topo_nd>(image_chip_size**2)*toss_threshold): 
# 			print(f'That image chip contains {round((class_vals/image_chip_size**2)*100,2)} percent valid class \nlabels, and {(topo_nd/image_chip_size**2)*100} percent topo no data values. Skipping...')
# 			return None
# 		else: 
# 			print(f'That chip has more than {toss_threshold} valid class, processing...')
# 		#if we're going to keep the chip make sure we change all pixels that are under noData class label to a uniform value 

# 		#get the boolean mask of the class label 
# 		class_mask = w[-1].mask	

# 		#get the topographic data mask (this is currently the second to last band or the last topographic band which is the DEM)
# 		topo_mask = w[-2]==topo_nodata
		
# 		#combined_mask = np.logical_and(class_mask, topo_mask) #this is getting confused because True+False evaluates to False 
		
# 		#first fill topo bands that default to zero when topo should be noData with an arbitrary value outside the possible range 
# 		w[5] = np.where(w[5] == 0, 10000, w[5])
# 		w[6] = np.where(w[6] == 0, 10000, w[6])
		
# 		#apply the masks 
# 		out_arr=np.ma.array(w, mask=w*class_mask[np.newaxis,:,:]).filled(fill_value=class_nodata)
# 		out_arr=np.ma.array(w, mask=w*topo_mask[np.newaxis,:,:]).filled(fill_value=class_nodata)

# 		#then set valid vals back to 0 for writing 
# 		w[5] = np.where(w[5] == 10000, 0, w[5])
# 		w[6] = np.where(w[6] == 10000, 0, w[6])

# 		#make a mask band 
# 		alpha = np.where(out_arr[-1]==class_nodata,0,1).reshape((1,image_chip_size,image_chip_size))

# 		out_arr = np.concatenate((out_arr, alpha), axis=0)

# 		#get the image metadata for writing 
# 		profile=src.profile
		
# 		#make a few updates to the metadata before writing
# 		profile.update(
# 		count=11,
# 		driver='GTiff',
# 		compress='lzw', 
# 		dtype='float64',
# 		width=image_chip_size, 
# 		height=image_chip_size,
# 		transform=rasterio.windows.transform(subset_window,src.transform)) #reassign the output origin to the subset_window 
		
# 	with rasterio.open(filename, 'w', **profile) as dst:
# 			dst.write(out_arr)

# 	return None 
#working but applies no thresholds

# def read_raster_to_np_and_write_to_tif(input_raster,filename,col_off,row_off,image_chip_size,window,no_data_value): 
# 	"""Read a vrt raster into np array."""

# 	if window: 
# 		subset_window=Window(col_off, row_off, image_chip_size, image_chip_size)
# 	else: 
# 		pass
	
# 	with rasterio.open(input_raster) as src: #window works like Window(col_off, row_off, width, height)
# 		if window: 
# 			print('processing with window')
# 			w = src.read(window=subset_window,masked=True)
# 			#get a couple of stats about the tile 
# 			num_cells = w.size
# 			nd_vals = np.count_nonzero(w == no_data_value)
# 			#in instances where the 128x128 chip is more than 50% no data, don't write just exit the function
# 			if nd_vals>0: 
# 				print('That image chip contains null values, skipping...')
# 				return None

# 			profile=src.profile
			
# 			#make a few updates to the metadata for writing
# 			profile.update(
# 			driver='GTiff',
# 			compress='lzw', 
# 			width=image_chip_size, 
# 			height=image_chip_size,
# 			transform=rasterio.windows.transform(subset_window,src.transform)) #reassign the output origin to the subset_window 
		
# 		else: 
# 			print('processing without window')
# 			w = src.read(1)
# 			profile = src.profile
# 			profile.update(
# 			driver='GTiff', 
# 			compress='lzw'
# 			)

# 	with rasterio.open(filename, 'w', **profile) as dst:
# 		if not window: 
# 			dst.write(w,indexes=1)
# 		else: 
# 			dst.write(w)

# 	return None 
# import os
# import sys
# import glob
# import pandas as pd 
# import numpy as np 
# import json
# import rasterio 
# import subprocess
# import multiprocessing
# import random
# import time
# from rasterio.windows import Window
# import re
# from functools import partial
# import random 
# import shutil
# from scipy import stats
# from sklearn.model_selection import train_test_split
# import _3_extract_image_chips as base_chips 
# import _3b_convert_vrt_to_tif as convert_to_tif


# def read_raster_to_np_and_write_to_tif(input_raster,filename,col_off,row_off,image_chip_size,class_nodata,topo_nodata): 
# 	"""Read a vrt raster into np array."""

# 	subset_window=Window(col_off, row_off, image_chip_size, image_chip_size)
	
# 	with rasterio.open(input_raster) as src: #window works like Window(col_off, row_off, width, height)

# 		w = src.read(window=subset_window,masked=True)
# 		print(w.dtype)
# 		#make sure the noData values match the image dtype 
# 		if w.dtype == 'float32': 
# 			topo_nodata = float(topo_nodata)
# 			class_nodata = float(class_nodata) 
# 		elif (w.dtype == 'int16') | (w.dtype == 'int32'): 
# 			topo_nodata = int(topo_nodata)
# 			class_nodata = int(class_nodata) 

# 		#first deal with chips that we want to toss. These have >50% class noData and/or they have >50% missing topo data 
# 		class_vals = np.count_nonzero(np.logical_and(w[-1]<=7, w[-1]>=3)) #count the number of valid class values 
# 		topo_nd = np.count_nonzero(w[-2]==topo_nodata) #count the number of topo no data values 

# 		if (class_vals < (image_chip_size**2)/2) | (topo_nd>(image_chip_size**2)/2): 
# 			print(f'That image chip contains {round((class_vals/image_chip_size**2)*100,2)} percent valid class \nlabels, and {(topo_nd/image_chip_size**2)*100} percent topo no data values. Skipping...')
# 			#return None

# 		#if we're going to keep the chip make sure we change all pixels that are under noData class label to a uniform value 

# 		#get the boolean mask of the class label 
# 		class_mask = w[-1].mask	

# 		#get the topographic data mask (this is currently the second to last band or the last topographic band which is the DEM)
# 		topo_mask = w[-2]==topo_nodata
		
# 		#combined_mask = np.logical_and(class_mask, topo_mask) #this is getting confused because True+False evaluates to False 
		
# 		#first fill topo bands that default to zero when topo should be noData with an arbitrary value outside the possible range 
# 		w[5] = np.where(w[5] == 0, 10000, w[5])
# 		w[6] = np.where(w[6] == 0, 10000, w[6])
		
# 		#apply the masks 
# 		out_arr=np.ma.array(w, mask=w*class_mask[np.newaxis,:,:]).filled(fill_value=class_nodata)
# 		out_arr=np.ma.array(w, mask=w*topo_mask[np.newaxis,:,:]).filled(fill_value=class_nodata)

# 		#then set valid vals back to 0 for writing 
# 		w[5] = np.where(w[5] == 10000, 0, w[5])
# 		w[6] = np.where(w[6] == 10000, 0, w[6])

# 		#make a mask band 
# 		alpha = np.where(out_arr[-1]==class_nodata,0,1).reshape((1,image_chip_size,image_chip_size))
# 		print(alpha)

# 		out_arr = np.concatenate((out_arr, alpha), axis=0)
# 		print(out_arr.shape)
# 		#get the image metadata for writing 
# 		profile=src.profile
		
# 		#make a few updates to the metadata before writing
# 		profile.update(
# 		count=11,
# 		driver='GTiff',
# 		compress='lzw', 
# 		dtype='float64',
# 		width=image_chip_size, 
# 		height=image_chip_size,
# 		transform=rasterio.windows.transform(subset_window,src.transform)) #reassign the output origin to the subset_window 
		
# 	with rasterio.open(filename, 'w', **profile) as dst:
# 			dst.write(out_arr)

# 	return None 


# test_raster="/vol/v3/ben_ak/raster_files/neural_net/512x512_grid_tiles/train_pt_id_894_class_id_grid.vrt"
# output = f"/vol/v3/ben_ak/raster_files/neural_net/testing/894_test7.tif"
# read_raster_to_np_and_write_to_tif(test_raster,output,32,0,128,-32768,-9999)