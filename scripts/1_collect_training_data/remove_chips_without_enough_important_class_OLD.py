import os
import sys
import glob
import pandas as pd 
import numpy as np 
import json
from osgeo import gdal
import subprocess
import multiprocessing
import random
import time
import rasterio
import shutil
import warnings
import csv


def remove_image_chips_with_threshold(args): 
	"""Read a single raster into numpy array."""
	input_file,class_band,rejects,class_threshold = args
	#print('the input file is: ', input_file)
	#print('The class band is: ', class_band)
	if not os.path.exists(rejects): 
		os.mkdir(rejects)
	else: 
		pass

	with rasterio.open(input_file) as src: 
		arr = src.read(class_band) #get the last band as that is the class band 
		
		#get pixel counts (counts) of class labels (unique)
		unique,counts = np.unique(arr,return_counts=True)
	
		#get the intended class
		class_type = float((os.path.split(input_file)[1][:-4])[-1]) #get the file name, strip the tif and then get the last element which is the class- hardcoded but created in the previous script 

		count_dict = dict(zip(unique,counts))
		
		#if the class of interest is below the threshold then move the chip to a secondary dir 
		try: 
			if count_dict[class_type]/np.sum(counts) < class_threshold: 
				shutil.move(input_file,rejects)
				increment = 1
			else: 
				increment = 0 
				
		#there should not be a key error because we check the class based on the file but just make sure 
		except KeyError: 
			print(f'There was a KeyError on file {os.path.split(input_file[1])}')

	return increment
		
def move_list_of_files_btwn_dirs(id_list,origin_dir,ext='.tif',**kwargs): 
	"""Move selected files from one directory to another."""
	if not 'dest_dir' in kwargs: 
		dest_dir = origin_dir[:-1]+'_rejects'

	if not os.path.exists(dest_dir): 
		os.mkdir(dest_dir)
	else: 
		pass
	#get list of file names 
	files = glob.glob(origin_dir+f'*{ext}')
	#move the files 
	count = 0 
	for file in files: 
		move_file = [i for i in id_list if i in file]
		if move_file: 
			if len(move_file) > 1: 
				warnings.warn(f'There were more than 1 entries for that id. \n They are: {move_file}')
				#[shutil.move(item,dest_dir) for item in move_file ]
				
				#count +=1

			else: 
				print(file)
				shutil.move(file,dest_dir)
				count +=1 
		else: 
			pass 
	print(f'Done. Moved {count} files from {origin_dir} to {dest_dir}')


def main(): 

	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		vrt_dir = variables['vrt_dir']
		rejects = variables['rejects']
		train_image_dir = variables['train_image_dir']
		
		#get txt file of reject tiles 
		with open(rejects, 'r') as fd:
			reader = csv.reader(fd)
			reject_ids = [row for row in reader][0]

		#image_chips = sorted(glob.glob(vrt_dir+'*.tif'))
		
		#run commands
		start_time = time.time()
		print('Working...')
		move_list_of_files_btwn_dirs(reject_ids,vrt_dir)#,'/vol/v3/ben_ak/raster_files/neural_net/train_256x256_partitions_rejects')
		
		#move files based on threshold of nodata or some other characteristic 
		# count = 0 
		# for f in image_chips:
		# 	remove_chips = remove_image_chips_with_threshold([f,11,train_rejects,0.1]) #should change this to work in paralell 
		# 	count += remove_chips
		# # launch a process for each file (ish).
		# # The result will be approximately one process per CPU core available.
		# #p.apply_async(remove_image_chips_with_threshold, [f,11,train_rejects,0.1]) 
		# print(count)
		# #p.close()
		# #p.join() # Wait for all child processes to close.
		# print(f'Time elapsed for image chip extraction is: {((time.time() - start_time))/60} minutes')

		
if __name__ == '__main__':
    main()
