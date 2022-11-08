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
from datetime import date
from datetime import datetime
import progressbar
from pathlib import Path



#read in the grid shapefile and get the upper corner for each
def run_cmd(cmd):
  print(cmd)  
  return subprocess.call(cmd, shell=True)

def extract_chips(input_raster,filename,col_off,row_off,image_chip_size=128): 
	"""Read a vrt raster into np array."""

	subset_window=Window(col_off, row_off, image_chip_size, image_chip_size)
	
	with rasterio.open(input_raster) as src: #window works like Window(col_off, row_off, width, height)

		w = src.read(window=subset_window,masked=True)

		#get the image metadata for writing 
		profile=src.profile
		
		#make a few updates to the metadata before writing
		profile.update(
		driver='GTiff',
		compress='lzw', 
		width=image_chip_size, 
		height=image_chip_size,
		transform=rasterio.windows.transform(subset_window,src.transform)) #reassign the output origin to the subset_window 
		
	with rasterio.open(filename, 'w', **profile) as dst:
			dst.write(w)

	return None 

def run_moving_window_for_one_partition(input_raster,output_dir,image_chip_size=128,chip_step=8,class_label_band=10,tile_size=256,not_glacier_class=0,increase_density=False): 
	"""Function to iterate extract chips. This function reads in a (default) 256x256 pixel partition and then stepwise (chip step) samples (default 128x128)
	image chips and writes them to disk. 
	Optional args: 
	Image chip size
	chip step
	class label band
	tile size 
	non glacier class 
	NOTE: This indexes from one (the rasterio default) but when masking in the script to prep the training data it indexes from 0 because it is a numpy slicing scheme. Ensure the 
	class label band number is correct."""
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)
	else: 
		print(f'output dir {output_dir} already exists')
	#check how much glacier is in the tile by looking at the class label band 
	with rasterio.open(input_raster) as src: 
		class_arr = src.read(class_label_band)
		#print(class_arr)
		#check partition glacier/non glacier ratio requires increase in sampling density only in train set 
		if increase_density: 			
			#get the number of glacier pixels in tile 
			glac_pix = np.count_nonzero(class_arr > not_glacier_class) 
			#get the number of non-glacier pixels in the tile
			non_glac_pix = np.count_nonzero(class_arr==not_glacier_class)
							
			#when the ratio of glacier/non glacier is more than 40% but less than 60 increase sampling density, otherwise leave it at the default 
			try: 
				if ((glac_pix/(tile_size**2) >= 0.4) & (glac_pix/(tile_size**2) <= 0.6)) & ((non_glac_pix/(tile_size**2) >= 0.4) & (non_glac_pix/(tile_size**2) <= 0.6)): 
					chip_step = chip_step/2
					print(f'the ratio is: {glac_pix/tile_size**2} and {non_glac_pix/tile_size**2}')
			except ZeroDivisionError as e: 
				pass 
	#extract the partition id from the file string 
	partition_id = re.findall('\d+', os.path.split(input_raster)[1])[0]
	step_count = tile_size-image_chip_size #this gives the number of steps after the first chip we want to take in the tile
	
	count = 0
	for i in range(0,int(step_count)+chip_step,chip_step): #move vertically this count 
		#print(f'The row is {i}')
		for j in range(0,int(step_count)+chip_step,chip_step): #move horizontally this count 
		#	print(f'The column is {j}')
			filename = output_dir+f'grid_tile_{partition_id}_train_chip_row{i}_col{j}.tif'
		#	print(filename)
		#	do the actual extraction 
			extract_chips(input_raster,filename,j,i)
			count +=1 
	
	return partition_id, count 

def log_metadata(output_file,output_dict): 
	"""Create a metadata file about the run and write to disk."""

	# create list of strings from dictionary 
	list_of_strings = [ f'{key} : {output_dict[key]}' for key in output_dict ]

	# write string one by one adding newline
	with open(output_file, 'w') as file:
	    [ file.write(f'{st}\n') for st in list_of_strings ]
	return None



def main(input_files,output_dir,**kwargs): 
	#check if the output dir exists and if it doesn't, create one
	if not os.path.exists(output_dir): 
			print('Making output dir')
			Path(output_dir).mkdir(parents=True, exist_ok=True)

	#run commands
	start_time = time.time()
	bar = progressbar.ProgressBar(maxval=len(input_files), \
		    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	#make the train set chips 
	pool = multiprocessing.Pool(processes=20)
	
	if 'chip_step' in kwargs: 
		chip_step = kwargs.get('chip_step')
		moving_window=partial(run_moving_window_for_one_partition, output_dir=output_dir,chip_step=int(chip_step))
	else: 
		moving_window=partial(run_moving_window_for_one_partition, output_dir=output_dir)
	result_list = pool.map(moving_window, input_files)

	bar.finish()
	elapsed_time = (time.time() - start_time)/60
	print(f'Time elapsed for image chip extraction is: {elapsed_time} minutes')
	
	#log the output

	#create the outputs
	partitions = [i[0] for i in result_list]
	chips = np.cumsum([i[1] for i in result_list])[-1]

	now = datetime.now()
	date_time = date.today().strftime('%m_%d_%y')+'_'+datetime.strftime(now, '%H_%M')
	log_filename = os.path.join(output_dir,f'log_file_for_{date_time}_run.txt')

	#make a dictionary of output metadata to write to disk
	output_dict = {'Generation time':str(date_time), 'Partitions processed':str(len(partitions)),'Chips processed':str(chips),
	'Time for extraction':f'{elapsed_time} minutes','njobs':'20','Partition size':str(256),'Chip size':str(128),'Step size':kwargs.get('chip_step')} #note that there are a bunch of default values here that are hardcoded
	#generate metadata file
	if not os.path.exists(log_filename):
		log_metadata(log_filename,output_dict)
		
if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		train_partitions = variables['train_partitions'] #train partitions dir from _2_prep_train_locations
		dev_partitions = variables['dev_partitions'] #test partitions dir from _2_prep_train_locations
		train_dest = variables['train_dest'] #where to put the 128 chips
		dev_dest = variables['dev_dest']

		#get a list of partitions to use for chip extraction 
		input_files = glob.glob(dev_partitions+'*.tif')	
		#note that by default the increase density arg is set to true- if you don't want to do that for dev set it needs to be explicitly set to false
		main(input_files=input_files,output_dir=dev_dest)
	