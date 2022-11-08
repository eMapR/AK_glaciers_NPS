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
import _1_prepare_training_pts as prep_train

def run_cmd(cmd):
  print(cmd)  
  return subprocess.call(cmd, shell=True)

def create_bounding_box(pt_lat,pt_lon,box_size,resolution): 
	'''
	Create a bounding box of given size with point xy coords as the centroid. Note that this assumes a 
	UTM type projected coordinate system where lon has + and - values and lat has positive values. 
	'''
	#for UTM-like coordinate systems centered on 0 within the study area you must accomodate negative and positive lon values
	lrx = pt_lon + ((box_size/2)*resolution)
	ulx = pt_lon - ((box_size/2)*resolution) 
	uly = pt_lat + ((box_size/2)*resolution)
	lry = pt_lat - ((box_size/2)*resolution)
	
	return list([ulx,uly,lrx,lry])

def randomly_move_centroid(pts_tiles_df,box_size): 
	'''
	Randomly reassign a training image chip centroid for image augmentation and variable surface capture.
	Assumes you are working in a projected coordinate system with meters as the default horizontal unit.
	'''
	offset = round(box_size/3)
	pts_tiles_df['new_lon'] = pts_tiles_df['lon'] + random.randint((0-offset),offset)
	pts_tiles_df['new_lat']	= pts_tiles_df['lat'] + random.randint((0-offset),offset)
	
	return pts_tiles_df

def add_bounding_box(pts_tiles_df,image_size,resolution): 
	'''
	Add the coords of a bounding box for each train point.
	'''

	try: 
		pts_tiles_df['bbox'] = pts_tiles_df.apply(lambda x: create_bounding_box(x['new_lat'], x['new_lon'],image_size,resolution), axis=1)
	except KeyError: 
		print('Looking for a lat and lon column but those seem to be missing...')
	
	return pts_tiles_df

def make_single_clip_command(output_dir,pt_id,class_id,bbox_coords,vrt_file,gdal_driver,file_ext): 
	print('pt id is: ',pt_id)
	output_filename = output_dir+f'train_pt_id_{pt_id}_class_id_{class_id}.{file_ext}' #needs to be amended
	#bbox_coords = bbox_df.bbox.iloc[0] #get the list of bounding box coords 
	
	cmd = 'gdal_translate -ot Int16 -projwin_srs EPSG:3338 -projwin '+f'{bbox_coords[0]} {bbox_coords[1]} {bbox_coords[2]} {bbox_coords[3]}'+f'{gdal_driver}'+vrt_file+' '+output_filename

	return cmd 

def create_image_chips(pts_tiles_df,vrt_file,train_image_dir,class_col,id_col,gdal_driver,file_ext,bbox_col): #set this thing up so that its passed to a parallel processing function with each worker doing all the pts in a tile?  
	'''
	Take df with bounding boxes and point ids and clip from the multiband rasters.
	'''
	#add a unique id 
	pts_tiles_df['unique_id'] = np.arange(pts_tiles_df.shape[0])
	pts_tiles_df['cmds'] = pts_tiles_df.apply(lambda x: make_single_clip_command(train_image_dir,x[id_col],x[class_col],x[bbox_col],vrt_file,gdal_driver,file_ext),axis=1)
	cmd_list = list(pts_tiles_df['cmds'])
	return cmd_list

def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		tiles_shapefile = variables["tiles_shapefile"]
		pts_shapefile = variables["pts_shapefile"]
		image_tiles_dir = variables["image_tiles_dir"]
		output_dir = variables["output_dir"]
		vrt_dir = variables["vrt_dir"]
		train_image_dir = variables["train_image_dir"]
		train_class = variables["train_class"]
		image_chip_size = variables["image_chip_size"] #in pixels
		resolution = variables["resolution"] #m
		full_study_area_vrt = variables["full_study_area_vrt"]
		
		#run commands
		joined_df = prep_train.collect_tile_ids(tiles_shapefile,pts_shapefile)
		centroid_df = randomly_move_centroid(joined_df,int(image_chip_size))
		bbox_df=add_bounding_box(joined_df,int(image_chip_size),int(resolution)) #creates a df with tile id, pt id and bounding boxes 
		
		# run the image creation commands in parallel 
		start_time = time.time()
		pool = multiprocessing.Pool(processes=20)
		pool.map(run_cmd, create_image_chips(bbox_df,full_study_area_vrt,train_image_dir,'class_id','unique_id',' -ot GTiff ','.tif','bbox'))  #note that the 'class_id' column was added in QGIS to each pts dataset before merging them
		pool.close()

		print(f'Time elapsed for image chip extraction is: {((time.time() - start_time))/60} minutes')
		
		
if __name__ == '__main__':
    main()

