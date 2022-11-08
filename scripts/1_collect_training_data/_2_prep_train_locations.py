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
import re
import csv
from pathlib import Path


def run_cmd(cmd):
  #print(cmd)  
  return subprocess.call(cmd, shell=True)
#read in the tiles and intersect with glacier bounds 

def make_single_clip_command(output_dir,pt_id,region,bbox_coords,vrt_file,gdal_driver=' ',file_ext='vrt',**kwargs): 
	"""Make a single command using gdal_translate that will snip out a partition from a large VRT file."""

	if 'partition_year' in kwargs: 
		output_filename = os.path.join(output_dir,f'train_pt_id_{pt_id}_partition_year_{kwargs.get("partition_year")}_{region}.{file_ext}') #needs to be amended
	else: 
		output_filename = os.path.join(output_dir,f'train_pt_id_{pt_id}_{region}.{file_ext}') #needs to be amended
	
	cmd = 'gdal_translate -ot Int16 -projwin_srs EPSG:3338 -projwin '+f'{bbox_coords[0]} {bbox_coords[1]} {bbox_coords[2]} {bbox_coords[3]}'+f'{gdal_driver}'+vrt_file+' '+output_filename

	return cmd 

def get_geom_dict(input_gdf,sort_col='new_id'): 
	"""Add a column with bbox coords to geopandas df.
	optional args- 
	sort_col- specify if you want to use a different index column that already exists in the dataset."""
	if not sort_col in input_gdf.columns: 
		print(f'Your shapefile does not contain a suitable ID column. \n Adding a new ID col with header: {sort_col}')
		input_gdf[sort_col]=range(0,input_gdf.shape[0])
	else: 
		pass 
	output_dict = input_gdf.set_index(sort_col)[['left','top','right','bottom']].to_dict(orient='index')
	return output_dict


def get_partition_year(input_gdf,sort_col,year_col='year'): #col of interest in this case should be the same as in get_geom_dict 
	"""Get RGI data and get the mode year for each partition in train or dev. 
	NOTE that in cases where there are equal groups (e.g. 2000,2001) the first year will be selected.
	"""
	input_gdf[year_col] = input_gdf[year_col].astype(int) #make sure year col is a number for next step 
	output_df = input_gdf.groupby(sort_col)[year_col].apply(lambda x: x.mode().iloc[0]).reset_index()
	output_df[year_col][output_df[year_col]%2!=0] = output_df[year_col]+1 #a[a%2 == 1] = -1 

	return dict(zip(output_df[sort_col],output_df[year_col]))
	#list([ulx,uly,lrx,lry])


def create_grid_image_cmds(input_dict,output_dir,vrt_files,**kwargs): 
	"""Make a list of cmnds for subprocess from dictionary.
	Optional args: 
	gdal_driver- change if you don't want VRT
	file_ext- change if you want tif instead of VRT which is the default 
	"""
	cmd_list = []
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)
		
	for k,v in input_dict.items(): 
		print('k is k: ', k)
		#check if there are any tiles that we don't want in the output- specified by a user generated txt file below 
		if 'reject_ids' in kwargs: 
			if str(k) in kwargs.get('reject_ids'): 
				print(f'Skipping tile {k}')
		if 'years_dict' in kwargs: 
			partition_year = kwargs.get('years_dict').get(k)
			print(f'the partition year is: {partition_year}')
			cmd = make_single_clip_command(output_dir,k,'grid',list(v.values()),vrt_files[str(partition_year)],partition_year=partition_year) #this las	
		else: 
			cmd = make_single_clip_command(output_dir,k,'grid',list(v.values()),vrt_files)
		cmd_list.append(cmd)

	return cmd_list


def main(input_grid,output_dir,full_study_area_vrts,**kwargs): 
	#check if the output dir exists and if not, create it 
	if not os.path.exists(output_dir): 
		print('Making output dir')
		Path(output_dir).mkdir(parents=True, exist_ok=True)

	#run commands
	#get the grid that defines the 256x256 partitions
		
	try: 
		gdf = gpd.read_file(kwargs.get('grid')).drop_duplicates('geometry') #if this is from a sj function there will be duplicate geometries 
	except: 
		print('grid is not an arg in kwargs')
		raise 

	try: 
		years=get_partition_year(gdf,sort_col='label')

	except KeyError as e: 
		raise
		year_col = input('There is no year col in that shapefile. Please input another col header and hit enter: ')
		years=get_partition_year(gdf,sort_col='label',year_col=year_col)				
	
	#get the bounding box coords for each grid partition 
	grid_coords = get_geom_dict(gpd.read_file(input_grid),sort_col='label') #sort_col is an optional arg. Default is 'new_id'
	#print(grid_coords)
	#get the list of full study area files (one for each year)
	input_files = glob.glob(full_study_area_vrts+'*class_label_multiband.vrt') #change if you have more things in that folder you don't want to use 
	#create a dict that holds the file year with the file, this is for grabbing a year of the data to align with the RGI 
	input_file_dict = {}
	for file in input_files: 
		year = re.findall('(\d{4})', os.path.split(file)[1])[0] #hardcoded for file format 
		input_file_dict.update({year:file})

	#print(input_file_dict)

	start_time = time.time()
	
	pool = multiprocessing.Pool(processes=20)
	pool.map(run_cmd, create_grid_image_cmds(grid_coords,output_dir,input_file_dict,years_dict=years))  
	pool.close()

	print(f'Time elapsed for image partition extraction is: {((time.time() - start_time))/60} minutes')
		
		
if __name__ == '__main__':
    
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		grid = variables["grid"]
		train_partitions = variables["train_partitions"] #train partition destination 
		dev_partitions = variables["dev_partitions"] #test partition destination 
		full_study_area_vrts = variables["full_study_area_vrts"] #one vrt with all predictor vars as bands for each year in your study period 
		output_dir = variables["output_dir"]
		rejects = variables["rejects"]
		dev_grid = variables['dev_grid'] #fishnet shapefile for test
		train_grid = variables['train_grid'] #fishnet shapefile for train 

	#get txt file of reject tiles. This is optional. 
	with open(rejects, 'r') as fd:
		reader = csv.reader(fd)
		reject_ids = [row for row in reader][0]
	# if sys.argv[2]: 
	# 	main(grid,train_partitions,full_study_area_vrt,sys.argv[2]) #this needs to be tidied up for the extra commandline args 
	# else: 
	#main(grid,train_partitions,full_study_area_vrts,rgi="/vol/v3/ben_ak/vector_files/glacier_outlines/rgi_epsg_3338_clipped_to_northern_region_US.shp")
	#note that here the sjoin_shp is just the grid from GEE which has been reduced to the AK bounds with a spatial join. MAKE SURE IT IS CHANGED IF YOU ARE RUNNING DEV OR TRAIN

	main(grid,train_partitions,full_study_area_vrts,grid=train_grid) 