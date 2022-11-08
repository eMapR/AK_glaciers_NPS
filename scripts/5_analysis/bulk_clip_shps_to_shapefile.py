import os 
import sys
import glob
from osgeo import gdal, ogr
import pandas as pd 
import geopandas as gpd 
import multiprocessing
from functools import partial

"""Supply a shapefile and a directory of shapefiles, clip all the shapefiles in the directory to the shapefile and write them to a new directory."""

def clip_shp(to_clip, bounds, out_dir): 
	file_end = os.path.split(to_clip)[1]
	place = os.path.split(bounds)[1][:-4]
	out_fn = os.path.join(out_dir,place+'_'+file_end)

	output = gpd.clip(gpd.read_file(to_clip),gpd.read_file(bounds))
	output['clip_area'] = output.area
	if not output.empty: 
		output.to_file(out_fn)
	return out_fn

if __name__ == '__main__':
	clip_bounds = "/vol/v3/ben_ak/vector_files/boundaries/brooks_range_block.shp"
	input_dir = "/vol/v3/ben_ak/vector_files/neural_net_data/NSIDC_dataset_submission/final_dataset/AK_glaciers_G10040-V001/overall_area/"
	output_dir = "/vol/v3/ben_ak/vector_files/glacier_outlines/GN_brooks_range/"

	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)

	#set up to iterate through a directory of clipping bounds (national parks) and then through a time series of shapefiles (GlacierNet outputs)
	########use to clip a dir of shapefiles to another dir of shapefiles
	# for park_file in glob.glob(clip_bounds+'*.shp'): 
	# 	print('doing something else')
	# 	print(park_file)
	# 	place = os.path.split(park_file)[1][:-4]

	# 	park_output_dir = os.path.join(output_dir,place+'/')
	# 	print('files are going to: ', park_output_dir)
	# 	if not os.path.exists(park_output_dir): 
	# 		os.mkdir(park_output_dir)
	# 	else: 
	# 		pass

	# 	ts_files = glob.glob(input_dir+'*.shp')

	# 	pool = multiprocessing.Pool(processes=10)
	
	# 	clipping=partial(clip_shp, bounds=park_file,out_dir=park_output_dir)
		
	# 	result_list = pool.map(clipping, ts_files)

	for file in glob.glob(input_dir+'*.shp'): 
		file_end = os.path.split(file)[1]
		# out_fn = os.path.join(park_output_dir,place+'_'+file_end)

		clip_shp(file, clip_bounds, output_dir)
