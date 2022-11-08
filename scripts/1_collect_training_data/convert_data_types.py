import os 
import sys 
import pandas as pd 
import numpy as np 
from osgeo import gdal
import geopandas as gpd 
import rasterio
import glob
import json 
import itertools
import subprocess
import time
import multiprocessing
import _2_stack_predictors_to_multiband as base_funcs

##########################################################################################
############################Data cleaning or prep functions###############################
##########################################################################################
#these are mostly one time use functions when prepping data

def run_cmd(cmd):
  print(cmd)  
  return subprocess.call(cmd, shell=True)

def convert_data_type(paths,band_num,modifier,negate): 
	'''
	For a given tif writes a new tif with a new datatype. Currently defaults to int16.
	'''
	for input_file in base_funcs.flatten_lists(paths):
		if (negate in input_file) or ('rmse' in input_file): 
			print(f'Passing the {input_file} file')

		else: 
			output_file = input_file[:-4]+f'_{modifier}.tif'
			if not os.path.exists(output_file): 
				#print(output_file)
				ds = gdal.Open(input_file)
				band = ds.GetRasterBand(band_num)
				arr = band.ReadAsArray()
				rows,cols = arr.shape
				driver = gdal.GetDriverByName("GTiff")
				outdata = driver.Create(output_file, cols, rows, band_num, gdal.GDT_Int16)
				outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
				outdata.SetProjection(ds.GetProjection())##sets same projection as input
				outdata.GetRasterBand(band_num).WriteArray(arr)
				#outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
				outdata.FlushCache() ##saves to disk!!
				outdata = None
				band=None
				ds=None

			else: 
				print(f'{output_file} already exists, passing')

def fill_noData(input_raster): 
	#print(output_file)
	output_file = input_raster[:-4]+f'_filled_noData.tif'
	ds = gdal.Open(input_raster)
	band = ds.GetRasterBand(1)
	arr = band.ReadAsArray()
	arr = np.where(arr<0,0,arr)
	rows,cols = arr.shape
	driver = gdal.GetDriverByName("GTiff")
	outdata = driver.Create(output_file, cols, rows, 1, gdal.GDT_Int16)
	outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
	outdata.SetProjection(ds.GetProjection())##sets same projection as input
	outdata.GetRasterBand(1).WriteArray(arr)
	#outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
	outdata.FlushCache() ##saves to disk!!
	outdata = None
	band=None
	ds=None

def translate_data_type(paths,modifier): 
	'''
	Convert multiband raster to float.
	'''
	cmd_list=[]
	for input_file in flatten_lists(paths):
		output_file = input_file[:-4]+f'_{modifier}.tif'
		cmd = 'gdal_translate -ot Float32 '+input_file+' '+output_file
		cmd_list.append(cmd)
	return cmd_list



def tif_to_vrt(input_tif,output_dir,bands): 
	outvrt = output_dir+f'{os.path.split(input_tif)[1][:-4]}.vrt'
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)
	if not os.path.exists(outvrt): 
		outds = gdal.BuildVRT(outvrt, input_tif, bandList=[bands],srcNodata=-32768)
	return outvrt


def convert_vrt_to_tif(raster_list,output_dir): 
	"""Just what it sounds like."""
	# child_dir = os.path.split(input_shapefile)[1][:-4] #get just the shapefile name with no extension
	# dir_name = os.path.join(output_dir,child_dir)
	# if not os.path.exists(dir_name): 
	# 	os.mkdir(dir_name)
	# else: 
	# 	print(f'{dir_name} already exists, preceeding...')
	cmd_list = []
	for raster in raster_list: 
		output_filename = output_dir+f'{os.path.split(raster)[1][:-4]}.tif'
		cmd = f'gdal_translate -ot Int16 {raster} {output_filename} -co COMPRESS=DEFLATE -co BIGTIFF=YES -a_nodata 0'
		cmd_list.append(cmd)
		#subprocess.call(cmd, shell=True)
	return cmd_list

def main(**kwargs): 
	# params = sys.argv[1]
	# with open(str(params)) as f:
	# 	variables = json.load(f)
	# 	tiles_shapefile = variables["tiles_shapefile"]
	# 	pts_shapefile = variables["pts_shapefile"]
	# 	image_tiles_dir = variables["image_tiles_dir"]
	# 	output_dir = variables["output_dir"]
	# 	single_tif = variables["single_tif"]
	# 	topo_tiles_dir = variables["topo_tiles_dir"]
	# 	vrt_dir = variables["vrt_dir"]
	# 	train_class = variables["train_class"]
	# 	class_label_raster = variables["class_label_raster"]
	# 	class_label_dir = variables["class_label_dir"]

	# 	# vrts = glob.glob(vrt_dir+'*multiband.vrt')
	# 	# print(len(vrts))
	# 	if 'output_dir' in kwargs: 
	# 		output_dir = sys.argv[2]
	# 	if 'single_tif' in kwargs: 
	# 		single_tif = sys.argv[3]
		# pool = multiprocessing.Pool(processes=18)
		# pool.map(run_cmd, convert_vrt_to_tif(vrts,output_dir))  
		# pool.close()
	tif_to_vrt("/vol/v3/ben_ak/raster_files/neural_net/class_label_prep/model_2009_classes_int_infilled_sieved_filled_noData.tif","/vol/v3/ben_ak/raster_files/neural_net/class_vrts/southern_region/",1)
	#fill_noData('/vol/v3/ben_ak/raster_files/neural_net/class_label_prep/model_2009_classes_int_infilled_sieved.tif')
		#one time use data cleaning or prep functions - these should be moved to another script following the convetion of other scripts 
		#translate_tifs =base_funcs.get_image_tiles_from_subdirectories(topo_tiles_dir,'temp',output_dir,'topo',None,False,region='northern')[0]
		#tif_to_vrt(single_tif,'/vol/v3/ben_ak/raster_files/neural_net/class_vrts/',1)
	#convert_data_type('/vol/v3/ben_ak/raster_files/neural_net/class_label_prep/model_2009_classes_int_infilled_sieved.tif',1,'int','float') #the optical data is of type int16 while topo is of type float32. These need to be the same for the buildvrt to work so we're converting topo to int 

if __name__ == '__main__':
    main()

