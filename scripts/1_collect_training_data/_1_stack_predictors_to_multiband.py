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
import re 

##########################################################################################
############################Helper functions##############################################
##########################################################################################
#helper functions used by other functions in this script 
def run_cmd(cmd):
	'''
	Helper function.
	'''
	print(cmd)  
	return subprocess.call(cmd, shell=True)


def flatten_lists(list_of_lists): 
	'''
	Helper function.
	'''
	flatten=itertools.chain.from_iterable

	return list(flatten(list_of_lists))

def write_list_to_txt_file(input_list,output_filename): 
	'''
	Helper function.
	'''
	output_file =open(output_filename,'w') 
	for file in input_list: 
		output_file.write(file + os.linesep)
	return output_filename

def reordering(files,lookup): 
	'''Helper function.'''
	return [i for i in files if lookup in os.path.split(i)[1]][0]

def get_image_tiles_from_subdirectories(parent_dir,file_modifier,output_dir,dir_type,band_list,write_out,region=''): 
	'''
	Helper function for get_imge_files.
	'''

	if write_out: 
		first_band = output_dir+f'file_{dir_type}_dir_{band_list[0]}_list_{region}_updated.txt'
		second_band = output_dir+f'file_{dir_type}_dir_{band_list[1]}_list_{region}_updated.txt'
		third_band = output_dir+f'file_{dir_type}_dir_{band_list[2]}_list_{region}_updated.txt'
		fourth_band = output_dir+f'file_{dir_type}_dir_{band_list[3]}_list_{region}_updated.txt'
		fifth_band = output_dir+f'file_{dir_type}_dir_{band_list[4]}_list_{region}_updated.txt'
		sixth_band = output_dir+f'file_{dir_type}_dir_{band_list[5]}_list_{region}_updated.txt'

	first_list = []
	second_list = []
	third_list = []
	fourth_list = []
	fifth_list = []
	sixth_list = []

	#output_txt = open(txt_filename,'w')
	output_list = []
	output_dict = {}
	for subdir, dirs, files in os.walk(parent_dir): 
		tile_id = os.path.split(subdir)[1] #hardcoded for the default STEM directory structure 
		if tile_id.startswith('0'): 
			tile_id = tile_id[1:]
		else:
			pass
		#get a list of the tifs in each sub-directory 
		output_tifs = sorted(glob.glob(subdir+f'/*{file_modifier}*.tif'))
		output_list.append(output_tifs)
		output_dict.update({tile_id:output_tifs})
	if write_out: 
		for filename in flatten_lists(output_list): 
			print(filename)
			if filename.endswith(f'{band_list[0]}.tif'):
				first_list.append(filename)#open(first_band,'w').write(filename + os.linesep)
			elif filename.endswith(f'{band_list[1]}.tif'):
				second_list.append(filename)#open(second_band,'w').write(filename + os.linesep)
			elif filename.endswith(f'{band_list[2]}.tif'):
				third_list.append(filename)#open(third_band,'w').write(filename + os.linesep)
			elif filename.endswith(f'{band_list[3]}.tif'):
				fourth_list.append(filename)#open(fourth_band,'w').write(filename + os.linesep)
			elif filename.endswith(f'{band_list[4]}.tif'):
				fifth_list.append(filename)#open(fifth_band,'w').write(filename + os.linesep)
			elif filename.endswith(f'{band_list[5]}.tif'):
				sixth_list.append(filename)#open(sixth_band,'w').write(filename + os.linesep)
			else: 
				pass
		write_list_to_txt_file(first_list,first_band)
		write_list_to_txt_file(second_list,second_band)
		write_list_to_txt_file(third_list,third_band)
		write_list_to_txt_file(fourth_list,fourth_band)
		write_list_to_txt_file(fifth_list,fifth_band)
		write_list_to_txt_file(sixth_list,sixth_band)
		return output_list,output_dict,[first_band,second_band,third_band,fourth_band,fifth_band,sixth_band]

	else: 
		print('Not writing files to txt file, just returning list and dict')
		return output_list,output_dict
	


##########################################################################################
############################File collection###############################################
##########################################################################################
#File collection and organizing for vrt creation 

def get_image_files(optical_directory,topo_directory,other_directory,modifier): 
	'''
	Make lists of images in subdirectories that are the default data storage method for STEM. Currently it is expected that you have any data you want to add in this format. 
	'''
	output_dict = {}

	#first get optical images
	optical=get_image_tiles_from_subdirectories(optical_directory,'')

	if topo_directory: #note that topo_directory could be whatever other predictors you want
		topo = get_image_tiles_from_subdirectories(topo_directory,modifier)
		#print(topo[1])
	if other_directory: 
		other = get_image_tiles_from_subdirectories(other_directory,modifier) #this is currently setup to be the class label layer but could be other predictor variables
		#print(other[1])
	for k,v in optical[1].items(): #iterate a dictionary that looks like {tile_id:list_of_files}
		try: 
			output_dict.update({k:optical[1][k]+topo[1][k]+other[1][k]})
		except Exception as e: 
			print('The error was: ',e)
			output_dict.update({k:optical[1][k]})
	#print(output_dict)
	return output_dict


##########################################################################################
############################Create final vrt files #######################################
##########################################################################################
def make_singleband_vrts(paths,output_dir,years,dir_type): 
	'''
	Deal with multi-band rasters when making a mosaic.
	'''
	for band in range(1,19): #was hardcoded for the 2000-2019 period, changed to do 1984-2020 (18 epochs)
		output_filename = output_dir+f'{dir_type}_single_band_{band}_for_year_{years[band]}.vrt' 
		cmd = f'gdalbuildvrt -b {band} -input_file_list '+paths+' '+output_filename
		subprocess.call(cmd, shell=True)
	return cmd 


def make_multiband_rasters(paths,output_dir,years,bands,dir_type,separate,noData_value,band_type): 
	'''
	Read in the list of tifs in a directory and convert to multiband vrt file.
	'''
	#for k,v in paths.items(): 
	outvrt = output_dir+f'{dir_type}_{years}_{band_type}_multiband.vrt'#f'optical_topo_multiband_{k}_tile_yearly_composite_{years[bands]}_w_class_label.vrt' #this is dependent on the dictionary of bands:years supplied below and assumes datatype Float32
	#print('the small tile output file is: ', outvrt)
	if not os.path.exists(outvrt): 
		if separate: 
			print('processing with separate')
			outds = gdal.BuildVRT(outvrt, paths, separate=True,bandList=[bands],srcNodata=noData_value)
		else: 
			print('processing without separate')
			outds = gdal.BuildVRT(outvrt, paths,bandList=[bands],srcNodata=noData_value)
	return None


	
def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		tiles_shapefile = variables["tiles_shapefile"]
		pts_shapefile = variables["pts_shapefile"]
		image_tiles_dir = variables["image_tiles_dir"]
		output_dir = variables["output_dir"]
		topo_tiles_dir = variables["topo_tiles_dir"]
		vrt_dir = variables["vrt_dir"]
		class_label_raster = variables["class_label_raster"]
		class_label_dir = variables["class_label_dir"]
		topo_raster = variables["topo_raster"]
		distance_raster = variables["distance_raster"]
		
		#bands_to_years = {1:2001,2:2003,3:2005,4:2007,5:2009,6:2011,7:2013,8:2015,9:2017,10:2019} #this is set up for current data processing 12/28/2020
		bands_to_years = {1:1986,2:1988,3:1990,4:1992,5:1994,6:1996,7:1998,8:2000,
		9:2002,10:2004,11:2006,12:2008,13:2010,14:2012,15:2014,16:2016,17:2018,18:2020}
		####################################################################################
		#get the topo and class label data from dirs. This could be negated if you just go from a big tif or vrt 
		#topo = get_image_tiles_from_subdirectories(topo_tiles_dir,'int',None,'topo',None,None)[0]
		# class_label = get_image_tiles_from_subdirectories(class_label_dir,'int',None,'class_label',None,None)[0]
		
		####################################################################################
		#create one vrt file for each optical predictor per year (60 files in total)- use this to make one file for each band for each year and then these will get stacked
		# txt_files = "/vol/v3/ben_ak/raster_files/neural_net/file_list_txt_files/"
		yearly_optical_vrts="/vol/v3/ben_ak/raster_files/neural_net/optical_vrts_by_year/southern_region/"

		# band_list = ['nbr','ndsi','ndvi','tcb','tcw','rmse']
		# optical = get_image_tiles_from_subdirectories(image_tiles_dir,'',txt_files,'optical',band_list,'True',region='southern')
		
		# for files in glob.glob(txt_files+"*updated.txt"): 
		# 	make_singleband_vrts(files,yearly_optical_vrts,bands_to_years,os.path.split(files)[1][:-4])
		
		####################################################################################
		#stack the annual optical vrt files this is depreceated 3/22/2021
		# for k,v in bands_to_years.items(): 

		# 	vrt_list = glob.glob(yearly_optical_vrts+f'*{v}.vrt')
		# 	vrt_list = [file for file in vrt_list if not 'rmse' in file]
		# 	make_multiband_rasters(vrt_list,output_dir,v,1,'optical',True,-32768,'optical')
		
		####################################################################################
		#make a vrt for the topo data 
		north = []
		south = []
		curvature = []
		base_dem = []

		topo_vrts = "/vol/v3/ben_ak/raster_files/neural_net/topo_vrts/southern_region/"

		# for file in flatten_lists(topo): 
		# 	print(file)
		# 	if file.endswith('north_temp_int.tif'): 
		# 		north.append(file)
		# 	elif file.endswith('south_temp_int.tif'): 
		# 		south.append(file)
		# 	elif file.endswith('curvature_temp_int.tif'): 
		# 		curvature.append(file)
		# 	elif file.endswith('cubic_temp_int.tif'): 
		# 		base_dem.append(file)
		# 	else: 
		# 		pass 

		# make_multiband_rasters(north, topo_vrts, 'all',1,'topo_north',None,-9999,'singleband_int')
		# make_multiband_rasters(south, topo_vrts, 'all',1,'topo_south',None,-9999,'singleband_int')
		# make_multiband_rasters(curvature, topo_vrts, 'all',1,'topo_curvature',None,-9999,'singleband_int')
		# make_multiband_rasters(base_dem, topo_vrts, 'all',1,'topo_dem',None,-9999,'singleband_int')

		
		####################################################################################
		#make a vrt for the class label data 
		#make_multiband_rasters(flatten_lists(class_label), "/vol/v3/ben_ak/raster_files/neural_net/class_vrts/", 'all',1,'class_label',None)

		# ####################################################################################
		#make the final output- this is a stack of single band vrts for each predictor 
		topo_files = glob.glob(topo_vrts+'*.vrt') #these need to be in a specific order for the noData handling to work so they need to be reordered 
		
		#reorder the bands- kind of clunky 		
		first = reordering(topo_files,'north')
		second = reordering(topo_files,'south')
		third = reordering(topo_files,'curvature')
		fourth = reordering(topo_files,'dem')
		
		del(topo_files)
		topo_files = [first,second,third,fourth]

		#print(topo_files)
		
		optical_files = glob.glob(yearly_optical_vrts+'*.vrt')
		for k,v in bands_to_years.items(): 
			optical_year = [file for file in optical_files if (str(v) in file) and ('rmse' not in file)]
			#print(optical_year)
			#file_str = os.path.split(file)[1][:-4]
			#year = re.findall('\d+', file_str)[0]
			make_multiband_rasters(flatten_lists([optical_year,topo_files,[class_label_raster],[distance_raster]]),vrt_dir,v,1,'full_time_series_year','separate',-32768,'revised_class_label') #paths,output_dir,years,bands,dir_type,separate): 


if __name__ == '__main__':
    main()
