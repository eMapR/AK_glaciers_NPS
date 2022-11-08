import os 
import sys 
import pandas as pd 
import numpy as np 
import geopandas as gpd 
import glob
import json
import itertools
from osgeo import gdal 
import subprocess
from rasterio.merge import merge
import string 
import rasterio 

def write_list_to_txt_file(input_list,output_filename): 
	'''
	Helper function.
	'''
	output_file =open(output_filename,'w') 
	for file in input_list: 
		output_file.write(file + os.linesep)
	return output_filename

def get_file_list(root_dir): 
	"""Helper function to traverse a parent directory."""
	output_list = []
	# root_dir needs a trailing slash (i.e. /root/dir/)
	for filename in glob.iglob(root_dir + '**/*.tif', recursive=True):
		#print(filename)
		output_list.append(filename)

	return output_list

def make_large_vrt(paths,output_dir,model,year): 
	outvrt = output_dir+f'{model}_model_mosaic_{year}_full_study_area.vrt' #this is dependent on the dictionary of bands:years supplied below
	#outvrt = output_dir+'test_file.vrt'
	if not os.path.exists(outvrt): 
		#outds = gdal.BuildVRT(outvrt, paths,srcNodata=-32768,outputSRS='EPSG:3338')# bandList=list(range(1,10)) currently hardcoded for the bands
		cmd = f'gdalbuildvrt -vrtnodata -32768 -b 1 -input_file_list {paths} {outvrt}'
		#cmd = f'gdalwarp -of VRT -r max --optfile {paths} {outvrt}' #f'gdalwarp $(list_of_tiffs) merged.tiff

		subprocess.call(cmd, shell=True)
	return outvrt

def main(tifs_dir,output_dir,model):
	
	for year in range(1986,2022,2): 
		input_tifs = get_file_list(tifs_dir)
		input_tifs = [file for file in input_tifs if str(year) in file]
		#write list of files to txt file 
		txt_fn = os.path.join(output_dir,f'{model}_{year}_for_stitching.txt')
		if not os.path.exists(txt_fn): 
			write_list_to_txt_file(input_tifs,txt_fn)
		#make a vrt from the subsets for one year (full study area)
		make_large_vrt(txt_fn,output_dir,model,year)

if __name__ == '__main__':
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		model_outputs = variables['classified_dir']
		model = variables['model']
		output_dir = variables['output_dir']
	main(model_outputs,output_dir,model)