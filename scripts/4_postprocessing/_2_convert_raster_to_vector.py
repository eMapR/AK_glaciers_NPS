from osgeo import gdal, ogr, osr 
import sys
import os
import glob 
import re 
import numpy as np
import fiona
import rasterio
import rasterio.features
from shapely.geometry import shape, mapping
from shapely.geometry.multipolygon import MultiPolygon
import json
from pathlib import Path
import multiprocessing
from functools import partial
import pandas as pd 

#works but generates some invalid geometries 
def gdal_raster_to_vector(input_file,output_dir,model='',**kwargs): 
	"""Converts a geotiff to a shp file. Use in a for loop to convert a directory of raster to vectors using GDAL. 
	Does not do any additional post-processing, just does a straight conversion."""
	
	print('The file is...')
	print(str(input_file))
	#extract the year from the filename 
	try: 
		year = re.findall('(\d{4})', os.path.split(input_file)[1])[0] #gets a list with the start and end of the water year, take the second one. expects files to be formatted a specific way from GEE 
	except IndexError: 
		year = kwargs.get('year')
	#get the directory one step up the tree from the filename to create a new subdir in the output dir 
	subdir=os.path.dirname(input_file).split('/')[-1] #this is hardcoded and should be changed to run on another platform 
	
	dir_name = os.path.join(output_dir,subdir)
	try: 
		if not os.path.exists(dir_name):
			print('making dir ',dir_name) 
			os.mkdir(dir_name)
		else: 
			pass
	except Exception as e: 
		print(f'That produced the {e} error')
	gdal.UseExceptions()
	
	#create the name of the output
	#IMPORTANT!! uncomment when running FCNN outputs
	#dst_layername = os.path.join(output_dir,os.path.split(input_file)[1][:-4])
	dst_layername = os.path.join(dir_name,f'{model}_model_{year}_{subdir}')
	if not os.path.exists(dst_layername+'.shp'): 
		print('making file')
		#open a raster file and get some info 
		src_ds = gdal.Open(str(input_file))
		srs = osr.SpatialReference()
		srs.ImportFromWkt(src_ds.GetProjection())
		srcband = src_ds.GetRasterBand(1) #hard coded for a one band raster
		
		drv = ogr.GetDriverByName("ESRI Shapefile")
		dst_ds = drv.CreateDataSource(dst_layername + ".shp")
		dst_layer = dst_ds.CreateLayer(dst_layername, srs = srs)
		newField = ogr.FieldDefn('rgi_label', ogr.OFTInteger)
		dst_layer.CreateField(newField)

		#write it out 
		gdal.Polygonize(srcband, None, dst_layer, 0, [], 
		callback=None )
		src_ds=None

		return dst_ds 
	else: 
		pass

def main(input_dir,output_dir,model): 
	#create output dir if it doesn't exist 
	
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)
	#change to recursively iterate through a dir of subdirs and convert each file in parallel 

	#get a list of input files 
	#input_files = glob.iglob(input_dir, recursive=True)#list(Path(input_dir).rglob('*.tif'))
	#use this format for GlacierNet output data that is organized in subdirs 
	input_files = glob.glob(input_dir + '/**/*/*', recursive=True)
	
	pool = multiprocessing.Pool(processes=25)
	
	vectorize=partial(gdal_raster_to_vector, output_dir=output_dir,model=model)
	
	result_list = pool.map(vectorize, input_files)

	print(f'Done converting {len(result_list)} raster files to vector')

	#################################################################################
	#DEPRECEATED as of 8/13/2021
	#use this format for anything else that is just in one dir 
	#this is mostly for processing daymet data
	# years = range(1986,2022,2)
	# years = [str(year) for year in years]
	# input_files = glob.glob(input_dir+'*resolution.tif')
	#input_files = [file for file in input_files if os.path.split(file)[1].split('_')[1] in years] #check if the year of daymet aligns with an FCNN output 

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		input_dir = variables['input_dir']
		output_dir = variables['output_dir']
		model_version = variables['model_version']

	main(input_dir,output_dir,model_version)
