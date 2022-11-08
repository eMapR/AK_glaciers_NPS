import geopandas as gpd 
import pandas as pd 
import numpy as np 
import os 
import sys 
from osgeo import gdal, ogr, osr 
import glob 
import re 
from osgeo import ogr
import json
from pathlib import Path
import multiprocessing
from functools import partial
import pandas as pd 
from functools import reduce


def remove_background_from_vector_file(input_file,field='rgi_label',crs='EPSG:3338',**kwargs): 
	"""Remove erroneous garbage from vector file which was converted from a raster."""
	if 'no_background' in str(input_file): 
		return None 
	else: 
		#create the output file name and check to see if it already exists
		output_file = str(input_file)[:-4]+'_no_background.shp'
		if not os.path.exists(output_file): 
			#read in a shapefile and sort the field that comes from conversion 
			try: 
				gdf = gpd.read_file(input_file).sort_values(field)
			except KeyError: 
				print(f'That shapefile does not have the field {field}.')
				raise
			#get only values above zero, this gets rid of the background garbage 
			gdf = gdf.loc[gdf[field]!=0] #changed 5/20/2021 so that it leaves neg values as those are the unlabled vectors we need to deal with in subsequent steps
			gdf['id'] = range(gdf.shape[0])
			gdf = gdf.to_crs(crs)
			print('The file to process is: ')
			print(output_file)
			#write the result to disk 
			try: 
				gdf.to_file(output_file)
			except ValueError: 
				print('That file does not appear to have any debris covered glacier.')
			return None 
		else: 
			pass

def createBuffer(input_file,output_dir,bufferDist=0,char_strip=-18,modifier=''):
	try: 
		output_file = os.path.join(output_dir,os.path.split(str(input_file))[1][:char_strip]+f'_buffer.shp')
		#IMPORTANT: change for running GlacierNet outputs
		#output_file = str(input_file)[:char_strip]+f'_buffer_{modifier}.shp' #hardcoded for the files that are produced by the prior function 
		if not os.path.exists(output_file): 
		# gdf = gpd.read_file(input_file)
		# print(gdf.columns)
		# gdf=gdf.buffer(bufferDist)
		# print(gdf.columns)
		# gdf.to_file(output_file)
			print('making ',str(input_file))
			inputds = ogr.Open(str(input_file))
			inputlyr = inputds.GetLayer()

			#srs = inputlyr.GetSpatialRef().ExportToWkt()
			srs = osr.SpatialReference()
			srs.ImportFromEPSG(3338)
			shpdriver = ogr.GetDriverByName('ESRI Shapefile')
			# if not os.path.exists(output_file):
			# 	shpdriver.DeleteDataSource(output_file)
			outputBufferds = shpdriver.CreateDataSource(output_file)
			bufferlyr = outputBufferds.CreateLayer(output_file, srs, geom_type=ogr.wkbPolygon)
			featureDefn = bufferlyr.GetLayerDefn()

			#Create new fields in the output shp and get a list of field names for feature creation
			fieldNames = []
			for i in range(inputlyr.GetLayerDefn().GetFieldCount()):
				fieldDefn = inputlyr.GetLayerDefn().GetFieldDefn(i)
				bufferlyr.CreateField(fieldDefn)
				fieldNames.append(fieldDefn.name)

			for feature in inputlyr:
				ingeom = feature.GetGeometryRef()
				fieldVals = [] # make list of field values for feature
				for f in fieldNames: fieldVals.append(feature.GetField(f))
				geomBuffer = ingeom.Buffer(bufferDist)

				outFeature = ogr.Feature(featureDefn)
				outFeature.SetGeometry(geomBuffer)
				for v, val in enumerate(fieldVals): # Set output feature attributes
					outFeature.SetField(fieldNames[v], val)
				bufferlyr.CreateFeature(outFeature)
				outFeature = None
			inputds=None


			# outFeature = ogr.Feature(featureDefn)
   #      geomBuffer = ingeom.Buffer(bufferSize)
   #      outFeature.SetGeometry(geomBuffer)
   #      for v, val in enumerate(fieldVals): # Set output feature attributes
   #          outFeature.SetField(fieldNames[v], val)
   #      bufferlyr.CreateFeature(outFeature)

   #      outFeature = None

    # Copy the input .prj file
    # from shutil import copyfile
    # copyfile(inShpPath.replace('.shp', '.prj'), buffShpPath.replace('.shp', '.prj'))

		else: 
			print('That file already exists...')
	#just trying to find some kind of error that is happening but I think its from conversion step (4/29/2021)
	except Exception as e: 
		print('There was an error and it was: ', e)
		raise#print(f'Error was {e}')


def clip_shps(input_shape,clip_bounds,output_dir): 
	"""Clip a directory of shapefiles to a boundary."""
	#cast input shape as str
	input_shape = str(input_shape)
	#check if the output dir exists and if not make it 
	if output_dir: 
		source_file = os.path.split(clip_bounds)[1][:-4]
		dir_name = os.path.join(output_dir,source_file)
		#print(dir_name)
		#being a little finicky when you don't want to create a subfolder
		# if not os.path.exists(dir_name):
		# 	print('making dir ',dir_name) 
		# 	os.mkdir(dir_name)

	## Input
	driverName = "ESRI Shapefile"
	driver = ogr.GetDriverByName(driverName)
	inDataSource = driver.Open(input_shape, 0)
	inLayer = inDataSource.GetLayer()

	print(inLayer.GetFeatureCount())
	## Clip
	inClipSource = driver.Open(clip_bounds, 0)
	inClipLayer = inClipSource.GetLayer()
	print(inClipLayer.GetFeatureCount())

	## create output and write 
	if output_dir: 
		output_file = os.path.join(output_dir,f'{os.path.split(input_shape)[1][:-4]}.shp')
	else: 
		output_file = input_shape[:-4]+'_clipped.shp'
	print(output_file)
	outDataSource = driver.CreateDataSource(output_file)
	outLayer = outDataSource.CreateLayer('FINAL', geom_type=ogr.wkbMultiPolygon)

	ogr.Layer.Clip(inLayer, inClipLayer, outLayer)
	inDataSource.Destroy()
	inClipSource.Destroy()
	outDataSource.Destroy()

	return None 


def edit_and_filter_attributes(input_shp,crs='EPSG:3338',min_size=0.01,char_strip=-9): #the min size is given in km2 
	"""Used to create an area field in a shapefile attribute table and then remove items below a certain threshold."""
	print(input_shp)

	gdf = gpd.read_file(input_shp)
	try: 
		gdf = gdf.set_crs(crs)
	except AttributeError as e: 
		pass
		
	if 'r_area' not in gdf.columns: 
		gdf['r_area'] = gdf['geometry'].area / 10**6 #this assumes that the field that is being created is in meters and you want to change to km2

	gdf = gdf.loc[gdf['r_area']>=min_size]

	output_file = str(input_shp)[:char_strip]+f'_w_{min_size}_min_size.shp' #changed to just take off the '__buffer' and the ext
	try: 
		gdf.to_file(output_file)
	except ValueError: 
		print('there is no data in that file.')
	return output_file


def main(input_dir,clip_bounds,output_dir,**kwargs): 
	##########################################################################
	##########Use this section for processing larger areas in bulk (e.g. full or large chunks of outputs from CNN model) 
	##########################################################################	
	"""run the functions for a directory of raster-vector conversions like: 
	1. remove_background_from_vector_file
	2. createBuffer
	3. edit_and_filter_attributes
	""" 
	################
	##do a little test for one shapefile
	################
	# tw = '/vol/v3/ben_ak/vector_files/neural_net_data/dem_asl_data/zero_and_below_sieved_no_background_buffer.shp'
	# createBuffer(tw,bufferDist=10,char_strip=-4,modifier='10m')
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)

	################
	##run a bunch of files in parallel 
	################
	#get a list of input files recursively
	#input_files = list(Path(input_dir).rglob('*.shp')) #recursive 
	#print(input_files)
	input_files = glob.glob(input_dir+'*.shp') #one dir
	pool = multiprocessing.Pool(processes=25)
	
	clean_vectors=partial(createBuffer,output_dir=input_dir,char_strip=-4)#,char_strip=-4)#clip_shps, clip_bounds=clip_bounds,output_dir=None)#, char_strip=-4)
	
	result_list = pool.map(clean_vectors, input_files)
	##########################################################################
	##########Use this section to create small areas for visualization and making gifs 
	##########################################################################
	#input_file = "/vol/v3/ben_ak/vector_files/neural_net_data/outputs/04222021_model_run/northern_region/04222021_model_run/04222021_model_2008_04222021_model_run.shp"
	#remove_background_from_vector_file(input_file)
	#createBuffer("/vol/v3/ben_ak/vector_files/neural_net_data/outputs/04222021_model_run/northern_region/04222021_model_run/04222021_model_2008_04222021_model_run_no_background.shp")
	#edit_and_filter_attributes("/vol/v3/ben_ak/vector_files/neural_net_data/outputs/04222021_model_run/northern_region/04222021_model_run/04222021_model_2008_04222021_model_run_buffer.shp")

	#input files for background fix 
	#files = glob.glob(input_dir+'*.shp')
	#input files for buffer function 
	#files = glob.glob(input_dir+'*_no_background.shp')
	#input files for clip function 
	# files = glob.glob(input_dir+'*buffer.shp')
	# # print(files)

	# for file in files: 
	# # 	#remove_background_from_vector_file(file)#,field=kwargs.get('field')) #step 1- get rid of garbage 
	# # 	#createBuffer(file) #step 2- buffer with a distance of zero to fix topological errors 
	# 	clip_shps(file,clip_bounds,output_dir) #step 3- clip to a small area for visualization 
	
if __name__ == '__main__':

	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		input_dir = variables['input_dir']
		clip_bounds = variables['clip_bounds']
		output_dir = variables['output_dir']
		model = variables['model_version']

	main(input_dir,clip_bounds,output_dir,field='pixelvalue',model=model)