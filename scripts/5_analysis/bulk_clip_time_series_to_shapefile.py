import os
import sys
from osgeo import gdal
import multiprocessing
import subprocess 
import json
import glob
import time

def run_cmd(cmd):
  #print(cmd)  
  return subprocess.call(cmd, shell=True)

def clip_raster_to_shapefiles(input_raster,shapefile_list,output_dir,noData=0): 
	"""Instead of clipping a list of rasters to one shapefile, clip one raster to a list of shapefiles."""
	cmds=[]
	for shp in shapefile_list: 
		output_filename = os.path.join(output_dir,f'{os.path.split(input_raster)[1][:-4]}_{os.path.split(shp)[1][:-4]}.tif')
		print(output_filename)
		if not os.path.exists(output_filename): 
			cmd = f'gdalwarp -cutline {shp} -crop_to_cutline {input_raster} {output_filename} -co COMPRESS=LZW -srcnodata {noData} -dstnodata {noData}'
			cmds.append(cmd)
			#subprocess.call(cmd, shell=True)
		else: 
			print(f'The {output_filename} file already exists, passing...')
	return cmds

def clip_rasters_to_shapefile(raster_list,input_shapefile,output_dir,fn_modifier,noData=0): 
	"""Just what it sounds like."""
	child_dir = os.path.split(input_shapefile)[1][:-4] #get just the shapefile name with no extension
	dir_name = os.path.join(output_dir,child_dir)
	#print(dir_name)
	#IMPORTANT!!! CHANGE BACK TO RUN IN MULTIPLE SUBDIRS
	# if not os.path.exists(dir_name):
	# 	print('making dir ',dir_name) 
	# 	os.mkdir(dir_name)
	# else: 
	# 	print(f'{dir_name} already exists, preceeding...')
	cmds=[]
	for raster in raster_list: 
		#change back to put stuff into subdirs NOTE#####################
		output_filename = os.path.join(output_dir,f'{os.path.split(raster)[1][:-4]}_{fn_modifier}_clipped.tif')
		print(output_filename)
		if not os.path.exists(output_filename): 
			cmd = f'gdalwarp -cutline {input_shapefile} -crop_to_cutline {raster} {output_filename} -co COMPRESS=LZW -srcnodata {noData} -dstnodata {noData}'
			#uncomment to run in parallel 
			cmds.append(cmd)
			#subprocess.call(cmd, shell=True)
		else: 
			print(f'The {output_filename} file already exists, passing...')
	return cmds

def main(input_dir,clip_bounds,output_dir,vect_dir,single_raster): 
		# if not os.path.exists(output_dir): 
		# 	os.mkdir(output_dir)
		# output_dir = os.path.join(output_dir,'negative')
		if not os.path.exists(output_dir): 
			os.mkdir(output_dir)
		tifs = glob.glob(input_dir+'*.tif')
		# clip_rasters_to_shapefile(tifs,clip_bounds,output_dir)	
		# shps = glob.glob(input_dir+'*.shp')
		# shps = [shp for shp in shps if ('merge' not in shp) & (('16' in shp) | ('15' in shp))] #remove the one that isn't a subset
		# print(shps)

		#process in parallel
		# shps = glob.glob(vect_dir+'*.shp')
		# shps = [shp for shp in shps if ('15' in shp) | ('16' in shp)]
		# start_time = time.time()
		for shp in glob.glob(vect_dir+'*buffer_.shp'): 
			elev_band = os.path.split(shp)[1].split('_')[0]
			pool = multiprocessing.Pool(processes=25)
			pool.map(run_cmd, clip_rasters_to_shapefile(tifs,shp,output_dir,elev_band,noData=-10000))  
			pool.close()

		# print(f'Time elapsed for image chip extraction is: {((time.time() - start_time))/60} minutes')
		
		# shapefiles = glob.glob(vect_dir+'*buffer_.shp')
		# pool = multiprocessing.Pool(processes=25)
		# pool.map(run_cmd, clip_raster_to_shapefiles(single_raster,shapefiles,output_dir,noData=-10000))  
		# pool.close()
		

		#if you want to run for a directory of shapefiles and raster simultaneously 
		#vrts = [file for file in vrts if 'all_predictors' not in file]
		
		#print(vrts)
		# print(chunk_dir)
		# for shp in glob.glob(vect_dir + '*.shp'): 
		# 	clip_rasters_to_shapefile(tifs,shp,output_dir)

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		clip_bounds = variables["clip_bounds"]
		raster_dir = variables['raster_dir']
		output_dir = variables['output_dir']
		input_dir = variables['input_dir']
		vect_dir = variables['vect_dir']
		base_raster = variables['base_raster']
		main(input_dir,clip_bounds,output_dir,vect_dir,single_raster=base_raster)