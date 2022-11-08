#!/usr/bin/env python

import rasterio 
import os
import sys 
import pandas as pd 
import geopandas as gpd 
import json 
import numpy as np 
from osgeo import gdal
import fiona 
import rasterio.mask
import glob
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst, ogr, osr

def resample_raster(target,source,output_dir,year): 
	# Source
	src_filename = source
	src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
	src_proj = src.GetProjection()
	src_geotrans = src.GetGeoTransform()

	# We want a section of source that matches this:
	match_filename = target
	match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
	match_proj = match_ds.GetProjection()
	match_geotrans = match_ds.GetGeoTransform()
	wide = match_ds.RasterXSize
	high = match_ds.RasterYSize

	# Output / destination
	dst_filename = os.path.join(output_dir,f'08122021_model_{year}_resampled_1000m.tif')
	dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Int16)
	dst.SetGeoTransform( match_geotrans )
	dst.SetProjection( match_proj)

	# Do the work
	gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_NearestNeighbour)

	del dst # Flush

def read_raster(raster,rast_band):
	print('raster is: ',raster)
	ds = gdal.Open(raster)
	band = ds.GetRasterBand(rast_band)
	return band.ReadAsArray(),ds

def multiply_rasters(model_raster, daymet_raster, isotherm, output_dir, model_band=1, daymet_band=1): 
	print('model raster is ', model_raster)
	print('daymet_raster is: ', daymet_raster)
	model_arr,model_param = read_raster(model_raster,model_band)
	#the GlacierNet output is a three class raster, make it binary 
	model_arr[model_arr > 0] = 1 
	daymet_arr = read_raster(daymet_raster,daymet_band)
	base_fn = os.path.split(model_raster)[1][:-4]
	output_file = os.path.join(output_dir, f'{base_fn}_daymet_{isotherm}.tif')
	output = model_arr * daymet_arr[0]

	if not os.path.exists(output_file): 

		#for some reason the inputs don't always have the right projection, make sure that's corrected 
		proj = osr.SpatialReference()
		proj.ImportFromEPSG(3338)

		rows,cols = daymet_arr[0].shape #this shouldn't matter which you take from because they need to be the same size 

		driver = gdal.GetDriverByName("GTiff")

		#create the output and write to disk 
		outdata = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32,
			options=['COMPRESS=LZW'])
		outdata.SetGeoTransform(model_param.GetGeoTransform())##sets same geotransform as input
		outdata.SetProjection(proj.ExportToWkt())#sets the projection to a default, something werid was happening where the proj wasn't right 
		outdata.GetRasterBand(1).WriteArray(output)
		#outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
		outdata.FlushCache() ##saves to disk!!
		outdata = None
		ds=None 

def main(input_dir, daymet_dir, output_dir, dem): 
	#1
	#use to resample model output data to match Daymet resolution and extent
	#check if the output dir exists, if not create it 
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir) 
	########################################################################
	#use to resample just one raster
	# daymet_raster = "/vol/v3/ben_ak/raster_files/daymet/daymet_annual_nonbinary/daymet_2020_ak_all_isotherm_nonbinary_annual.tif"
	# # dem = "/vol/v2/ak_glaciers/dem/AK_5m_IFSAR/processed/AK_original_mosaic_resampled_1000m.tif"
	# # output = "/vol/v3/ben_ak/raster_files/daymet/dem/"
	# # resample_raster(daymet_raster, dem, output,'')
	# #use to resample a directory of rasters
	# for raster in glob.glob(input_dir+'*.vrt'): 
	# 	fp = os.path.split(raster)[1][:-4] #get the file name and strip the ext 
	# 	print(fp)
	# 	year = fp.split('_')[3] 
	# 	print('The year is: ',year)
	# 	resample_raster(daymet_raster, raster, output_dir,year)
	
	########################################################################
	#2
	#use to multiply resampled model output data to the daymet positive or negative data 
	for file in glob.glob(input_dir+'*.tif'): 
		fp = os.path.split(file)[1][:-4] #get the file name and strip the ext 
		year = fp.split('_')[2] 
		print('year is: ', year)
		#use when you want to multiply model outputs by Daymet to restrict Daymet to model areas 
		daymet = [file for file in glob.glob(daymet_dir+'*.tif') if year in file][0]
		#use to get model outputs with elevations assigned to each 1000m pixel 
		multiply_rasters(file,daymet,'all',output_dir,daymet_band=8) #note that for daymet you need to set daymet_band to 8 for avg temp

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f: 
		variables = json.load(f)
		input_dir = variables['input_dir']
		daymet_dir = variables['daymet_dir']
		output_dir = variables['output_dir']
		dem = variables['dem']
	main(input_dir, daymet_dir, output_dir, dem)




#########################################################
#########################################################
## everything below here should be used to prepare data for this script in an automated (non-QGIS) fashion 


# class LabelSupport(): 
# 	def __init__(self,input_arr,label_arr,base_arr): 
# 		self.input_arr = input_arr
# 		self.label_arr = label_arr
# 		self.base_arr = base_arr

# 	def max_filtering(self): 
# 		"""Apply a max filter to numpy array. Designed to be applied in iteration."""

# 		#apply the max filter 
# 		labels1 = ndimage.maximum_filter(self.label_arr, size=3, mode='constant') #formerly labels 
		
# 		#get a mask which is just the pixels to be infilled 
# 		new_mask = labels1*self.base_arr #formerly ones 

# 		#fill in pixels that overlap between max arr and CNN class outside RGI labels 
# 		output = np.ma.array(self.input_arr,mask=new_mask).filled(new_mask) #formerly masked

# 		return output

# 	def pad_width(self,small_arr,big_arr): 
# 		"""Helper function."""
# 		col_dif = big_arr.shape[1]-small_arr.shape[1] 
		
# 		return np.pad(small_arr,[(0,0),(0,col_dif)],mode='constant')
		

# 	def pad_height(self,small_arr,big_arr): 
# 		row_dif = big_arr.shape[0]-small_arr.shape[0] 
# 		return np.pad(small_arr,[(0,row_dif),(0,0)],mode='constant')

# 	def match_raster_dims(self): 
# 		"""Take two numpy arrs and match their dimensions."""
# 		c_rows,c_cols = self.base_arr.shape
# 		id_rows,id_cols = self.label_arr.shape

# 		if c_rows < id_rows: #modify the classified raster  
# 			output = self.pad_height(self.base_arr,self.label_arr)
# 			return output,self.label_arr 
		
# 		elif id_rows < c_rows: #modify the id raster 
# 			output = self.pad_height(self.label_arr,self.base_arr)
# 			return self.base_arr,output

# 		elif c_cols < id_cols: #modify the classified raster 
# 			output = self.pad_width(self.base_arr,self.label_arr)
# 			return output, self.label_arr

# 		elif id_cols < c_cols: #modify the id raster 
# 			output = self.pad_width(self.label_arr,self.base_arr)
# 			return self.base_arr, output 
# 		else: 
# 			print('The dimensions of the two input arrs are the same, not sure why we went to this function.')
# 			print(f'Dimensions are {self.base_arr.shape} and {self.label_arr.shape}')

# class LabelRaster(): 
# 	def __init__(self,classified_raster,id_raster,output_file,subset_id,epsg=3338,band=1,change_thresh=0.01,binary_val=1): #added the binary_val 6/15/2021 to deal with debris covered glacier maps  
# 		self.classified_raster=classified_raster
# 		self.id_raster=id_raster
# 		self.output_file=output_file
# 		self.subset_id=subset_id
# 		self.epsg=epsg
# 		self.band=band
# 		self.change_thresh=change_thresh
# 		self.binary_val=binary_val

# 	def read_raster(self,raster):
# 		ds = gdal.Open(raster)
# 		band = ds.GetRasterBand(self.band)
# 		return band.ReadAsArray(),ds

# 	def assign_glacier_id(self): 
# 		"""Read in a classified raster and give it a new id based on RGI."""
		
# 		c_arr=self.read_raster(self.classified_raster)[0]
		
# 		#this is hardcoded to assume the value you want regardless of debris covered or just ice is 1 
# 		#if c_arr.max() != self.binary_val: 
# 		print(f'Changing to binary with target value {self.binary_val}')
# 		#first isolate whatever value you're looking for 
# 		c_arr[c_arr != self.binary_val] = 0
# 		#then change it to one (e.g. in some of the debris covered maps that value is 3 and that will make crazy results when we multiply down the line)
# 		c_arr[c_arr != 0] = 1
		
# 		id_arr=self.read_raster(self.id_raster)[0] 

# 		#deal with an instance where the raster sizes don't align 
# 		c_rows,c_cols = c_arr.shape 
# 		id_rows,id_cols = id_arr.shape
	 
# 		while True: 

# 			if (c_rows != id_rows) | (c_cols != id_cols): 
# 				c_arr,id_arr=LabelSupport(None,id_arr,c_arr).match_raster_dims() #match_raster_dims ALWAYS returns classified raster, ID raster!!
				
# 				#update these vals so we can break the loop when they've been changed 
# 				c_rows,c_cols = c_arr.shape 
# 				id_rows,id_cols = id_arr.shape			
# 			else: 
# 				break