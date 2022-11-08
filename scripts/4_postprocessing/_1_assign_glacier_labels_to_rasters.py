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
from scipy import ndimage, misc
import time 
import matplotlib.pyplot as plt
from functools import partial

"""Use a rasterized RGI 6.0 dataset to label the outputs of GlacierNet model. 
Pixel values of the rasterized RGI are the RGI labels as int16 values.
"""

class LabelSupport(): 
	def __init__(self,input_arr,label_arr,base_arr): 
		self.input_arr = input_arr
		self.label_arr = label_arr
		self.base_arr = base_arr

	def max_filtering(self): 
		"""Apply a max filter to numpy array. Designed to be applied in iteration."""

		#apply the max filter 
		labels1 = ndimage.maximum_filter(self.label_arr, size=3, mode='constant') #formerly labels 
		
		#get a mask which is just the pixels to be infilled 
		new_mask = labels1*self.base_arr #formerly ones 

		#fill in pixels that overlap between max arr and CNN class outside RGI labels 
		output = np.ma.array(self.input_arr,mask=new_mask).filled(new_mask) #formerly masked

		return output

	def pad_width(self,small_arr,big_arr): 
		"""Helper function."""
		col_dif = big_arr.shape[1]-small_arr.shape[1] 
		
		return np.pad(small_arr,[(0,0),(0,col_dif)],mode='constant')
		

	def pad_height(self,small_arr,big_arr): 
		row_dif = big_arr.shape[0]-small_arr.shape[0] 
		return np.pad(small_arr,[(0,row_dif),(0,0)],mode='constant')

	def match_raster_dims(self): 
		"""Take two numpy arrs and match their dimensions."""
		c_rows,c_cols = self.base_arr.shape
		id_rows,id_cols = self.label_arr.shape

		if c_rows < id_rows: #modify the classified raster  
			output = self.pad_height(self.base_arr,self.label_arr)
			return output,self.label_arr 
		
		elif id_rows < c_rows: #modify the id raster 
			output = self.pad_height(self.label_arr,self.base_arr)
			return self.base_arr,output

		elif c_cols < id_cols: #modify the classified raster 
			output = self.pad_width(self.base_arr,self.label_arr)
			return output, self.label_arr

		elif id_cols < c_cols: #modify the id raster 
			output = self.pad_width(self.label_arr,self.base_arr)
			return self.base_arr, output 
		else: 
			print('The dimensions of the two input arrs are the same, not sure why we went to this function.')
			print(f'Dimensions are {self.base_arr.shape} and {self.label_arr.shape}')

class LabelRaster(): 
	def __init__(self,classified_raster,id_raster,output_file,subset_id,epsg=3338,band=1,change_thresh=0.01,binary_val=1): #added the binary_val 6/15/2021 to deal with debris covered glacier maps  
		self.classified_raster=classified_raster
		self.id_raster=id_raster
		self.output_file=output_file
		self.subset_id=subset_id
		self.epsg=epsg
		self.band=band
		self.change_thresh=change_thresh
		self.binary_val=int(binary_val)

	def read_raster(self,raster):
		ds = gdal.Open(raster)
		band = ds.GetRasterBand(self.band)
		return band.ReadAsArray(),ds

	def assign_glacier_id(self): 
		"""Read in a classified raster and give it a new id based on RGI."""
		
		c_arr=self.read_raster(self.classified_raster)[0]
		
		#this is hardcoded to assume the value you want regardless of debris covered or just ice is 1 
		#if c_arr.max() != self.binary_val: 
		print(f'Changing to binary with target value {self.binary_val}')
		#first isolate whatever value you're looking for 
		if not self.binary_val > 10: #this is just an arbitrary number above the GlacierNet output classes 
			c_arr[c_arr != self.binary_val] = 0
		else: 
			c_arr[c_arr <= 0] = 0 #setting binary_val to a number larger than 10 will select all areas with a class higher than zero 

		#then change it to one (e.g. in some of the debris covered maps that value is 3 and that will make crazy results when we multiply down the line)
		c_arr[c_arr != 0] = 1
		
		id_arr=self.read_raster(self.id_raster)[0] 

		#deal with an instance where the raster sizes don't align 
		c_rows,c_cols = c_arr.shape 
		id_rows,id_cols = id_arr.shape
	 
		while True: 

			if (c_rows != id_rows) | (c_cols != id_cols): 
				c_arr,id_arr=LabelSupport(None,id_arr,c_arr).match_raster_dims() #match_raster_dims ALWAYS returns classified raster, ID raster!!
				
				#update these vals so we can break the loop when they've been changed 
				c_rows,c_cols = c_arr.shape 
				id_rows,id_cols = id_arr.shape			
			else: 
				break 
		#first give the pixels that overlap a RGI boundary an ID 
		#give areas that have no label a 1 so they persist through the next step 
		mask = np.where(id_arr==0,1,id_arr)

		#multiply the labels by a binary glacier map- this leaves 1 where CNN says glacier but RGI doesn't 
		masked = mask*c_arr

		#check how many pixels are missing labels 
		num_label_pix = masked[masked==1].sum()
		
		#get everything that has a label and doesn't have a label 
		#ones = np.where(masked>0,1,0)
		ones = np.where(masked==1,1,0)
		
		#get all the areas that have RGI ids 
		labels = np.where(masked!=1,masked,0)
		
		#apply the max filter to grow the existing labels 
		filled_arr = LabelSupport(masked,labels,ones).max_filtering()
		times = {} #not sure if this is needed but leaving it for now
		count = 0 
		pct_left = 0

		while True: 
			#figure out how many unlabeled pixels are left 
			num_label_left = filled_arr[filled_arr==1].sum()
			
			#sort of clunky but we need the previous iteration to know the change 
			previous_pct = pct_left

			#calculate a pct of pixels labeled of the original without a label 
			pct_left = num_label_left/num_label_pix
			
			#calculate the change from previous percent left 
			change = previous_pct-pct_left
			
			#record some data about iterations and pixels left- not currently in use 
			times.update({count:pct_left})

			#let the process run until we start to reach an asymtote- when change from one iteration to 
			#the next dips below 0.01 (1% stop the process). Also check that we don't stop it on the first 
			#iteration because that has to be below 0. 
			if (change >= self.change_thresh) | (change < 0): 

				#apply the max filter again to grow the existing labels 
				filled_arr = LabelSupport(masked,filled_arr,ones).max_filtering() 
				count +=1
			else:
				#before writing to disk change all of the remaining pixels to -1 so they can't be confused with existing labels 
				filled_arr = np.where(filled_arr==1,-1,filled_arr)
				print(f'Between iteration change was less than {self.change_thresh*100} percent of pixels remain unlabeled, stopping...') 
				break 
		return filled_arr,times 
		
	def write_raster(self,input_arr): 
		#write it out driver = gdal.GetDriverByName("GTiff")
		#for the filename snip off the raster name and combine with the subset id so we know where it came from 
		if not os.path.exists(self.output_file): 

			#for some reason the inputs don't always have the right projection, make sure that's corrected 
			proj = osr.SpatialReference()
			proj.ImportFromEPSG(self.epsg)

			rows,cols = input_arr.shape #this shouldn't matter which you take from because they need to be the same size 

			ds = self.read_raster(self.classified_raster)[1]
			driver = gdal.GetDriverByName("GTiff")

			#create the output and write to disk 
			outdata = driver.Create(self.output_file, cols, rows, 1, gdal.GDT_Int16, #dtype and compression algo are hardcoded here 
				options=['COMPRESS=LZW'])
			outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
			outdata.SetProjection(proj.ExportToWkt())#sets the projection to a default, something werid was happening where the proj wasn't right 
			outdata.GetRasterBand(1).WriteArray(input_arr)
			#outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
			outdata.FlushCache() ##saves to disk!!
			outdata = None
			ds=None
	
def run_parallel(file,id_raster,output_dir,subset_id,binary_val): 
	
	output_file = os.path.join(output_dir,f'labeled_{os.path.split(file)[1]}') 

	if not os.path.exists(output_file):
		label_gen=LabelRaster(file,id_raster,output_file,subset_id,binary_val=binary_val) #binary_val is default set to 1, in the most current glacier data that is debris cover 7/5/2021
				
		#do the actual work of labeling
		output=label_gen.assign_glacier_id()

		#write the output to disk 
		label_gen.write_raster(output[0])

def main(classified_rasters,id_rasters,output_dir,binary_val): 
	time0 = time.time()
	
	#check if the output dir exists and if not make one
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)
	else: 
		pass

	#get the subset so we know which ID raster to grab 
	dirs = glob.glob(classified_rasters)
	dirs = [item for item in dirs if not 'labeled' in item] #remove erroneous dir
	print('working...')

	for child in dirs: 
	
		subset_id = child.split('/')[-2] #hardcoded for the structure of the files and directory 
		
		id_raster = [file for file in glob.glob(id_rasters+'*.tif') if subset_id in file][0] #get first (only) element of resulting list. This expects a file naming structure that comes from the bulk clipping script in ../scripts/4_postprocessing/
		
		rasters_2_label = glob.glob(child+'*.tif')
		
		#check if the output dir we want is there 
		subdir = os.path.join(output_dir,subset_id)
		if not os.path.exists(subdir): 
			os.mkdir(subdir)
	
		#run rasters in parallel 
		pool = multiprocessing.Pool(processes=25)
		label_it=partial(run_parallel, id_raster=id_raster,output_dir=subdir,subset_id=subset_id,binary_val=binary_val)
		#blast off 
		result_list = pool.map(label_it, rasters_2_label)

	print(f'That took {round((time.time()-time0)/60,2)} minutes')
		

if __name__ == '__main__':

	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		classified_rasters = variables['classified_rasters']
		id_rasters=variables['id_rasters']
		output_dir=variables['output_dir']
		binary_val=variables['binary_val']

		#note that the GlacierNet outputs are organized in child dirs so 
		#we append * to move through those child dirs recursively 
		classified_rasters = os.path.join(classified_rasters,'*/')

	main(classified_rasters,id_rasters,output_dir,int(binary_val))

