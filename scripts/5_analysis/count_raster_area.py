import numpy as np 
import os
import sys 
from osgeo import gdal
import geopandas as gpd 

np.seterr(over='raise')

def convert_pixels_to_area(pixels,resolution):
	'''Get number of pixels and convert to sq km.'''
	print('pixels are: ',pixels)
	return (pixels*resolution*resolution)/1000000

def sum_pixels_in_binary_raster(input_raster): 
	'''Read in a raster and get the sum (pixel count). Needs to be a binary raster.'''
	ds = gdal.Open(input_raster)
	band = ds.GetRasterBand(1) #hardcoded for the first band- assume it only has one band if binary 
	arr = band.ReadAsArray() 
	#arr[arr!=0] = 1
	arr[arr != 1] = 0 #np.where(arr,1,0)
	# print(arr.max())
	print(arr.shape)

	return np.sum(arr)

def set_raster_values_to_nan(input_raster,input_value): 
	'''Convert raster values to nan.'''
	output_file = input_raster[:-4]+'_no_data.tif'
	ds = gdal.Open(input_raster)
	band = ds.GetRasterBand(1) #hardcoded for the first band- assume it only has one band if binary 
	arr = band.ReadAsArray().astype('float') 
	if input_value: 
		arr[arr!=input_value] = np.nan
	else: 
		arr[arr!=1] = np.nan
	rows,cols = arr.shape
	driver = gdal.GetDriverByName("GTiff")
	outdata = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32)
	outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
	outdata.SetProjection(ds.GetProjection())##sets same projection as input
	outdata.GetRasterBand(1).WriteArray(arr)
	#outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
	outdata.FlushCache() ##saves to disk!!
	outdata = None
	band=None
	ds=None

#cnn_sum = convert_pixels_to_area(sum_pixels_in_binary_raster("/vol/v3/ben_ak/raster_files/neural_net/outputs/area_analysis/03272021_model_output_2008_times_rgi_southern_region_5km_buff.tif"),30)
# sub_15 = convert_pixels_to_area(sum_pixels_in_binary_raster("/vol/v3/ben_ak/raster_files/neural_net/outputs/04222021_model_run/areas/2020_sub_15_binary.tif"),30)
# sub_16 = convert_pixels_to_area(sum_pixels_in_binary_raster("/vol/v3/ben_ak/raster_files/neural_net/outputs/04222021_model_run/areas/2020_sub_16_binary.tif"),30)
pred = convert_pixels_to_area(sum_pixels_in_binary_raster('/vol/v3/ben_ak/raster_files/neural_net/outputs/03272021_model_run/subset_10/full_time_series_year_2008_2009_class_label_int_multiband_subset_10.tif.tif'),30)#("/vol/v3/ben_ak/raster_files/neural_net/outputs/validation/2008_clipped_to_southern_test_partitions_SE_border_removed.tif"),30)
#ref = convert_pixels_to_area(sum_pixels_in_binary_raster("/vol/v3/ben_ak/raster_files/neural_net/outputs/validation/rgi_southern_region_clipped_to_test_partitions_no_SE_border.tif"),30)
#diff = convert_pixels_to_area(sum_pixels_in_binary_raster("/vol/v3/ben_ak/raster_files/neural_net/outputs/validation/03272021_2008_minus_rgi_noSE_border.tif"),30)
#print(diff)
print(pred)
#print(ref)
#print((pred-ref)/ref)
#print(cnn_sum)
#stem_sum = convert_pixels_to_area(sum_pixels_in_binary_raster("/vol/v3/ben_ak/raster_files/model_comparison/rgi_binary_US_times_2001_STEM_model_probablity_binary_output.tif"),30)
#rgi_sum = convert_pixels_to_area(sum_pixels_in_binary_raster("/vol/v3/ben_ak/raster_files/rgi/rgi_epsg_3338_clipped_US_bounds.tif"),30)
# rgi_sum = gpd.read_file("/vol/v3/ben_ak/vector_files/glacier_outlines/rgi_epsg_3338_clipped_to_northern_region_US.shp")
# print('area')
# print(rgi_sum['Area'].sum())
# print(rgi_sum.head())

# print(rgi_sum.columns)
#print(rgi_sum)
#print(cnn_sum,stem_sum,rgi_sum)

# set_raster_values_to_nan('/vol/v3/ben_ak/raster_files/neural_net/outputs/optical_topo_predictors_composite_2001_full_study_area_w_class_label_sorted_clipped_to_wrangall_subset.tif',None)
# print('done')