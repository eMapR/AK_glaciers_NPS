import numpy as np 
import os
import sys 
from osgeo import gdal
import geopandas as gpd 


def calculate_vector_area(input_vector,area_col='rev_area'):

	gdf = gpd.read_file(input_vector)
	gdf["rev_area"] = gdf['geometry'].area/ 10**6
	try: 
		return gdf['rev_area'].sum()
	except KeyError as e: 
		column = input('The Area column does not exist in this dataset\nplease enter the area column you want to use: ')
		return gdf[column].sum()

def calculate_diffs(pred,ref): 

	fp = gpd.read_file(pred).difference(gpd.read_file(ref))
	fn = gpd.read_file(ref).difference(gpd.read_file(pred))

	return fp,fn 

def main(fn,fp):#input_file1,ref,**kwargs):#,n_region,start_year,end_year,mid_year,other): 

	#southern region 
	#debris cover
	# model_output_2010 = calculate_vector_area(input_file1)
	# herreid = calculate_vector_area(ref1)
	# scherler = calculate_vector_area(ref2)

	# print('GlacierNet is: ')
	# print(model_output_2010)

	# print('herreid')
	# print(herreid)

	# print('scherler')
	# print(scherler)

	# model_output_2010 = calculate_vector_area(input_file1)
	# model_output_2016 = calculate_vector_area(input_file2)

	# print('model output for 2010 is: ')
	# print(model_output_2010)
	# print('model output for 2016 is: ')
	# print(model_output_2016)
	#model_output_2010 = calculate_vector_area(input_file1)
	# rgi = calculate_vector_area(ref)
	# print('overall 2006 area is: ')
	# print(model_output_2006)
	# print('overall 2008 area is: ')
	# print(model_output_2008)
	# print('overall 2010 area is: ')
	# print(model_output_2010)
	# print('rgi area is: ')
	# print(rgi)


	# #do fns and fps 
	# fp_2008=calculate_vector_area(kwargs.get('fp_2008'))
	# fn_2008=calculate_vector_area(kwargs.get('fn_2008'))
	fn=calculate_vector_area(fn)
	fp=calculate_vector_area(fp)

	# print('2008 fp are: ')
	# print(fp_2008)
	# print('2008 fn are: ')
	# print(fn_2008)
	print('fn are: ')
	print(fn)
	print('fp are: ')
	print(fp)

	#northern region 
	# model_area = calculate_vector_area(input_file1)
	# rgi_area = calculate_vector_area(ref)
	# fn = calculate_vector_area(kwargs.get('fn'))
	# fp = calculate_vector_area(kwargs.get('fp'))

	# print('model is: ')
	# print(model_area)
	# print('rgi is: ')
	# print(rgi_area)
	# print('false neg are: ')
	# print(fn)
	# print('false pos are: ')
	# print(fp)

	# start=calculate_vector_area(start)
	# end=calculate_vector_area(end)

	# print(end-start)
if __name__ == '__main__':

	#southern region 
	# model_output_2006 = "/vol/v3/ben_ak/vector_files/final_error_assessment/combined/08122021_2006_combined_clipped_to_northern_test_revised.shp"
	# model_output_2008 = "/vol/v3/ben_ak/vector_files/final_error_assessment/combined/08122021_2008_combined_clipped_to_northern_test_revised.shp"
	# model_output_2010 = '/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2010_debris_covered_clipped_to_southern_region_revised.shp'
	# model_output_2016 = '/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2016_debris_covered_clipped_to_southern_region_revised.shp'
	# main(model_output_2010,model_output_2016)
	# rgi = "/vol/v3/ben_ak/vector_files/final_error_assessment/rgi_clipped_to_northern_test_partitions.shp"
	# # fn_2008 = "/vol/v3/ben_ak/vector_files/neural_net_data/validation/southern_region/03272021_model/03272021_model_year_2008_false_negatives.shp"
	# # fp_2008 = "/vol/v3/ben_ak/vector_files/neural_net_data/validation/southern_region/03272021_model/03272021_model_year_2008_false_positives.shp"
	# fn_2010 = '/vol/v3/ben_ak/vector_files/final_error_assessment/combined/2010_southern_region_false_positives_revised.shp'
	# fp_2010 = '/vol/v3/ben_ak/vector_files/final_error_assessment/combined/2010_southern_region_false_negatives_revised.shp'
	# fn = "/vol/v3/ben_ak/vector_files/final_error_assessment/combined/2010_southern_region_false_negatives.shp"
	# fp = "/vol/v3/ben_ak/vector_files/final_error_assessment/combined/2010_southern_region_false_positives.shp"
	# #debris covered glacier (this only applies to the southern region for the 8/12/2021) model 
	# debris_covered_2010 = "/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2010_debris_covered_clipped_to_southern_region.shp"
	# debris_covered_2016 = "/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2016_debris_covered_clipped_to_southern_region.shp"
	# herreid = "/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/Herreid_1km_min_clipped_to_southern_test.shp"
	# scherler = "/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/Scherler_LS8_NDSI_clipped_to_southern_test.shp"

	# fn_2010 = "/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2010_model_scherler_fn.shp"
	# fn_2016 = "/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2016_model_scherler_fn.shp"
	# fp_2010 = "/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2010_model_scherler_fp.shp"
	# fp_2016 = "/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2016_model_scherler_fp.shp"

	# herreid_scherler = "/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/herreid_difference_scherler.shp"
	# scherler_herreid = "/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/scherler_difference_herreid.shp"
	# #main(model_output_2008,model_output_2010,rgi,fn_2008=fn_2008,fp_2008=fp_2008,fn_2010=fn_2010,fp_2010=fp_2010)#rgi_south,rgi_north,start_year,end_year,mid_year,other)

	# #northern region 
	# # model_output_circa_2008 = '/vol/v3/ben_ak/vector_files/neural_net_data/outputs/04222021_model_run/northern_region/04222021_model_run/northern_region_clipped_to_test_partitions.shp'
	# # rgi = '/vol/v3/ben_ak/vector_files/glacier_outlines/rgi/rgi_clipped_to_northern_region_test_partitions.shp'
	fn_2010 = '/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2010_model_herreid_fn_revised.shp'
	fn_2016 = '/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2016_model_herreid_fn_revised.shp'

	fp_2010 = '/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2010_model_herreid_fp_revised.shp'
	fp_2016 = '/vol/v3/ben_ak/vector_files/final_error_assessment/debris_covered/2016_model_herreid_fp_revised.shp'

	main(fn_2010,fp_2010)#herreid_scherler,scherler_herreid)

	#change 
	# start = '/vol/v3/ben_ak/vector_files/neural_net_data/testing/1990_merged_kfnp.shp'
	# end = '/vol/v3/ben_ak/vector_files/neural_net_data/testing/2020_merged_kfnp.shp'
	#main(debris_covered_2016,herreid,scherler)