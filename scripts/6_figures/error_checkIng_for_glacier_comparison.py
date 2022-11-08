import os
import sys
import pandas as pd 
import glob 
import geopandas as gpd 


if __name__ == '__main__':
	#get the outlier metatdata 
	metadata = pd.read_csv("/vol/v3/ben_ak/excel_files/glacier_wise_comparison/outliers.csv")
	#get the rgi shapefile
	rgi = gpd.read_file("/vol/v3/ben_ak/vector_files/neural_net_data/NSIDC_dataset_submission/final_dataset/overall/AK_2010_overall_glacier_covered_area_buffer.shp")
	#select the rgi polygons that are included in the outliers 
	rgi = rgi.loc[rgi['RGIId'].isin(metadata['RGIId'])]
	print(metadata.shape)
	print(rgi.shape)
	#now merge them 
	output = rgi.merge(metadata, on='RGIId',how = 'inner')
	output_dir = "/vol/v3/ben_ak/vector_files/neural_net_data/validation/glacierwise_comparison/"
	out_fn = os.path.join(output_dir,'outlier_selection_GlacierNet_2010_polygons.shp')
	if not os.path.exists(out_fn): 
		output.to_file(out_fn)