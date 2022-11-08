import os 
import sys
import pandas as pd 
import geopandas as gpd
import glob
import numpy as np 
import re 
"""For the submission of GlacierNet glaciers to GLIMS we need to take out glaciers that cross the border into Canada. This is currently being accomplished by 
creating a buffer of the Canadian border and removing glaciers that cross that border. 
"""

if __name__ == '__main__':
	#this should be the dir of files you want to truncate
	intersect_bounds = "/vol/v3/ben_ak/vector_files/boundaries/canada_epsg_3338_clipped_1000m_buffer.shp"
	#after running the overall shapes we should just use the ones that are in the overall data, otherwise there will be a disconnect between the repos
	overall_shps = "/vol/v3/ben_ak/vector_files/neural_net_data/NSIDC_dataset_submission/final_dataset/GLIMS_submission/overall_area/"
	input_dir = "/vol/v3/ben_ak/vector_files/neural_net_data/NSIDC_dataset_submission/final_dataset/AK_glaciers_G10040-V001/supraglacial_debris_area/"
	output_dir = "/vol/v3/ben_ak/vector_files/neural_net_data/NSIDC_dataset_submission/final_dataset/GLIMS_submission/supraglacial_debris_area/"

	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)

	files = sorted(glob.glob(input_dir+'*.shp'))
	base_files = sorted(glob.glob(overall_shps+'*.shp'))
	#add an additional buffer to the to boundary shp file
	#bounds = gpd.read_file(intersect_bounds).buffer(90)

	for overall,f in zip(base_files,files): 
		#expects the year to be the only number in the file name 
		year = re.findall('(\d{4})',os.path.split(overall)[1])[0]
		base_df = gpd.read_file(overall)
		print(base_df.columns)
		gdf = gpd.read_file(f)
		#double check that the years match
		if year in os.path.split(f)[1]: 
			gdf = gdf.loc[gdf['RGIId'].isin(base_df['RGIId'])]
		else: 
			print('Trying to process: ')
			print(overall)
			print(f)
		#do the intersect- just use for the base overall area files
		# df1 = gpd.sjoin(gpd.read_file(intersect_bounds),gdf,how="right",op="intersects")
		# #the shapes we want will have something valid in the 'index_left' col which is produced from the join
		# df1 = df1.loc[df1['index_left'] != 0.0] #0.0 is the default value in this field
		
		output_fn = os.path.join(output_dir,os.path.split(f)[1])
		if not os.path.exists(output_fn): 
			print(f'writing {output_fn} to file')
			gdf.to_file(output_fn)
		else: 
			print(f'The file: {output_fn} already exists')
