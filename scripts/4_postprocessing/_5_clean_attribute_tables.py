import pandas as pd 
import geopandas as gpd 
import glob
import os 
import sys 
import json
"""
Take the processes vector files and remove extraneous cols from processing then join with 
RGI attribute tables so GlacierNet outputs have RGI metadata. 
"""

def calculate_area_col(gdf): 
	gdf['GN_area'] = gdf['geometry'].area / 10**6 #this assumes that the field that is being created is in meters and you want to change to km2 
	return gdf 

def main(input_dir,attr_file): 
	#read in the RGI file for attributes
	attributes = gpd.read_file(attr_file)
	#remove extraneous cols from the RGI data 
	attributes.drop(columns=['start_yr','area_test','year','layer','path','geometry'],inplace=True) #this is hardcoded and will likely need to be adjusted 
	#rename the rgi id col to match GlacierNet outputs 
	attributes.rename(columns={'rgi_id':'rgi_label'},inplace=True)
	#read in GlacierNet outputs- these are either combined, debris free or debris covered 
	input_files = glob.glob(input_dir+'*.shp')
	#iterate through the files in the output dir 
	for file in input_files: 
		gdf = gpd.read_file(file)
		year = os.path.split(file)[1].split('_')[1] #hardcoded for final file naming structure, adjust if that changes 
		print(f'processing year {year}')
		#get the extra cols we want to remove from GlacierNet outputs 
		drop_cols = [col for col in gdf if not ('rgi_label' in col) | ('geometry' in col)]
		#remove cols we don't want in the outputs- these are leftover from processing 
		gdf.drop(columns=drop_cols,inplace=True)
		#join the two gdfs on their common column
		output = gdf.merge(attributes, on='rgi_label', how='left')
		#add an area col to the updated outputs 
		output = calculate_area_col(output)
		#remove the rgi_label col which was used for merging 
		output.drop(columns='rgi_label',inplace=True)
		#add a col which is the year of the composite
		output['GN_comp_year'] = year 
		#write the output to disk 
		output_fn = file[:-4]+'_attributes.shp'
		if not os.path.exists(output_fn): 
			output.to_file(output_fn)
		else: 
			print(f'{output_fn} already exists')
		
if __name__ == '__main__':

	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		input_dir = variables['input_dir']
		attr_file = variables['attr_file']

	main(input_dir,attr_file)