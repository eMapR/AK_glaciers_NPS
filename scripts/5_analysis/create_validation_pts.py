import pandas as pd 
import numpy as np 
import os  
import sys
import glob 
import geopandas as gpd 	


import codecs

# doc = codecs.open('document','rU','UTF-16') #open for reading with "universal" type set

# df = pandas.read_csv(doc, sep='\t')

def main(input_shp,output_dir): 
	if input_shp.endswith('shp'): 
		gdf = gpd.read_file(input_shp)
	elif input_shp.endswith('csv'): 
		try: 
			print('trying')
			df = pd.read_csv(input_shp)
			print(df.head())
			gdf = gpd.GeoDataFrame(
	    	df, geometry=gpd.points_from_xy(df.lon, df.lat))	
			print('fucked up')
		except AttributeError: 
			print('here')
			doc = codecs.open(input_shp,'rU','UTF-16') #open for reading with "universal" type set
			#df = pandas.read_csv(doc, sep='\t')

			df = pd.read_csv(input_shp,sep='\t')
			print(df.head())
			gdf = gpd.GeoDataFrame(
	    	df, geometry=gpd.points_from_xy(df.lon, df.lat))	

	gdf=gdf.sample(frac=1)
	print(gdf)
	gdf.to_file(os.path.join(output_dir,os.path.split(input_shp)[1][:-4]+'_shuffled.shp'))
	
if __name__ == '__main__':
	shp="/vol/v3/ben_ak/vector_files/neural_net_data/validation/southern_region_pts_revised.shp"
	northern_shp = "/vol/v3/ben_ak/excel_files/validation/northern_region_photo_interp_pts_combined_certainties_revised.csv"
	output_dir = "/vol/v3/ben_ak/vector_files/neural_net_data/validation/"
	main(northern_shp,output_dir)


