import os 
import sys
import glob
import geopandas as gpd
import pandas as pd 
from glob import iglob
import re

"""Take a time series of data in shapefile format, makesure areas are calculated and then make time series of these data. Export as csvs."""


def check_area_col(gdf): 
	"""Read a shp file and make sure it has the right area col from post clipping."""
	if not 'clip_area' in gdf.columns: 
		gdf['clip_area'] = gdf.area 
	return gdf

def make_time_series(fn,col='clip_area'): 
	gdf = gpd.read_file(fn)
	#check to make sure the area col is there 
	gdf = check_area_col(gdf)
	#get just the name of the file w/o the rest of the fp and then grab the year 
	year = re.findall('(\d{4})',os.path.split(fn)[1])[0]
	#areas got calculated in m2, convert to km2
	return year, round((gdf[col].sum()/1000000), 2)


if __name__ == '__main__':
	input_dir = "/vol/v3/ben_ak/vector_files/glacier_outlines/GN_brooks_range/"
	output_dir = "/vol/v3/ben_ak/excel_files/areal_change_stats/brooks_range/"

	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)

	#here a subdir is an area for a glacier type. It contains all the years. 
	for subdir, dirs, files in os.walk(input_dir):
		file_paths = glob.glob(subdir+'/'+'*.shp')
		#calculate total area for a year, check area col and then split into 
		#two lists for each area. 
		outputs = [make_time_series(x) for x in file_paths]
		years = [y[0] for y in outputs]
		areas = [i[1] for i in outputs]
		df = pd.DataFrame.from_dict({'year':years,'area':areas}).sort_values('year')
		print(df)
		if not df.empty: 
			out_fn = os.path.join(output_dir,'brooks_range_areas.csv')
			print(out_fn)
			df.to_csv(out_fn)
	    # for file in files:
	    #     print('subdir is: ',subdir)
	    #     print('dirs is: ', dirs)
	    #     #print(os.path.join(subdir, file))