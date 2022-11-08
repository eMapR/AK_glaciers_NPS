import geopandas as gpd 



def clip_to_aoi(clip_shape,aoi,year,output_dir=None): 
	"""Clip a shapefile with geopandas and return gdf."""
	if output_dir: #this is an instance where we're writing the clipped shapefiles to disk
		print('here')
		output_fn = os.path.join(output_dir,f'model_year_{year}_park_{aoi["UNIT_NAME"].iloc[0].split(" ")[0]}.shp')
		if not os.path.exists(output_fn):
			clipped=gpd.clip(clip_shape,aoi) 
			if not clipped.empty: 
				clipped.to_file(output_fn)
			return clipped
		else: 
			print('reading from file')
			return gpd.read_file(output_fn)
	else: 
		clipped = gpd.clip(clip_shape,aoi)
		if not clipped.empty: 
			return clipped
		else: 
			print(f'That shape ({clip_shape} is empty so we pass.')


def add_area_col(gdf,min_size=0.01): 
	try: 
		gdf['area'] = gdf['geometry'].area / 10**6 #this assumes that the field that is being created is in meters and you want to change to km2 
		gdf = gdf.loc[gdf['area']>=min_size]

		return gdf
	except TypeError: 
		print('It appears there is an empty df here as there was a noneType error')








def main(): 
	#get climate regions
	c_divs = '/vol/v3/ben_ak/vector_files/ak_climate_regions/climate_divisions_watershed_based_EPSG_3338.shp'

	#get test GlacierNet file
	gn_file = '/vol/v3/ben_ak/vector_files/neural_net_data/outputs/08122021_model_run/revised_outputs/merged_outputs_combined/08122021_2020_combined_output_attributes.shp'



if __name__ == '__main__':
	main()