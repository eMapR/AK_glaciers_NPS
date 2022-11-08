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
import matplotlib.pyplot as plt

def prep_raster_layer(input_raster,input_shape,output_dir): 
	"""Mask a raster to a shapefile area."""

	gdf = gpd.read_file(input_shape)
	#print(gdf.Name)
	with fiona.open(input_shape,'r') as shapefile: 
		shapes = [feature["geometry"] for feature in shapefile]
	#	print(shapes[0])
		with rasterio.open(input_raster) as src:
		    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
		    out_meta = src.meta

		    out_meta.update({"driver": "GTiff",
		                 "height": out_image.shape[1],
		                 "width": out_image.shape[2],
		                 "transform": out_transform})
		    print(out_image.shape)
	# with rasterio.open(os.path.join(output_dir,'testing.tif'), "w", **out_meta) as dest:
	#     dest.write(out_image)

def run_elev_ranges(arr,resolution): 
	plotting = {}
	count = 0 
	for item in [(0,500),(501,1000),(1001,1500),(1501,2000),(2001,2500),(2501,3000),(3001,3500),(3501,4000),(4001,4500),(4501,5000)]: 
		if count != 0: 
			subset = np.count_nonzero((arr >= item[0])&(arr<=item[1]))
		else: 
			subset = np.count_nonzero((arr > item[0])&(arr<=item[1]))

		plotting.update({item[1]:(subset*resolution**2)/1000000})
		count +=1 
	return plotting

def get_pixel_counts(input_list,output_dir,resolution=30): #input list should be the start and end rasters in the order you want them run 
	"""Make a plot(s) of change by elevation for a region or by glacier?"""
	output_list = []
	for raster,label in zip(input_list,['above_start','above_end','below_start','below_end']): 
		with rasterio.open(raster) as dst: 
			arr = dst.read(1) #assumes a 1 band raster
			arr[arr < 0] = 0
			print('running: ',label)
			print('overall area: ')
			print((np.count_nonzero(arr)*resolution**2)/1000000) 
			output_dict = run_elev_ranges(arr,resolution)
			
			output_list.append(output_dict)
	#write out the data for writing 
	output_fn = os.path.join(output_dir,'isotherm_change_with_elevation.csv')
	if not os.path.exists(output_fn): 
		df = pd.DataFrame(output_list)
		df.to_csv(output_fn)
	return output_list

def plot_change_by_elevation(dict_list,fig_dir): 

	fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True)
	count = 0 
	for x1,x2 in zip(dict_list[:2],dict_list[2:]): 
		df1 = pd.DataFrame(x1.items(), columns=['upper_elev', 'area'])
		df2 = pd.DataFrame(x2.items(), columns=['upper_elev', 'area'])
		if count == 0: 
			color = 'black'
			label = '1988'
		elif count > 0: 
			color = 'darkred'
			label = '2020'
		df1.plot(x='upper_elev',y='area',ax=ax1,color=color,linewidth=2.5,legend=False,label=label)
		df2.plot(x='upper_elev',y='area',ax=ax2,color=color,linewidth=2.5,legend=False,label=label)

		count += 1 

	#add grids
	ax1.grid(alpha=0.5)
	ax2.grid(alpha=0.5)
	#add letter identifiers 
	ax1.annotate(f'{chr(97)}',xy=(0.1,0.9),xycoords='axes fraction',size='x-large',weight='bold')
	ax2.annotate(f'{chr(97+1)}',xy=(0.1,0.9),xycoords='axes fraction',size='x-large',weight='bold')
	#add captions that encapsulate both axes 
	fig.text(0.5, 0.02, 'Elevation (m asl)', ha='center', va='center',size='large')
	fig.text(0.02, 0.5, 'Glacier area (km2)', ha='center', va='center', rotation='vertical', size='large')
	for ax in (ax1,ax2): 
		ax.set_xlabel(' ')
		ax.set_ylabel(' ')
	ax2.legend()

	output_fn = os.path.join(fig_dir,'zero_deg_isotherm_w_elevation_draft3.jpg')
	#plt.savefig(output_fn,dpi=400)
	plt.show()
	plt.close()


if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		output_dir = variables['output_dir']
		dem = variables['dem']
		pickles = variables['pickles']
		ecoregions = variables['ecoregions']
		fig_dir = variables['fig_dir']
		above_zero_1988 = "/vol/v3/ben_ak/raster_files/neural_net/isotherms/1988_glaciers_w_elevations_prism_above_zero.tif"
		above_zero_2020 = "/vol/v3/ben_ak/raster_files/neural_net/isotherms/2020_glaciers_w_elevations_prism_above_zero.tif"
		below_zero_1988 = "/vol/v3/ben_ak/raster_files/neural_net/isotherms/1988_glaciers_w_elevations_prism_below_zero.tif"
		below_zero_2020 = "/vol/v3/ben_ak/raster_files/neural_net/isotherms/2020_glaciers_w_elevations_prism_below_zero.tif"
		
		counts = get_pixel_counts(list([above_zero_1988, above_zero_2020, below_zero_1988, below_zero_2020]),output_dir,resolution=800)

		plot_change_by_elevation(counts,fig_dir)
