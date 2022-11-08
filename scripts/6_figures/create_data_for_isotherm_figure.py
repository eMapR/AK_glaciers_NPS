import rasterio 
import os
import sys 
import pandas as pd 
import json 
import numpy as np 
import numpy.ma as ma
import glob
from annual_zero_deg_isotherm import multiply_rasters
import fiona
import rasterio
import rasterio.mask

"""Calculate stats for elevation bands defined by AK IfSAR data for the GlacierNet outputs. Stats include area with elevation bands and mean annual temp at elevation 
bands. These are calculated for each biannual composite in the dataset. 

Inputs: 
-Elevation raster 
-GlacierNet outputs (raster)
-Temperature data (here from Daymet)

Output: 
-CSV with stats for each of these params at elevation bands. Use this as the input for plot_area_change_temp_by_elev.py 

"""

def run_elev_ranges(arr, temp_arr, resolution, year,band_size): 
	output = {}
	years = []
	elevs = []
	temps = []
	areas = []
	std = []
	count = 0 
	#create the elev ranges
	ls1 = list(range(0,5000,band_size))
	ls2 = list(range(band_size,5100,band_size))
	count = 0 
	ls1 = [x+1 for x in ls1]
	ls1[0] = 0
	elev_ls = list(zip(ls1,ls2))

	for item in elev_ls: 
		if count != 0: 
			#create a mask to get the data from the right elevation band 
			elev_mask = np.where((arr >= item[0]) & (arr <= item[1]),1,0).reshape(arr.shape[0],arr.shape[1])
		else: 
			elev_mask = np.where((arr > item[0]) & (arr <= item[1]),1,0).reshape(arr.shape[0],arr.shape[1])

		#get a count of pixels with pos elevations- this doesn't account for the small number of pixels that have a value below zero
		pix_count = np.count_nonzero(elev_mask)	
		#mask the temp arr by elev band so when we take the mean we're just getting pixels in that elev band 
		mx = temp_arr*elev_mask
		mx = np.where(mx != 0, mx, np.nan)
		area = (pix_count*resolution**2)/1000000
		#put the results into the lists 
		years.append(year)
		elevs.append(item[1])
		areas.append(area)
		#when there are no pixels (area) in an elev band the last valid temp is being infilled. 
		#just put zero in those cases because there is no area at that elev we care about. 
		if (area > 0): 
			temps.append(np.nanmean(mx))
			std.append(np.nanstd(mx))
		else: 
			temps.append(0)
			std.append(np.nanstd(mx))
		count += 1
	output.update({'year':years, 'elev_band_max':elevs, 'area':areas, 'mean_temp':temps, 'std':std})
	return pd.DataFrame(output)

def get_area_temp_stats_by_elev(elev_files,temp_files,band_size,resolution=1000,band=1): #input list should be the start and end rasters in the order you want them run 
	"""Make a plot(s) of change by elevation for a region or by glacier?"""
	output_list = []
	
	for elev_raster in elev_files: 
		#print(elev_raster)
		year = os.path.split(elev_raster)[1].split('_')[2] #this is hardcoded for 8/8/2021 elev file naming structure 
		#print(year)
		with rasterio.open(elev_raster) as dst: 
			arr = dst.read(band) #assumes a 1 band raster
			arr[arr < 0] = 0
			
			#get temp file with year matching elev
			try: 
				temp_raster = [file for file in temp_files if year in file][0]
			except IndexError: 
				print(f'It looks like there is a file missing for the year {year}')

			with rasterio.open(temp_raster) as temp_dst: 
				temp_arr = temp_dst.read(band)
				temp_arr[temp_arr < -100] = 0 #this is hardcoded to an arbitrarly low number. This is to remove noData vals but allow neg avg annual temps through 
				df = run_elev_ranges(arr, temp_arr, resolution, year,band_size)
				output_list.append(df)
	return pd.concat(output_list)

def main(input_dir,output_dir,daymet_dir,isotherm,**kwargs): 
	
	######################################################################################
	#use this to create the actual output df with elev ranges, years, mean temps and areas 
	#get the elevation files 
	elev_files = sorted(glob.glob("/vol/v3/ben_ak/raster_files/daymet/08122021_model_run_data/revised_resampling/GlacierNet_outputs_w_elevation/" + '*.tif'))
	#get the avg temperature files 
	temp_files = sorted(glob.glob("/vol/v3/ben_ak/raster_files/daymet/08122021_model_run_data/revised_resampling/GlacierNet_outputs_w_temp/" + '*.tif'))

	#generate some stats
	out_fn = os.path.join(output_dir,f'area_temp_by_{kwargs.get("band_size")}m_elev_band_and_year_08122021_model_revised_resampling.csv')
	if not os.path.exists(out_fn): 
		output = get_area_temp_stats_by_elev(elev_files,temp_files,kwargs.get('band_size'))
		output.to_csv(out_fn)

	print(f'stats written to: {output_dir}')
	
if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f: 
		variables = json.load(f)
		input_dir = variables['input_dir']
		output_dir = variables['output_dir']
		daymet_dir = variables['daymet_dir']
		dem = variables['dem']

	main(input_dir,
		 output_dir,
		 daymet_dir,
		 isotherm='all',
		 dem=dem, 
		 band_size=200
		 )	 