import os 
import re
import sys 
import json 
import glob 
import pickle
import pandas as pd 
import matplotlib
import numpy as np 
from sklearn import preprocessing
import geopandas as gpd 
import matplotlib.pyplot as plt 
import collections
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""Clip a list of vectors to the features of a shapefile and then pickle a dict of those results.
   Inputs: 
   -shapefile of aois, the default should be AK climate divs 
   -a list or directory of vector files you want to clip, the default should be the outputs of GlacierNet
   Ouput: 
   -a pickled object that is a dict like {'climate div':clipped object}

"""


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

def clip_each_model_year_for_plotting(change_data,region,**kwargs): 
	"""Clip time series to regions and save as a dictionary like {region:gdf} as pickle."""
	try: 
		eco = gpd.read_file(kwargs.get('ecoregions')).set_crs('EPSG:3338')

	except Exception as e: 
		print('Tried to set the crs for the ecoregions')
		print(kwargs.get('ecoregions'))
		eco = gpd.read_file(kwargs.get('ecoregions')).to_crs('EPSG:3338')
	
	#this section should just have to be run once for a model
	if region =='debris': 
		print('reading debris data')
		output_fn = os.path.join(kwargs.get('pickles'),f"{kwargs.get('model')}_dictionary_like_ecoregion_list_of_shps_clipped_to_debris_watersheds_revised.p")
	
	elif region =='all': 
		print('reading primary data')
		#this filename is for running all the climate divisions in the full area
		output_fn=os.path.join(kwargs.get('pickles'), f"{kwargs.get('model')}_dictionary_like_ecoregion_list_of_shps_clipped_to_combined_watersheds_revised.p")
		
	count = 0 
	if not os.path.exists(output_fn): 
		plot_dict = {}
		#iterate through the ecoregions
		for region in eco['Name'].unique():
			print('The current region is: ',region)
			#if (region == 'Northeast Gulf') | (region == 'Southeast Interior'): 
			#define a list to hold a time series 
			region_list = []

			#changed this section 6/18/2021 so that each of the SE panhandle sections run separately 
			#if not 'Panhandle' in region: 
			gdf = eco.loc[eco['Name']==region]

			# #there are three sections of the SE pandhandle, combine them for ease
			# elif 'Panhandle' in region: 
			# 	gdf = eco.loc[eco['Name'].str.contains('Panhandle')]

			# #check if a panhandle group is already in the dictionary, if it is pass this one 
			# proceed = {k:v for k,v in plot_dict.items() if 'Panhandle' in k}
			# print(proceed)
			
			# if (len(proceed)==0): 

			for file in change_data: #iterate through the files in the time series- probably not the fastest way to do this but it only has to be done once per model 
				print(f'The current file being processed is: {file}')


				year = re.findall('(\d{4})', os.path.split(file)[1][8:])[0] #gets a list with the start and end of the water year, take the first one. expects files to have the 8 digit model date first 
				
				print(year)
				try: 
					#do the actual clipping and catch instances where there is an empty df (no glaciers)
					clipped = add_area_col(clip_to_aoi(gpd.read_file(file).set_crs('EPSG:3338'),gdf,year))
					
					#add a year col for when it goes into the output and gets concatenated 
					clipped['file_yr'] = year
					region_list.append(clipped)

				except Exception as e: 
					print(f'Tried to add a col but could not because of a {e} error.')
			
					break #if the first one is nonetype its likely because there are no glaciers in that group, if that's the case skip the rest
			
			try: 
				#add the clipped gdf to the output dict with the regional name as key 
				plot_dict.update({region:region_list})

			except Exception as e: 
				print('Tried to update the clip dict and had a noneType')
				print(f'The error was: {e}')

			#increment the count to check for inclusion of panhandle group 
			count +=1

		#pickle it 
		print('Pickling to disk...')
		pickle.dump(plot_dict, open(output_fn,'wb'))
		print('Successfully pickled')
		return plot_dict

	else: 
		#in the case that this has already been run and the pickle exists just read it and create a new variable to hold it 
		print('That dictionary file already exists, trying to read from disk')
		try: 
			plot_dict = pickle.load( open(output_fn,'rb'))
			
			return plot_dict
		except Exception as e: 
			print('Tried to load the dictionary from disk but encountered the following error: ', e)



