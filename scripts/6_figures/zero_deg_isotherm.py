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
import matplotlib 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from areal_change_figs import clip_each_model_year_for_plotting

def create_df_from_list(df_list,col): 
	"""Concat a list of dfs into one df and case a col to int."""
	df = pd.concat(df_list)
	df[col] = df[col].astype('int')
	
	return df 

def get_df_subset(df,search,col='file_yr'):
	try: 
		return df.loc[df[col]==search]
	except KeyError: 
		print(f'{col} does not exist in your df.')
		return None 

def calc_area(input_df,col='area'): 
	return round(input_df[col].sum(),2)

def export_basic_area_stats(input_df,output_fn): 
	"""Export a pandas df as csv to file."""
	if not os.path.exists(output_fn): 
		input_df.to_csv(output_fn)

def isotherm_fig(neg_dict, pos_dict, colors, **kwargs): 
	"""Make a figure that has two subplots, one for below zero deg isotherm, one for above."""

	fig,(ax1,ax2) = plt.subplots(2,sharex=True,sharey=True)

	font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

	matplotlib.rc('font', **font)
	count = 0 
	area_dict = {}

	for k,v in neg_dict.items(): #v is now a list of dfs (060720201)
		
		print('k is: ',k)

		#dict is like {region id: list of dfs} make the list of dfs into something more useful 

		try: 
			df = create_df_from_list(v,'file_yr')
		except KeyError: 
			print(f'{k} does not exist in the primary data')
			break
			
		#start the actual plotting
		
		scaler = MinMaxScaler()

		#get all the individual glaciers for one year and scale the data to 0-1 so they have the same y axes 
		try: 
			gdf = df.groupby('file_yr')['area'].sum().reset_index()
			gdf['sc_area'] = scaler.fit_transform(gdf['area'].values.reshape(-1,1))
		except (UnboundLocalError, ValueError): 
			print('something happened, going on') 

		try: 
			gdf.plot(x='file_yr',y='sc_area',ax=ax1,linewidth=2.5,color=colors.get(k),legend=False,label=k)
		except KeyError: 
			print(f'The {k} df was empty so we passed on it')
	
	for k,v in pos_dict.items(): 	
		#get the secondary product (debris cover)
		try: 
			df2 = create_df_from_list(v,'file_yr') 
		except KeyError as e: 
			print(f'{k} does not exist in the debris data. Destroying debris df.')
			del df2 

		try: 
			gdf2 = df2.groupby('file_yr')['area'].sum().reset_index()
			gdf2['scnd_area'] = scaler.fit_transform(gdf2['area'].values.reshape(-1,1))
		except (UnboundLocalError, ValueError): 
			print('There is no data for the second dataset, skipping scaling...')
			del gdf2

		try: 
			gdf2.plot(x='file_yr',y='scnd_area',ax=ax2,linewidth=2.5,color=colors.get(k),legend=False,label=k)
		except Exception as e: #not 100% sure which error this will raise 
			print(f'Tried to plot the second df, got the error: {e}. Continuing as a result...')


	#add labels
	ax1.annotate('a', xy=(0.9,0.9), xycoords='axes fraction',fontsize='x-large')
	ax2.annotate('b', xy=(0.9,0.9), xycoords='axes fraction',fontsize='x-large')
	ax1.set_xlabel(' ')
	ax2.set_xlabel(' ')
	ax1.set_title('Annual mean temp below zero C')
	ax2.set_title('Annual mean temp above zero C')
	ax2.tick_params(axis='x', labelsize=14)
	ax1.tick_params(axis='y', labelsize=14)
	ax2.tick_params(axis='y', labelsize=14)
	# Set common labels
	#fig.text(0.5, 0.04, 'common xlabel', ha='center', va='center')
	fig.text(0.1, 0.5, 'Scaled area', ha='center', va='center', rotation='vertical')
	ax2.legend(loc='lower left')
	ax1.grid(alpha=0.5)
	ax2.grid(alpha=0.5)
	plt.show()
	plt.close('all')

def main(glacier_change,**kwargs): 

	#make a plot of glacier change just between 1990 and 2020 for the national parks 
	#nps_change_map(glob.glob(glacier_change+'*.shp'),**kwargs) 

	#make a plot of glacier change for the eco-regions of AK 

	# tw_data = clip_each_model_year_for_plotting(glob.glob(glacier_change+'*.shp'),**kwargs)

	# tw_data = dict((k,v) for k,v in base_data.items() if v)


	neg = clip_each_model_year_for_plotting(glob.glob(glacier_change+'*update.shp'),**kwargs)

	neg = dict((k,v) for k,v in neg.items() if v)

	
	#modify the kwargs in place to get debris (different model and region)
	
	kwargs['region'] = 'neg_isotherm'

	
	pos=clip_each_model_year_for_plotting(glob.glob(glacier_change+'*update.shp'),**kwargs)
	pos=dict((k,v) for k,v in pos.items() if v)


	# print('base',base_data.keys())

	#print('debris',debris_data.keys())

	# colors = dict(zip(base_data.keys(),['#A9E5BB','#FCF6B1','#ECB009','#9FE2BF','#9D6B06','#F72C25','#92252A','#2D1E2F','#000000','#4c004c','#00007f']))
	# print(colors)

	colors = {'Aleutians': '#a6cee3', 'Central Panhandle': '#1f78b4', 'North Panhandle': '#b2df8a', 
	'Northwest Gulf': '#33a02c', 'Southeast Interior': '#fb9a99', 'South Panhandle': '#e31a1c', 
	'Bristol Bay': '#fdbf6f', 'Central Interior': '#ff7f00', 'Cook Inlet': '#cab2d6', 'Northeast Gulf': '#6a3d9a'}
	#plot_time_series_change_by_region(base_data,debris_data,colors,**kwargs)

	isotherm_fig(neg,pos,colors,**kwargs)

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		nps_bounds = variables['nps_bounds']
		ak_bounds = variables['ak_bounds']
		glacier_change = variables['glacier_change']
		output_dir = variables['output_dir']
		canada = variables['canada']
		russia = variables['russia']
		fig_dir = variables['fig_dir']
		ecoregions = variables['ecoregions']
		pickles = variables['pickles']
		model = variables['model']
		above_zero = variables['above_zero']
		below_zero = variables['below_zero']
		tidewater = variables['tidewater']
		mcnabb_tw = variables['mcnabb_tw']
		rgi = variables['rgi']

	main(tidewater,pickles=pickles,ak_bounds=ak_bounds,canada=canada,russia=russia,fig_dir=fig_dir,
	ecoregions=ecoregions,model=model,region='pos_isotherm',output_dir=output_dir,tw_list=mcnabb_tw,rgi=rgi) #here setting the region to all will not do isotherms, 
	#set to above_zero or below_zero for an isotherm run #None is just a placeholder for the glacier change data 