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
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None  # default='warn'

def plot_change_by_elevation(input_df,**kwargs): #adjust inputs as needed 
	"""Make a plot that has a subplot for every year and then scaled temp and area by elev band."""
	fig,axs = plt.subplots(3,6, figsize=(8,6),
							sharex=True,sharey=True,
							gridspec_kw={'wspace':0,'hspace':0.1})

	years = range(1986,2022,2)
	axs = axs.flatten()
	scaler = MinMaxScaler()

	for year,ax in zip(years,axs): 
		plot_df = input_df.loc[input_df['year'] == year]
		# plot_df['sc_area'] = scaler.fit_transform(plot_df['area'].values.reshape(-1,1))
		# plot_df['sc_temp'] = scaler.fit_transform(plot_df['mean_temp'].values.reshape(-1,1))
		plot_df.plot(x='elev_band_max',y='area',ax=ax,c='red',linewidth=2,legend=False)
		ax.annotate(f'{year}',xycoords='axes fraction',xy=(0.3,0.85),fontsize=14)
		#plot temp on a second y axis 
		ax1 = ax.twinx()
		ax1.set_ylim([-25,5])
		plot_df.plot(x='elev_band_max',y='mean_temp',ax=ax1,c='black',linewidth=2,legend=False)
		#remove erroneous ticks and labels 
		if year not in [1996,2008,2020]: 
			ax1.yaxis.set_visible(False)
			ax1.tick_params(axis='y',right=False,left=False)
		if year not in [1986,1998,2010]:
			ax.tick_params(axis='y',right=False,left=False)
		if year not in [2010,2012,2014,2016,2018,2020]: 
			ax.tick_params(axis='x',bottom=False)
			ax1.tick_params(axis='x',bottom=False)
		#set the actual axis labels 
		ax1.set_xlabel('')
		ax.set_xlabel('')
		ax.grid(axis='y',alpha=0.25)
		#label the common x axis  
		fig.text(0.5, 0.05, 'Elevation band max (m)', ha='center',size='large')
		#label the area y axis 
		fig.text(0.03, 0.5, 'Glacier area (km2)', va='center', rotation='vertical',size='large')
		#label the temp y axis 
		fig.text(0.96, 0.5, 'Mean annual temp (deg C)', va='center', rotation='vertical',size='large')
	plt.show()
	plt.close()

	# output_fn = os.path.join(kwargs.get('fig_dir'),f'mean_temp_area_elevation_bands_NN_draft4.jpg')
	# if not os.path.exists(output_fn): 
	# 	plt.savefig(output_fn,dpi=500, 
	# 		bbox_inches = 'tight',
	#     	pad_inches = 0.1)
	# else: 
	# 	print('That file already exists')


if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		output_dir = variables['output_dir']
		dem = variables['dem']
		pickles = variables['pickles']
		ecoregions = variables['ecoregions']
		fig_dir = variables['fig_dir']
		data = "/vol/v3/ben_ak/excel_files/stats_for_isotherm_figs/area_temp_by_500m_elev_band_and_year_08122021_model_revised_resampling.csv"
		plot_change_by_elevation(pd.read_csv(data),fig_dir=fig_dir)























