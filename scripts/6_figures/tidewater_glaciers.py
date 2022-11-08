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
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from areal_change_figs import clip_each_model_year_for_plotting


def clean_gdf(gdf,source): 
	
	try: 
		for col in gdf.columns: 
			if 'area' in col: 
				gdf.drop(columns=col,inplace=True)
	except KeyError: 
		pass
	gdf[f'{source}_area'] = (gdf['geometry'].area)/10**6

	gdf = gdf.loc[gdf[f'{source}_area']>1.0]

	return gdf

def make_tw_plot(input_dict,colors,**kwargs): 
	"""Should be a plot that is like glacier size vs loss? """

	fig,axes = plt.subplots()

	#read in some shapefiles once
	canada = gpd.read_file(kwargs.get('canada'))
	russia = gpd.read_file(kwargs.get('russia'))
	ak_bounds = gpd.read_file(kwargs.get('ak_bounds'))
	
	regions = list(input_dict.keys())
	print(regions)
	#plt.clf()
	fig, axes = plt.subplots(2, 4, gridspec_kw = {'wspace':0, 'hspace':0.1},sharey=True)

	axes = axes.flatten()
	count = 0 
	for ax in axes: 
		try: 
			print('region is ', regions[count])
		except IndexError: 
			pass

		try: 
			dfs = input_dict[regions[count]]

			#gets a df from the pickled data which is like: {'region': [list of dfs with one for each year]}
			first = [df for df in dfs if int(df['file_yr'].iloc[0])==1988][0]
			last = [df for df in dfs if int(df['file_yr'].iloc[0])==2020][0]

			first = clean_gdf(first,'first')
			last = clean_gdf(last,'last')

			combined = first.merge(last, on = 'rgi_label', how='inner')
			combined['diff'] = ((combined['last_area']-combined['first_area'])/combined['first_area'])*100

			combined['diff_no_pct'] = combined['last_area']-combined['first_area']

			print(combined.last_area.min())
			print(combined.last_area.max())
			print(combined['diff'].max())
			print(combined['diff'].min())


			print(combined[['rgi_label','first_area','last_area','diff','diff_no_pct']])
		except IndexError: 
			continue

		#plot it 
		try: 
		
			ax.scatter(combined['first_area'],combined['diff'],facecolors='none', edgecolors=colors[regions[count]],s=8**2)
			# print('max area is: ', combined['first_area'].max())
			# print('max change is: ', combined['diff'].min())
			#ax.set_yscale('log')

			ax.set_title(f"{regions[count]}", x=0.5, y=0.925)
			ax.annotate(f'n = {len(combined)}',xy=(0.8,0.8),xycoords='axes fraction')
		
			#add point labels just for writing (uncomment to add)
			# for i, point in combined.iterrows():
			# 	ax.text(point['first_area'], point['diff'], str(point['rgi_label']))

			#add the inset map 
			inset_ax = inset_axes(ax,
                width="40%", # width = 30% of parent_bbox
                height="40%", # height : 1 inch
                loc='lower left',
                bbox_to_anchor=(0.0,0,1,1), bbox_transform=ax.transAxes)
		
			inset_ax.set_xticks([]) 
			inset_ax.set_yticks([]) 

			#add the base shapefiles

			aoi_bounds = ak_bounds.geometry.total_bounds


			xmin, ymin, xmax, ymax = aoi_bounds

			#set the aoi to the clipped shapefile 
			inset_ax.set_xlim(xmin-100,xmax+100) #changed 5/17/2021 to add one to each val
			inset_ax.set_ylim(ymin-100,ymax+100) 

			ak_bounds.plot(ax=inset_ax,color='lightgray',edgecolor='darkgray')
			russia.plot(ax=inset_ax,color='lightgray',edgecolor='darkgray')
			canada.plot(ax=inset_ax,color='lightgray',edgecolor='darkgray')
			
			#get the specific ecoregion we're going to plot and highlight
			eco_region = gpd.read_file(kwargs.get('ecoregions'))
			region = eco_region.loc[eco_region['Name']==regions[count]]
			region.plot(ax=inset_ax,color=colors.get(regions[count]),edgecolor=colors[regions[count]])
			
			count +=1 

		except Exception as e: 
			print(f'Tried to plot and got the {e} error')
			raise
	fig.delaxes(axes[7]) #hardcoded

	fig.text(0.5, 0.06, '1988 glacier area (km2)', ha='center', va='center',size='large')
	fig.text(0.095, 0.5, 'Percent areal change', ha='center', va='center', rotation='vertical',size='large')

	plt.show()
	plt.close('all')

def main(glacier_change,**kwargs): 




	# #we're going to try using tidewater glacier list from mcnabb and hock 2014, get the ids from that. 6/28/2021 this just needs 
	#to be run one time because it will generate shapefiles that can then be used to clip to climate divisions etc.  
	# tw_list = kwargs.get('tw_list')
	# tw_df = pd.read_csv(tw_list)
	# tw_glaciers = tw_df['Column1']
	# tw_glaciers = [item+' Glacier' for item in tw_glaciers] #the RGI names have 'Glacier' after them, just add this to match 

	# rgi = gpd.read_file(kwargs.get('rgi')).dropna()

	# rgi = rgi.loc[rgi['Name'].isin(list(tw_glaciers))]

	# tw_ids = list(rgi.rgi_id)
	
	# input_files = glob.glob(glacier_change+'*_0.shp')

	# for file in input_files: 
	# 	gdf = gpd.read_file(file)

	# 	gdf = gdf.loc[gdf['rgi_label'].isin(tw_ids)]
	# 	gdf = gdf.set_crs('EPSG:3338')
	# 	output_fn = os.path.join(kwargs.get('output_dir'), os.path.split(file)[1][:-4]+'_mcnabb_updated.shp')
	# 	if not os.path.exists(output_fn): 
	# 		print('writing to: ', output_fn)
	# 		gdf.to_file(output_fn)

	tw_data = clip_each_model_year_for_plotting(glob.glob(glacier_change+'*updated.shp'),**kwargs)

	tw_data = dict((k,v) for k,v in tw_data.items() if v)

	colors = {'Aleutians': '#a6cee3', 'Central Panhandle': '#1f78b4', 'North Panhandle': '#b2df8a', 
	'Northwest Gulf': '#33a02c', 'Southeast Interior': '#fb9a99', 'South Panhandle': '#e31a1c', 
	'Bristol Bay': '#fdbf6f', 'Central Interior': '#ff7f00', 'Cook Inlet': '#cab2d6', 'Northeast Gulf': '#6a3d9a'}

	tw_data.pop('Southeast Interior')
	
	make_tw_plot(tw_data,colors,**kwargs)


	

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
	ecoregions=ecoregions,model='03272021',region='tidewater',output_dir=output_dir,tw_list=mcnabb_tw,rgi=rgi) #here setting the region to all will not do isotherms, 
	#set to above_zero or below_zero for an isotherm run #None is just a placeholder for the glacier change data 