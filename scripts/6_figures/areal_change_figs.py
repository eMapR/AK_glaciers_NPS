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

def nps_change_map(glacier_data,**kwargs): 
	"""Make a map of glacier change over time in AK national parks."""

	nps = gpd.read_file(kwargs.get('nps_bounds'))
	#format the nps gdf to get rid of park, monument, etc distinctions 
	nps = nps.dissolve(by='UNIT_CODE').reset_index()
	nps = nps.loc[~nps['UNIT_CODE'].isin(kwargs.get('no_glacier'))]

	#add a polygon centroid col to use for arrow pointers 
	nps['centroid'] = nps['geometry'].centroid

	#add a name col to pull park names from
	nps['plot_name'] = nps['UNIT_CODE'].map(kwargs.get('rename'))

	#get the base shapes 
	ak = gpd.read_file(kwargs.get('ak_bounds'))	
	canada = gpd.read_file(kwargs.get('canada'))
	russia = gpd.read_file(kwargs.get('russia'))

	#make a plot 
	fig, ax = plt.subplots(1,1,figsize=(20,20))

	#increase font size 
	font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	matplotlib.rc('font', **font)
	
	#set the bounds so we get rid of the aleutians 
	aoi_bounds = nps.geometry.total_bounds
	ax.set_facecolor("lightblue")

	xmin, ymin, xmax, ymax = aoi_bounds

	#set the aoi to the clipped shapefile 
	ax.set_xlim(xmin-700000,xmax+500000) #changed 5/17/2021 to add one to each val
	ax.set_ylim(ymin-500000,ymax+300000) 

	#add the shapefile plots 
	ak.plot(ax=ax,color='lightgray',edgecolor='darkgray')
	nps.plot(ax=ax,color='#5198c0',edgecolor='darkgray')
	canada.plot(ax=ax,color='lightgray',edgecolor='darkgray')
	russia.plot(ax=ax,color='lightgray',edgecolor='darkgray')

	ax.set_title("Overall glacier covered area change in Alaska's National Parks 1990-2020")
	#annotate them with change stats - note that the Gates of the Arctic one will not be updated yet (6/2/2021) because this just includes shapefiles of the s zone 

	#make a dict of locations for the annotations- there must be a better way to do this 
	locations = {'ANIA':(0,-50),'DENA':(-75,10),'GAAR':(0,20),'GLBA':(20,-75),'KATM':(-100,-10),'KEFJ':(20,-40),'KLGO':(20,10),'LACL':(-80,0),'WRST':(30,50)}
	count = 0
	for code in nps['UNIT_CODE'].unique(): 
		nps = nps.sort_values('UNIT_CODE')
		park = nps.loc[nps['UNIT_CODE']==code]
		
		start = [file for file in glacier_data if '1990' in file][0]
		start = gpd.read_file(start)
		start = add_area_col(clip_to_aoi(start,park,'1990',kwargs.get('output_dir')))
		start = start['area'].sum()

		end = [file for file in glacier_data if '2020' in file][0]
		end = gpd.read_file(end)
		end = add_area_col(clip_to_aoi(end,park,'2020',kwargs.get('output_dir')))
		end = end['area'].sum()

		#add some annotations 
		ax.annotate(f'{park["plot_name"].iloc[0]}\n{round(((end-start)/start)*100)} %',
	            xy=(nps['centroid'].iloc[count].x,nps['centroid'].iloc[count].y), xycoords='data',
	            xytext=locations.get(code), textcoords='offset points',
	            bbox=dict(boxstyle='round', fc='#a8c0b6'),
	            arrowprops=dict(arrowstyle="->"))

		count += 1 

	# plt.show()
	# plt.close('all')
	plt.savefig(os.path.join(kwargs.get('fig_dir'),'nps_glacier_change_draft3.png'),dpi=350)

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

			# else: 
			# 	print('The plot dict already contains panhandle and looks like: ')
			# 	print(plot_dict.keys())
			# else: 
			# 	pass 

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


def plot_time_series_change_by_region(input_dict,second_data,colors,**kwargs): #currently (6/6/2021) the input dict is like: {region:df with all the years clipped to that region}
	"""Make a figure of subplots with each being the time series of a given region."""

	fig,ax = plt.subplots(3,4,figsize=(8,6),sharex=True,sharey=True,gridspec_kw={'wspace':0,'hspace':0.25}) #hardcoded

	#read in some shapefiles once
	canada = gpd.read_file(kwargs.get('canada'))
	russia = gpd.read_file(kwargs.get('russia'))
	ak_bounds = gpd.read_file(kwargs.get('ak_bounds'))
	
	ax = ax.flatten()

	count = 0 
	combined_area_dict = {}
	debris_area_dict = {}

	for k,v in input_dict.items(): #v is now a list of dfs (060720201)
		
		print('k is: ',k)
		#get the primary plotting product (overall glacier area)
		try: 
			df = create_df_from_list(v,'file_yr')
			#added 8/27/2021 to make sure that data is correctly sorted chronologically and 1986 was removed 
			df = df.sort_values('file_yr')
			#classifications for 1986 and 1988 in the northern zone don't look great so start the 
			if not (k == 'North Slope') | (k == 'Northeast Interior'): 
				df = df.loc[df['file_yr'] != 1986]
			else: 
				df = df.loc[(df['file_yr'] != 1986) & (df['file_yr'] != 1988)]
			print(df.head())
		except KeyError: 
			print(f'{k} does not exist in the primary data')
			break

		#get the secondary product (debris cover)
		try: 
			df2 = create_df_from_list(second_data[k],'file_yr')  
			#added 8/27/2021 to make sure that data is correctly sorted chronologically and 1986 was removed 
			df2 = df2.sort_values('file_yr')
			df2 = df2.loc[df2['file_yr'] != 1986]
			print('debris')
			print(df2.head())
		except KeyError as e: 
			print(f'{k} does not exist in the debris data. Destrying debris df.')
			del df2 
			
		#start the actual plotting
		
		scaler = MinMaxScaler()

		#get all the individual glaciers for one year and scale the data to 0-1 so they have the same y axes 
		#changed all the 'area' cols to GN area for 8/12/2021 model outputs
		try: 
			gdf = df.groupby('file_yr')['area'].sum().reset_index()
			gdf['sc_area'] = scaler.fit_transform(gdf['area'].values.reshape(-1,1))
		except UnboundLocalError: 
			raise 

		try: 
			gdf2 = df2.groupby('file_yr')['area'].sum().reset_index()
			gdf2['scnd_area'] = scaler.fit_transform(gdf2['area'].values.reshape(-1,1))
		except UnboundLocalError: 
			print('There is no data for the second dataset, skipping scaling...')
			del gdf2

		#experimental
		#gdf = gdf.merge(gdf2,on='file_yr',how='inner')
		#ax[count].set_xticks(range(1986,2022,2))
		try: 
			gdf.plot(x='file_yr',y='sc_area',ax=ax[count],linewidth=2,color='black',legend=False) #changed to remove colors 10/27/2021
		except ValueError: 
			pass
		try: 
			#don't add debris cover for the northern zone because its not calculated in the newest data 
			if not (k == 'North Slope') | (k == 'Northeast Interior'): 
				gdf2.plot(x='file_yr',y='scnd_area',ax=ax[count],linewidth=2,linestyle='--',color='black',legend=False) #changed to remove colors 10/27/2021
		except Exception as e: #not 100% sure which error this will raise 
			print(f'Tried to plot the second df, got the error: {e}. Continuing as a result...')

		#currently running with south, central and north panhandle as indidividual regions (6/20/2021)
		#if not 'Panhandle' in k: 
		ax[count].set_title(k,fontsize=10)

		ax[count].set_xlabel(' ')
		ax[count].set_ylabel(' ')
		# elif 'Panhandle' in k: 
		# 	ax[count].set_title('Southeast panhandle')
		
		#set the axis labels 
		# if count > 5: 
		# 	ax[count].set_xlabel('CNN composite year')
		
		# if (count == 0) | (count == 3) | (count==6): 
		# 	ax[count].set_ylabel('Scaled area')

		#write the areas out to a table for the appendix 
		# area_1990 = df.loc[df['file_yr']==1988]
		# area_2020 = df.loc[df['file_yr']==2020]
		
		#create a list to hold vals like: [1990 og area, 2020 og area, 1990 debris area, 2020 debris area]
		try: 
			combined_area_list = []
			debris_area_list = []
			for x in range(1988,2022,2): 
				combined_area_list.append(calc_area(get_df_subset(df,x)))
				debris_area_list.append(calc_area(get_df_subset(df2,x)))
			# area_list = [calc_area(get_df_subset(df,1988)),calc_area(get_df_subset(df,2020)),
			# calc_area(get_df_subset(df2,1988)),calc_area(get_df_subset(df2,2020))]
			
			#add the regional data to a dict to output as df (csv)
			combined_area_dict.update({k:combined_area_list})
			debris_area_dict.update({k:debris_area_list})
		
		except UnboundLocalError: 
			print('Tried to add data to csv dict but failed because a df is missing.')

		#add the total glacier area as text 

		# ax[count].annotate(f'1990 overall area:\n{round(area_1990["area"].sum(),2)}',xy=(0.5,0.8),xycoords='axes fraction',size='large')
		# ax[count].annotate(f'2020 overall area:\n{round(area_2020["area"].sum(),2)}',xy=(0.5,0.65),xycoords='axes fraction',size='large')

		#add the plot letter identifier 
		ax[count].annotate(f'{chr(97+count)}',xy=(0.8,0.9),xycoords='axes fraction',size='large',weight='bold')
		#remove erroneous tick marks 
		if count not in [8,9,10,11]: 
			ax[count].tick_params(axis='x',bottom=False)
		if count not in [0,4,8]: 
			ax[count].tick_params(axis='y',left=False,right=False)

		ax[count].grid(axis='y',alpha=0.25)

		#add an inset to show where in AK we're looking 

		# These are in unitless percentages of the figure size. (0,0 is bottom left)
		inset_ax = inset_axes(ax[count],
                width="40%", # width = 30% of parent_bbox
                height="40%", # height : 1 inch
                loc='lower left',
                bbox_to_anchor=(0,0,1,1), bbox_transform=ax[count].transAxes)
		
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
		region = eco_region.loc[eco_region['Name']==k]
		# if not 'Panhandle' in region['Name'].iloc[0]: #check if we're dealing with the panhandle geometries
		region.plot(ax=inset_ax,color='black',edgecolor='black') #colors.get(k)
		# elif 'Panhandle' in region['Name'].iloc[0]:
			#eco_region.loc[eco_region['Name'].str.contains('Panhandle')].plot(ax=inset_ax,color=colors.get(k),edgecolor='black')

		count += 1 
	
	fig.text(0.5, 0.02, 'GlacierNet\ncomposite year', ha='center', va='center',size='medium')
	fig.text(0.05, 0.5, 'Scaled area', ha='center', va='center', rotation='vertical', size='medium')

	#add a legend 
	custom_lines = [Line2D([0], [0], color='black', lw=2),
					Line2D([0], [0], color='black', lw=2, linestyle='--')]            
	fig.legend(custom_lines, ['Overall glacier area', 'Supraglacial debris'],loc=(0.01,0.01))
	
	# fig.delaxes(ax[10]) #hardcoded
	# fig.delaxes(ax[11]) #hardcoded

	plt.savefig(os.path.join(kwargs.get('fig_dir'),f'areal_change_by_region_{kwargs.get("region")}_watersheds_final3.png'),
		dpi=500, 
		bbox_inches = 'tight',
	    pad_inches = 0.1
	    )
	
	#export area stats for a table 
	combined_area_fn = os.path.join(kwargs.get('stats_dir'), 'overall_glacier_area_all_years_full_draft1.csv')
	debris_cover_area_fn = os.path.join(kwargs.get('stats_dir'), 'debris_cover_glacier_area_all_years_full_draft1.csv')
	export_basic_area_stats(pd.DataFrame.from_dict(combined_area_dict),combined_area_fn)
	export_basic_area_stats(pd.DataFrame.from_dict(debris_area_dict),debris_cover_area_fn)
	plt.show()
	plt.close('all')

def main(glacier_change,**kwargs):

	#make a plot of glacier change just between 1990 and 2020 for the national parks 
	#nps_change_map(glob.glob(glacier_change+'*.shp'),**kwargs) 

	#make a plot of glacier change for the eco-regions of AK 

	#first do the combined data 
	base_data = clip_each_model_year_for_plotting(glob.glob(glacier_change+'*buffer.shp'),region='all',**kwargs)
	print(base_data)
	base_data = dict((k,v) for k,v in base_data.items() if v)

	#then do debris cover
	debris_data=clip_each_model_year_for_plotting(glob.glob(kwargs.get('debris_cover')+'*buffer.shp'),region='debris',**kwargs)
	debris_data=dict((k,v) for k,v in debris_data.items() if v)

	# colors = dict(zip(base_data.keys(),['#A9E5BB','#FCF6B1','#ECB009','#9FE2BF','#9D6B06','#F72C25','#92252A','#2D1E2F','#000000','#4c004c','#00007f']))
	# print(colors)
	print('base data ')
	print(base_data.keys())
	print('debris data ')
	print(debris_data.keys())
	colors = {'Aleutians': '#a6cee3', 'Central Panhandle': '#1f78b4', 'North Panhandle': '#b2df8a', 
	'Northwest Gulf': '#33a02c', 'Southeast Interior': '#fb9a99', 'South Panhandle': '#e31a1c', 
	'Bristol Bay': '#fdbf6f', 'Central Interior': '#ff7f00', 'Cook Inlet': '#cab2d6', 
	'Northeast Gulf': '#6a3d9a', 'Northeast Interior':'#521515', 'North Slope':'black'}
	plot_time_series_change_by_region(base_data,debris_data,colors,region='all',**kwargs) #might need to adjust region 8/26/2021

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
		debris_cover = variables['debris_cover']

		no_glacier = ['ALAG','BELA','CAKR','KOVA','NOAT','SITK','YUCH']
		rename = {'ANIA':'Aniakchak','DENA':'Denali','GAAR':'Gates of the Arctic','GLBA':'Glacier Bay','KATM':'Katmai','KEFJ':'Kenai Fjords','KLGO':'Klondike\nGold Rush','LACL':'Lake Clark','WRST':'Wrangell-St. Elias'}

	main(glacier_change,
		nps_bounds=nps_bounds,
		pickles=pickles,
		ak_bounds=ak_bounds,
		no_glacier=no_glacier,
		rename=rename,
		canada=canada,
		russia=russia,
		fig_dir=fig_dir,
		ecoregions=ecoregions,
		model=model,
		stats_dir=output_dir,
		debris_cover=debris_cover) #here setting the region to all will not do isotherms, 
	#set to above_zero or below_zero for an isotherm run #None is just a placeholder for the glacier change data 

###############################################################
#working and slightly modified 8/26/2021
# def clip_each_model_year_for_plotting(change_data,**kwargs): 
# 	"""Clip time series to regions and save as a dictionary like {region:gdf} as pickle."""
# 	try: 
# 		eco = gpd.read_file(kwargs.get('ecoregions')).set_crs('EPSG:3338')

# 	except Exception as e: 
# 		print('Tried to set the crs for the ecoregions')
# 		eco = gpd.read_file(kwargs.get('ecoregions')).to_crs('EPSG:3338')
	
# 	#this section should just have to be run once for a model
	
# 	if kwargs.get('region').lower()=='debris': 
# 		print('reading debris data')
# 		output_fn = os.path.join(kwargs.get('pickles'),f"{kwargs.get('model')}_dictionary_like_ecoregion_list_of_shps_clipped_to_debris_watersheds.p")#f'{model}_dictionary_like_ecoregion_list_of_shps_clipped_to_{kwargs.get("region")}_watersheds_amended.p')
	
# 	elif kwargs.get('region').lower()=='all': 
# 		print('reading primary data')
# 		#this filename is for running all the climate divisions in the full area
# 		output_fn=os.path.join(kwargs.get('pickles'), f"{kwargs.get('model')}_dictionary_like_ecoregion_list_of_shps_clipped_to_combined_watersheds.p") #this is hardcoded for the legacy version of the data 
		
# 	count = 0 
# 	if not os.path.exists(output_fn): 
# 		plot_dict = {}
# 		#iterate through the ecoregions
# 		for region in eco['Name'].unique():
# 			print('The current region is: ',region)
# 			#if (region == 'Northeast Gulf') | (region == 'Southeast Interior'): 
# 			#define a list to hold a time series 
# 			region_list = []

# 			#changed this section 6/18/2021 so that each of the SE panhandle sections run separately 
# 			# if not 'Panhandle' in region: 
# 			gdf = eco.loc[eco['Name']==region]

# 			# #there are three sections of the SE pandhandle, combine them for ease
# 			# elif 'Panhandle' in region: 
# 			# 	gdf = eco.loc[eco['Name'].str.contains('Panhandle')]

# 			# #check if a panhandle group is already in the dictionary, if it is pass this one 
# 			# proceed = {k:v for k,v in plot_dict.items() if 'Panhandle' in k}
# 			# print(proceed)
			
# 			# if (len(proceed)==0): 

# 			for file in change_data: #iterate through the files in the time series- probably not the fastest way to do this but it only has to be done once per model 
# 				print(f'The current file being processed is: {file}')


# 				year = re.findall('(\d{4})', os.path.split(file)[1][8:])[0] #gets a list with the start and end of the water year, take the first one. expects files to have the 8 digit model date first 
				
# 				print(year)
				
# 				try: 
# 					#do the actual clipping and catch instances where there is an empty df (no glaciers)
# 					clipped = add_area_col(clip_to_aoi(gpd.read_file(file).set_crs('EPSG:3338'),gdf,year))
					
# 					#add a year col for when it goes into the output and gets concatenated 
# 					clipped['file_yr'] = year
# 					region_list.append(clipped)

# 				except Exception as e: 
# 					print(f'Tried to add a col but could not because of a {e} error.')
			
# 					break #if the first one is nonetype its likely because there are no glaciers in that group, if that's the case skip the rest
			
# 			try: 
# 				#add the clipped gdf to the output dict with the regional name as key 
# 				plot_dict.update({region:region_list})

# 			except Exception as e: 
# 				print('Tried to update the clip dict and had a noneType')
# 				print(f'The error was: {e}')

# 			#increment the count to check for inclusion of panhandle group 
# 			count +=1

# 			# else: 
# 			# 	print('The plot dict already contains panhandle and looks like: ')
# 			# 	print(plot_dict.keys())
# 			# else: 
# 			# 	pass 

# 		#pickle it 
# 		print('Pickling to disk...')
# 		pickle.dump(plot_dict, open(output_fn,'wb'))
# 		print('Successfully pickled')
# 		return plot_dict



############################################################################
##this is working fine (6/20/2021) to make a figure with one data type, rewriting to have main data and debris cover in the same plot 

# for k,v in input_dict.items(): #v is now a list of dfs (060720201)

# 		# for i in v: 
# 		# 	print(i['area'].mean())
# 		print('k is: ',k)
# 		# print(v)
# 		try: 
# 			df = pd.concat(v)
# 			#print(df.head())
# 			if not df.empty: 
				
# 				#start the actual plotting
				
# 				#x_array = np.array([2,3,5,6,7,4,8,7,6])
# 				# normalized_arr = preprocessing.normalize([x_array])
# 				# print(normalized_arr)

# 				scaler = MinMaxScaler()

# 				gdf = df.groupby('file_yr')['area'].sum().reset_index()
# 				gdf['sc_area'] = scaler.fit_transform(gdf['area'].values.reshape(-1,1))#preprocessing.normalize([gdf.area])
# 				#print(gdf['sc_area'])

# 				# print(v.area)
# 				# print(v['area'].mean())
# 				gdf.plot(x='file_yr',y='sc_area',ax=ax[count],linewidth=2.5,color=colors.get(k),legend=False)
				
# 				if not 'Panhandle' in k: 
# 					ax[count].set_title(k)
# 				elif 'Panhandle' in k: 
# 					ax[count].set_title('Southeast panhandle')
				
# 				if count > 3: 
# 					ax[count].set_xlabel('CNN composite year')
				
# 				if (count == 0) | (count == 4): 
# 					ax[count].set_ylabel('Scaled area')

# 				area_1990 = df.loc[df['file_yr']=='1990']
# 				area_2020 = df.loc[df['file_yr']=='2020']

				
# 				#add the total glacier area as text 

# 				ax[count].annotate(f'1990 area (km2):\n{round(area_1990["area"].sum(),2)}',xy=(0.5,0.8),xycoords='axes fraction',size='large')
# 				ax[count].annotate(f'2020 area (km2):\n{round(area_2020["area"].sum(),2)}',xy=(0.5,0.65),xycoords='axes fraction',size='large')

# 				#add the plot letter identifier 
# 				ax[count].annotate(f'{chr(97+count)}',xy=(0.5,0.9),xycoords='axes fraction',size='large',weight='bold')

# 				#add a line of best fit 
# 				# gdf['file_yr'] = gdf['file_yr'].astype('str')

# 				# m, b = np.polyfit(gdf['file_yr'], gdf['area'], 1)
# 				# ax[count].plot(gdf['file_yr'], m*gdf['area'] + b,linestyle='--',color='black',linewidth=2.5)

# 				#add an inset to show where in AK we're looking 

# 				# These are in unitless percentages of the figure size. (0,0 is bottom left)
# 				inset_ax = inset_axes(ax[count],
# 	                    width="40%", # width = 30% of parent_bbox
# 	                    height="40%", # height : 1 inch
# 	                    loc='lower left',
# 	                    bbox_to_anchor=(0,0,1,1), bbox_transform=ax[count].transAxes)

# 				#remove the ticks and labels from the inset 
# 				#for axi in [axins, axins2, axins3, axins4]:
# 				#inset_ax.tick_params(labelleft=False, labelbottom=False)
# 				inset_ax.set_xticks([]) 
# 				inset_ax.set_yticks([]) 

# 				#add the base shapefiles

# 				aoi_bounds = ak_bounds.geometry.total_bounds
		

# 				xmin, ymin, xmax, ymax = aoi_bounds

# 				#set the aoi to the clipped shapefile 
# 				inset_ax.set_xlim(xmin-100,xmax+100) #changed 5/17/2021 to add one to each val
# 				inset_ax.set_ylim(ymin-100,ymax+100) 

# 				ak_bounds.plot(ax=inset_ax,color='lightgray',edgecolor='darkgray')
# 				russia.plot(ax=inset_ax,color='lightgray',edgecolor='darkgray')
# 				canada.plot(ax=inset_ax,color='lightgray',edgecolor='darkgray')
				
# 				#get the specific ecoregion we're going to plot and highlight
# 				eco_region = gpd.read_file(kwargs.get('ecoregions'))
# 				region = eco_region.loc[eco_region['Name']==k]
# 				if not 'Panhandle' in region['Name'].iloc[0]: #check if we're dealing with the panhandle geometries
# 					region.plot(ax=inset_ax,color=colors.get(k),edgecolor='black')
# 				elif 'Panhandle' in region['Name'].iloc[0]:
# 					eco_region.loc[eco_region['Name'].str.contains('Panhandle')].plot(ax=inset_ax,color=colors.get(k),edgecolor='black')

# 				else: 
# 					print('Something went wrong trying to plot the ecoregions')
# 				count += 1 
			
# 			else: 
# 				print(f'The region {k} has an empty dataframe')
# 		except ValueError: 
# 			print('That gdf is empty')
# 			raise
# 	fig.delaxes(ax[7]) #hardcoded
# 	#plt.savefig(os.path.join(kwargs.get('fig_dir'),f'areal_change_by_region_{kwargs.get("region")}_watersheds_draft1.png'),dpi=350)
# 	#plt.tight_layout()
# 	plt.show()
# 	plt.close('all')