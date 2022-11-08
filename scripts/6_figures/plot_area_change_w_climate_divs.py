import os 
import re
import sys 
import json 
import glob 
import pickle
import pandas as pd 
import matplotlib
import pymannkendall as mk
from sklearn import preprocessing
import geopandas as gpd 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from prep_data_for_area_change import clip_each_model_year_for_plotting
from scipy import stats 
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

"""Plot changes in area of glaciers in alaska climate divisions. Note that as of 11/30/2021, this process is 
   amended to do aggregations of climate divs, not all of the climate divs alone. 
   Aggregations should be as follows: 
   NW Gulf of Ak: 
   	Cook Inlet
   	NW Gulf
   	Bristol Bay
   	Aleutians

   Central/SE Interior
   NE Gulf: 
   	-NE Gulf 
   	-S. Panhandle
   	-N. Panhandle
   	-C. Panhandle
   Brooks Range: 
   	-North Slope
   	-NE interior 

   Inputs: 
   - pickled object that is the output of prep_data_for_area_change.py- note that this will be for each climate division
     This object looks like: {region:df with all the years clipped to that region}
   - note that there are two pickled objects, one is overall glacier change, the other is supraglacial debris 
   Output: 
   - Figure plotting change
"""

def run_theil_sen_slope(x,y): 
	"""Implementation of Theil-Sen slope estimator. Takes two array-like objects."""
	res = stats.theilslopes(y,x,0.95)
	print(res)

def run_mk_test(data): 
	return mk.original_test(data)

def format_mk_data(df,col='group_id'): 
	"""Take a df from the plot func and reformat for the mk test. This just requires an array of data."""
	for i in df[col].unique():
		print(i)
		df_sub = df.loc[df[col]==i]
		print(run_mk_test(df_sub.area)) 

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

def lists_to_dfs(ls_of_ls,group_ids):
	"""Take the output lists from prep_dict_data and make into dict."""
	output_dict = {}	
	output_ls = []
	for ls,idx in zip(ls_of_ls,group_ids): 
		#make one df from the climate divs in that group 
		df = pd.concat(ls)
		#add up the areas for the climate divs 
		df = df.groupby(['file_yr'])['area'].sum().reset_index()
		#add a col that is the percent change from the first year 
		df['pct_change'] = df['area'].apply(lambda x: ((x-(df.area.iloc[0]))/df.area.iloc[0])*100)
		#add an id col 
		df['group_id'] = idx 
		output_ls.append(df)
		#output_dict.update({idx:df})
	return pd.concat(output_ls)

def prep_dict_data(input_dict,data_type='overall'):
	"""Take the pickled object and group it by regions.""" 
	
	#make some lists for the aggregated groups 
	nw_gulf = []
	interior = []
	ne_gulf = []
	brooks_r = []

	for k,v in input_dict.items(): #v is a list of dfs 
		print('k is: ',k)
		#get the primary plotting product (overall glacier area)
		try: 
			df = create_df_from_list(v,'file_yr')
			#added 8/27/2021 to make sure that data is correctly sorted chronologically and 1986 was removed 
			df = df.sort_values('file_yr')
			#classifications for 1986 and 1988 in the northern zone don't look great so start the 
			# if not (k == 'North Slope') | (k == 'Northeast Interior'): 
			# 	df = df.loc[df['file_yr'] != 1986]
			# else: 
			if (k == 'North Slope') | (k == 'Northeast Interior'): 
				#print('processing brooks range')
				df = df.loc[(df['file_yr'] != 1986) & (df['file_yr'] != 1988)]
				#print('the brooks range looks like: ')
				#print(df)

		except KeyError: 
			if not data_type == 'debris': 
				print(f'{k} does not exist in the primary data')
				break
			else: 
				print(f'{k} does not exist in the debris data. Destrying debris df.')
				del df2 
		#put the data in the right group 
		if k in ['Cook Inlet', 'Northwest Gulf', 'Bristol Bay', 'Aleutians']: 
			nw_gulf.append(df)
		elif k in ['Central Interior','Southeast Interior']: 
			interior.append(df)
		elif k in ['Northeast Gulf', 'South Panhandle', 'Central Panhandle', 'North Panhandle']: 
			ne_gulf.append(df)
		else: 
			try: 
				brooks_r.append(df)
			except Exception as e: 
				print('Tried to append to the brooks range but broke it')

	return lists_to_dfs([nw_gulf,interior,ne_gulf,brooks_r],['Northwest Gulf','Interior','Northeast Gulf','Brooks Range'])

def plot_time_series_change_by_region(input_dict,second_data,**kwargs): 
	"""Make a figure of subplots with each being the time series of a given region."""

	fig,ax = plt.subplots(2,figsize=(6,4),sharex=True,sharey='row',gridspec_kw={'wspace':0,'hspace':0.25}) #hardcoded

	# #read in some shapefiles once for maps 
	# canada = gpd.read_file(kwargs.get('canada'))
	# russia = gpd.read_file(kwargs.get('russia'))
	# ak_bounds = gpd.read_file(kwargs.get('ak_bounds'))
	
	# ax = ax.flatten()

	count = 0 
	combined_area_dict = {}
	debris_area_dict = {}
	#these now look like a dict: {meta_group id:df that has a pct change col from start year}
	overall_area = prep_dict_data(input_dict).reset_index()
	debris_area = prep_dict_data(second_data,data_type='debris').reset_index()

	#create a df of just brooks range data to plot below
	brooks_overall = overall_area.loc[overall_area.group_id == 'Brooks Range']
	#comment or uncomment to include the brooks range, maybe plot it on another axis? 
	overall_area = overall_area.loc[overall_area.group_id != 'Brooks Range']
	debris_area = debris_area.loc[debris_area.group_id != 'Brooks Range']
	
	print('overall area')
	print(overall_area)
	print(overall_area.columns)
	print('debris area')
	print(debris_area)
	print(debris_area.columns)
	#uncomment the last one if you want to include the brooks range
	colors = ['#060606','#444444','#8e8e8e','#000000']
	styles = ['-','--',':']

	sns.lineplot(data=overall_area, 
				 x='file_yr',
				 y='pct_change', 
				 hue='group_id', 
				 style='group_id',
				 palette=colors[:-1],
				 #markers=styles,
				 linewidth=2, 
				 legend=False,
				 ax=ax[0]
				)

	sns.lineplot(data=debris_area, 
				 x='file_yr',
				 y='pct_change', 
				 hue='group_id', 
				 style='group_id',
				 palette=colors[:-1], 
				 #markers=styles,
				 linewidth=2, 
				 legend=True,
				 ax=ax[1]
				)

	#add an inset plot
	print('the brooks range looks like: ')
	print(brooks_overall)
	ax3 = plt.axes([0,0,1,1])
	# # Manually set the position and relative size of the inset axes within ax2
	ip = InsetPosition(ax[0], [0.09,0.15,0.3,0.3])
	ax3.set_axes_locator(ip)
	sns.lineplot(x='file_yr', 
				 y='pct_change',  
				 data=brooks_overall,
				 color='#000000',
				 linewidth=1.5,
				 ax=ax3, 
				 legend=False
				 )
	ax3.set_ylabel('')
	ax3.set_xlabel('')
	#ax3.grid(axis='both',alpha=0.25)
	ax3.set_title('Brooks Range',fontsize=8)
	ax3.tick_params(axis='both', which='both', labelsize=8)

	ax[0].set_ylabel('')
	ax[1].set_ylabel('')
	ax[1].set_xlabel('')
	ax[0].grid(axis='both',alpha=0.25)
	ax[1].grid(axis='both',alpha=0.25)
	ax[0].set_title('Overall glacier area',fontsize=8)
	ax[1].set_title('Supraglacial debris area',fontsize=8)
	ax[0].tick_params(axis='both', which='both', labelsize=8)
	ax[1].tick_params(axis='both', which='both', labelsize=8)
	fig.text(0.5, 0.02, 'GlacierCoverNet\ncomposite year', ha='center', va='center',size='small')
	fig.text(0.05, 0.5, 'Percent change from 1986', ha='center', va='center', rotation='vertical', size='small')
	ax[1].legend(loc='upper left',title='',prop={'size':8})
	#plt.figure(linewidth=0.5, edgecolor="#000000")
	#fig.savefig("image_filename.png", edgecolor=fig.get_edgecolor())

	# plt.show()
	# plt.close('all')
	
	fig.savefig(os.path.join(kwargs.get('fig_dir'),f'aggregated_area_w_brooks_range_remote_sensing_submission.jpg'),
		dpi=1000, 
		bbox_inches = 'tight',
	  pad_inches = 0.1, 
	  edgecolor=fig.get_edgecolor() 
	  #edgecolor='black'
	    )
	
	
	# export area stats for a table 
	# combined_area_fn = os.path.join(kwargs.get('stats_dir'), 'overall_glacier_area_all_years_aggregated_draft1.csv')
	# debris_cover_area_fn = os.path.join(kwargs.get('stats_dir'), 'debris_cover_glacier_area_all_years_aggregated_draft1.csv')
	# export_basic_area_stats(overall_area,combined_area_fn)
	# export_basic_area_stats(debris_area,debris_cover_area_fn)
	
	return overall_area, debris_area

def main(glacier_change,**kwargs):

	#make a plot of glacier change just between 1990 and 2020 for the national parks 
	#nps_change_map(glob.glob(glacier_change+'*.shp'),**kwargs) 

	#make a plot of glacier change for the eco-regions of AK 

	#first do the combined data 
	base_data = clip_each_model_year_for_plotting(glob.glob(glacier_change+'*buffer.shp'),region='all',**kwargs)
	base_data = dict((k,v) for k,v in base_data.items() if v)

	#then do debris cover
	debris_data=clip_each_model_year_for_plotting(glob.glob(kwargs.get('debris_cover')+'*buffer.shp'),region='debris',**kwargs)
	debris_data=dict((k,v) for k,v in debris_data.items() if v)

	#make the plot
	overall, debris = plot_time_series_change_by_region(base_data,debris_data,region='all',**kwargs) #might need to adjust region 8/26/2021
	#run mann kendell tests
	#print('doing overall')
	#format_mk_data(overall)
	#print('doing debris')
	#format_mk_data(debris)
	# print('doing overall')
	# print('overall looks like: ')
	# print(overall)
	# for i in ['Northwest Gulf','Interior','Northeast Gulf', 'Brooks Range']: 
	# 	print('i is: ', i)
	# 	df = overall.loc[overall['group_id']==i]
	# 	run_theil_sen_slope(df['file_yr'],df['area'])

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

	# colors = dict(zip(base_data.keys(),['#A9E5BB','#FCF6B1','#ECB009','#9FE2BF','#9D6B06','#F72C25','#92252A','#2D1E2F','#000000','#4c004c','#00007f']))
	# print(colors)
	# print('base data ')
	# print(base_data.keys())
	# print('debris data ')
	# print(debris_data.keys())
	# colors = {'Aleutians': '#a6cee3', 'Central Panhandle': '#1f78b4', 'North Panhandle': '#b2df8a', 
	# 'Northwest Gulf': '#33a02c', 'Southeast Interior': '#fb9a99', 'South Panhandle': '#e31a1c', 
	# 'Bristol Bay': '#fdbf6f', 'Central Interior': '#ff7f00', 'Cook Inlet': '#cab2d6', 
	# 'Northeast Gulf': '#6a3d9a', 'Northeast Interior':'#521515', 'North Slope':'black'}