import os 
import sys 
import glob
import matplotlib.pyplot as plt 
import geopandas as gpd 
import pandas as pd 
import seaborn as sns 
import matplotlib.gridspec as gridspec

"""Make time series plots of the national parks, change over time for both overall glacier covered area and supraglacial debris. 
These are based on csvs of areas that were calculated from GlaicerCoverNet outputs clipped to the national park boundaries. 
"""

def calc_start_end(df,col='area'):
	start = float(df.loc[(df['year']==1986) | (df['year']=='1986')][col].iloc[0]) 
	end = float(df.loc[(df['year']==2020) | (df['year']=='2020')][col].iloc[0]) 
	return start,end 

def calc_pct_change(df,col='area'): 
	start = calc_start_end(df)[0]
	df['pct_change'] = ((df[col]-start)/start)*100

	return df

def plot_time_series_change_wo_mapdate(overall_fn,debris_fn,park,output_dir,c,xi='year',yi='area'): 
	
	df1 = calc_pct_change(pd.read_csv(overall_fn).sort_values(xi))
	df2 = calc_pct_change(pd.read_csv(debris_fn).sort_values(xi))
	overall_delta = round(calc_start_end(df1)[1] - calc_start_end(df1)[0],1)
	debris_delta = round(calc_start_end(df2)[1] - calc_start_end(df2)[0],1)
	fig,(ax1,ax2) = plt.subplots(2,figsize=(8,6),sharex=False) #hardcoded
	# print('df1 looks like: ',df1)
	# print('df2 looks like: ',df2)
	sns.lineplot(data=df1, 
				 x=xi,
				 y=yi,
				 linewidth=2, 
				 legend=False,
				 ax=ax1, 
				 color='#00007f'
				)
	sns.lineplot(data=df2, 
				 x=xi,
				 y=yi,
				 linewidth=2, 
				 ls='--',
				 legend=False,
				 ax=ax2, 
				 color='#631919'
				)

	ax1.grid(axis='both',alpha=0.25)
	ax1.set_title(park,fontsize=8)
	ax2.grid(axis='both',alpha=0.25)
	ax1.set_ylabel('Overall area \n(% change from 1986)',fontsize=8)
	ax2.set_ylabel('Supraglacial debris area \n(% change from 1986)',fontsize=8)
	ax1.annotate(f'\u0394 Area 1985-2020 = {overall_delta} $km^2$',xycoords='axes fraction',xy=(0.6,0.9),fontsize=8)
	ax2.annotate(f'\u0394 Area 1985-2020 = {debris_delta} $km^2$',xycoords='axes fraction',xy=(0.05,0.9),fontsize=8)

	fig.savefig(os.path.join(output_dir,park.replace(' ','_').replace('/','_')+'1.jpg'),
		dpi=500, 
		bbox_inches = 'tight',
		pad_inches = 0.1, 
	  	edgecolor=fig.get_edgecolor() 
	    )

	return None 

def plot_time_series_change_w_mapdate(overall_fn,debris_fn,park,output_dir,mapdate_data,c,xi='year',yi='area'): 
	"""Make a figure of subplots with each being the time series of a given region."""

	df1 = calc_pct_change(pd.read_csv(overall_fn).sort_values(xi))
	df2 = calc_pct_change(pd.read_csv(debris_fn).sort_values(xi))
	overall_delta = round(calc_start_end(df1)[1] - calc_start_end(df1)[0],1)
	debris_delta = round(calc_start_end(df2)[1] - calc_start_end(df2)[0],1)

	md_gdf = gpd.read_file(mapdate_data)
	md_gdf['park_area'] = (md_gdf.area)/1000000
	
	#calculate some mapdate year stats
	md_mean = round(md_gdf['start_year'].mean())
	md_std = round(md_gdf['start_year'].std())
	md_min = md_gdf['start_year'].min()
	md_max = md_gdf['start_year'].max()
	md_input = (md_min,round(md_gdf['park_area'].sum()))

	# print('the md data looks like', md_input)

	# fig,(ax1,ax2) = plt.subplots(2,figsize=(8,6),sharex=False) #hardcoded
	fig = plt.figure(tight_layout=True)
	gs = gridspec.GridSpec(2,2,width_ratios=[1,3])
	
	#add two plots on the top line to split the x axis...
	for i in range(2):
		ax = fig.add_subplot(gs[0, i])
		# ax.plot(np.arange(1., 0., -0.1) * 2000., np.arange(1., 0., -0.1))
		
		if i == 0: 
			if md_min != md_max: 
				ax.set_xlim(md_min,md_max)
				ax.spines['right'].set_visible(False)
				ax.set_ylabel('Overall area ($km^2$)',fontsize=8)
				ticks = [md_min,md_max]
				ax.set_xticks(ticks)
				ax.set_xticklabels([str(i) for i in ticks])
				ax.plot([md_min,md_max],
						[round(md_gdf['park_area'].sum()),
						round(md_gdf['park_area'].sum())],
						ls='dotted',
						c='#262626',
						linewidth=3)
				ax.set_ylim(round(calc_start_end(df1)[1]),round(md_gdf['park_area'].sum())+round(round(md_gdf['park_area'].sum())-calc_start_end(df1)[1])/15) #might be too much for some of the parks
				ax.grid(axis='both',alpha=0.25)
				ax.set_xlabel('')
				ax.annotate(f'Mapdate dates \nMean = {md_mean} \n\u03c3 = {md_std}', xycoords='axes fraction',xy=(0.1,0.5),fontsize=8)
			elif md_min == md_max: 
				print('plotting a point not line')
				ticks = [md_min]
				ax.set_xticks(ticks)
				ax.set_xticklabels([str(i) for i in ticks])
				ax.scatter(md_input[0],md_input[1],c='#262626')
				ax.grid(axis='both',alpha=0.25)
				ax.set_xlabel('')
				ax.annotate(f'Mapdate \nyear {md_min}' , xycoords='axes fraction',xy=(0.1,0.5),fontsize=8)
		elif i == 1: 
			ax.set_xlim(1985,2020)
			ticks = [1985,1990,1995,2000,2005,2010,2015,2020]
			ax.set_xticks(ticks)
			ax.set_xticklabels([str(i) for i in ticks])
			ax.set_ylim(round(calc_start_end(df1)[1]),round(md_gdf['park_area'].sum()))
			ax.spines['left'].set_visible(False)
			sns.lineplot(data=df1, 
				 x=xi,
				 y=yi,
				 linewidth=2, 
				 legend=False,
				 ax=ax, 
				 color='#00007f'
				)
			# ax.yaxis.tick_right()
			ax.annotate(f'\u0394 Area 1985-2020 = {overall_delta} $km^2$',xycoords='axes fraction',xy=(0.1,0.9),fontsize=8)
			ax.annotate('GlacierCoverNet', xycoords='axes fraction',xy=(0.5,0.5),fontsize=8)
			ax.grid(axis='both',alpha=0.25)
			ax.yaxis.set_visible(False)
			ax.set_xlabel('')
		# Make the spacing between the two axes a bit smaller
		plt.subplots_adjust(wspace=0.1)
		ax.grid(axis='both',alpha=0.25)
		
		# if i == 0:
		# 	ax.tick_params(axis='x', rotation=55)
	#now add the supraglacial debris on the bottom
	ax = fig.add_subplot(gs[1, :])
	sns.lineplot(data=df2, 
				 x=xi,
				 y=yi,
				 linewidth=2, 
				 ls='--',
				 legend=False,
				 ax=ax, 
				 color='#631919'
				)
	ax.set_xlabel('')
	ax.grid(axis='both',alpha=0.25)
	ax.set_ylabel('Supraglacial debris area ($km^2$)',fontsize=8)
	ax.annotate(f'\u0394 Area 1985-2020 = {debris_delta} $km^2$',xycoords='axes fraction',xy=(0.05,0.9),fontsize=8)
	fig.suptitle(park,fontsize=8)
	
	fig.savefig(os.path.join(output_dir,park.replace(' ','_').replace('/','_')+'1.jpg'),
		dpi=500, 
		bbox_inches = 'tight',
	  pad_inches = 0.1, 
	  edgecolor=fig.get_edgecolor() 
	    )
	

if __name__ == '__main__':
	overall_dir = "/vol/v3/ben_ak/excel_files/areal_change_stats/national_park_time_series_areas/overall_area/"
	debris_dir = "/vol/v3/ben_ak/excel_files/areal_change_stats/national_park_time_series_areas/supraglacial_debris/"
	output_dir = "/vol/v3/ben_ak/figure_building/nps_figs/area_change_by_park_revised/"
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)
	
	#add in the 'mapdate' data
	#clipped park bounds/glaciers 
	mapdate_dir = '/vol/v3/ben_ak/vector_files/glacier_outlines/mapdate_bounds/map_date_clipped_to_nps/'
	mapdate_shps = glob.glob(mapdate_dir+'*.shp')
	mapdate_shps = [f for f in mapdate_shps if os.path.split(f)[1].startswith('updated')]
	#this assumes that the directories are set up the same way and the naming structure is the same
	overall_files = sorted(glob.glob(overall_dir+'*.csv'))
	debris_files = sorted(glob.glob(debris_dir+'*.csv'))

	titles = {'aniakchak':'Aniakchak National Monument',
			  'denali':'Denali National Park/Preserve',
			  'gates':'Gates of the Arctic',
	          'glacier':'Glacier Bay National Park', 
			  'katmai':'Katmai National Park', 
			  'kenai':'Kenai Fjords National Park', 
			  'lake':'Lake Clark National Park', 
			  'wrangall':'Wrangell St. Elias National Park/Preserve'
			  }

	colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#99991e','#a65628','#f781bf']

	color_viewer = dict(zip(list(titles.keys()),colors))
	print(color_viewer)

	count = 0
	for file1,file2 in zip(overall_files,debris_files): 
		#iterate through a dir of csvs with two cols: year and area
		park_name = os.path.split(file1)[1].split('_')
		
		try: 
			park_title = list({k:v for k,v in titles.items() if k in park_name}.values())[0]
			park_key = list({k:v for k,v in titles.items() if k in park_name}.keys())[0]
			print(park_key)
			mapdate_data = [f for f in mapdate_shps if park_key in f][0]
			print(mapdate_data)
		except IndexError: 
			print('Could not find the park you were looking for.')
			pass
		try: 
			pass
			#if you miss a park then it will throw a nameError
			if (park_key == 'glacier') | (park_key == 'wrangall'): 
				plot_time_series_change_wo_mapdate(file1,file2,park_title,output_dir,c=colors[count])
			else: 
				plot_time_series_change_w_mapdate(file1,file2,park_title,output_dir,mapdate_data,c=colors[count])
			# plot_time_series_change_by_region(file1,file2,park_title,output_dir,mapdate_data,c=colors[count])
		except NameError: 
			print('That park is missing')
			
		count +=1

