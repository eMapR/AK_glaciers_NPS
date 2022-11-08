import os 
import sys
import pandas as pd 
import geopandas as gpd 
import glob 
import json 
import re 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle
import numpy as np 
from scipy.stats import pearsonr
import matplotlib.patches as patches

#uses the validation.txt params file

def amend_cols(gdf,year,modifier,min_size=0.01): 
	#recalculate the extra area col in case its changed 
	 
	gdf[f'{modifier}_area'] = gdf['geometry'].area / 10**6 #this assumes that the field that is being created is in meters and you want to change to km2
	# except KeyError: 
	# 	print('It seems like your input is not a GDF, double check and try again.')
	
	gdf = gdf.loc[gdf[f'{modifier}_area']>=min_size]
	
	#add a year col for id
	gdf['file_yr'] = year

	return gdf 
	

def compare_areas_by_glacier(class_shp,ref_shp,output_dir,year): 
	"""Pull out data from classified outputs and RGI to compare for validation."""

	#for file in class_shps: 
		
	pred = gpd.read_file(class_shp) #a shapefile for one year for a full region 

	#add a unique area col 
	pred = amend_cols(pred,year,'pred')

	ref = amend_cols(ref_shp,year,'ref')

	#get subsets
	# pred = pred[['rgi_label','persistenc','pred_area','file_yr']]
	# pred.rename(columns={'rgi_label':'rgi_id'},inplace=True)
	# ref = ref[['RGIId','Name','year','rgi_id','ref_area','file_yr']]#.sort_values('rgi_id')
	#updated 8/25/2021 for the 8/12/2021 model run and additional processing 
	pred = pred[['RGIId','pred_area','file_yr']]
	#pred.rename(columns={'rgi_label':'rgi_id'},inplace=True)
	ref = ref[['RGIId','Name','start_yr','rgi_id','ref_area','file_yr']]#.sort_values('rgi_id')
	# print(pred.rgi_id)
	# print(ref.rgi_id)

	#merge the data on the RGI id col 
	combined = ref.merge(pred,on='RGIId',how='inner')

	# print('area maxes are: ')
	# print(combined['ref_area'].max())
	# print(combined['pred_area'].max())
	#print(combined)
	#changed 6/3/2021 from concat to merge
	#output = pd.concat([gdf,ref_shp],axis=1) #concat col wise 

	# output_fn = os.path.join(output_dir,f'model_outputs_rgi_test_partition_rgi_yrs_combined_model_year_{year}.csv')

	# if not os.path.exists(output_fn): 
	# 	output.to_csv(output_fn)

	# else: 
	# 	print(f'The file {output_fn} already exists')

	return combined#pred,ref

def plot_glacier_wise_comparison(input_df,output_dir): 
	"""Make a plot comparing CNN output to RGI by glacier ID for specific years."""


	fig,ax = plt.subplots(2,3,
						  figsize=(8,5),
						  gridspec_kw={'wspace':0.25,'hspace':0.25})

	#plt.rcParams.update({'font.size': 16})

	ax = ax.flatten()
	fs = 9
	#select subsets of the data by glacier size 
	first = input_df.loc[(input_df['ref_area']>0)&(input_df['ref_area']<=1)]
	second = input_df.loc[(input_df['ref_area']>1)&(input_df['ref_area']<=5)]
	third = input_df.loc[(input_df['ref_area']>5)&(input_df['ref_area']<=25)]
	fourth = input_df.loc[(input_df['ref_area']>25)&(input_df['ref_area']<=50)]
	fifth = input_df.loc[(input_df['ref_area']>50)&(input_df['ref_area']<=100)]
	sixth = input_df.loc[(input_df['ref_area']>100)&(input_df['ref_area']<=3000)]
	
	#plot the subsections of data 
	for subset,x in zip([first,second,third,fourth,fifth,sixth],range(6)): 
		ax[x].scatter(subset['ref_area'],subset['pred_area'],facecolors='none',edgecolors='black',alpha=0.5, s=40)

		#calculate a line of best fit 
		m, b = np.polyfit(subset['ref_area'], subset['pred_area'], 1)

		#add the fit statistics
		if x <1: 
			ax[x].annotate(f'n = {len(subset)}\nr = {str(round(pearsonr(subset["ref_area"],subset["pred_area"])[0],2))}\nm = {round(m,2)}',
						xy=(0.625,0.76),
						xycoords='axes fraction', 
						fontsize=fs)
		elif x == 1: 
			ax[x].annotate(f'n = {len(subset)}\nr = {str(round(pearsonr(subset["ref_area"],subset["pred_area"])[0],2))}\nm = {round(m,2)}',
						xy=(0.25,0.76),
						xycoords='axes fraction', 
						fontsize=fs)

		else: 
			ax[x].annotate(f'n = {len(subset)}\nr = {str(round(pearsonr(subset["ref_area"],subset["pred_area"])[0],2))}\nm = {round(m,2)}',
						   xy=(0.65,0.1),
						   xycoords='axes fraction', 
						   fontsize=fs)

		#add a line of best fit 
		ax[x].plot(subset['ref_area'], m*subset['ref_area'] + b,linestyle='--',color='red')

		#make the label ticks a little bigger 
		ax[x].tick_params(axis='both', which='major', labelsize=fs)
		ax[x].tick_params(axis='both', which='minor', labelsize=fs)

		#add a panel id letter 
		ax[x].annotate(f'{chr(97+x)}',
					   xy=(0.01,0.9),
					   xycoords='axes fraction',
					   fontsize=fs,
					   weight='bold'
					   )
		ax[x].tick_params(axis='both', which='both', labelsize=fs)
		ax[x].tick_params(axis='both', which='both', labelsize=fs)
		#uncomment to add text labels for a second so we can get the outliers 
		# metadata = pd.read_csv("/vol/v3/ben_ak/excel_files/glacier_wise_comparison/outliers.csv")
		# labels = subset[['ref_area','pred_area','RGIId']]
		# labels = labels.loc[labels['RGIId'].isin(metadata['RGIId'])] #(['RGI60-01-20822','RGI60-01-02223'])]#
		# labels.apply(lambda y: ax[x].text(*y,fontsize=5),axis=1)
		#subset[['ref_area','pred_area','RGIId']].apply(lambda y: ax[x].text(*y,fontsize=fs),axis=1)
	
	#add some little boxes to show where the outliers are that are included in the manuscript 
	# create a patch for the little glacier in brooks range
	rect1 = patches.Rectangle((1.25, 4), 0.4, 1, linewidth=1.5, edgecolor='r', facecolor='none')
	rect2 = patches.Rectangle((24.25, 17), 3, 4, linewidth=1.5, edgecolor='r', facecolor='none')
	# Add the patch to the Axes
	ax[1].add_patch(rect1)
	ax[3].add_patch(rect2)

	#set the axis limits 
	ax[0].set_xlim([0,1])

	ax[2].set_xlim([0, 25])
	ax[2].set_ylim([0, 25])

	ax[3].set_xlim([15, 60])
	ax[3].set_ylim([15, 60])

	ax[4].set_xlim([15, 100])
	ax[4].set_ylim([15, 100])

	ax[5].set_xlim([0,3000])
	ax[5].set_ylim([0,3000])

	#set axis ticks 
	ax[0].set_xticks(np.arange(0, 2, 1))
	ax[2].set_xticks(np.arange(0, 30, 5.0))
	ax[2].set_yticks(np.arange(0, 30, 5.0))

	ax[3].set_xticks(np.arange(20, 65, 10.0))
	ax[3].set_yticks(np.arange(20, 65, 10.0))

	ax[4].set_xticks(np.arange(15, 120, 15.0))
	ax[4].set_yticks(np.arange(15, 120, 15.0))

	ax[5].set_xticks(np.arange(0, 4000, 1000.0))
	ax[5].set_yticks(np.arange(0, 4000, 1000.0))


	#label the subplots
	ax[0].set_title('0—1 km$^2$',fontsize=fs)
	ax[1].set_title('1—5 km$^2$',fontsize=fs)
	ax[2].set_title('5—25 km$^2$',fontsize=fs)
	ax[3].set_title('25—50 km$^2$',fontsize=fs)
	ax[4].set_title('50—100 km$^2$',fontsize=fs)
	ax[5].set_title('100—3000 km$^2$',fontsize=fs)

	#add x axis labels
	ax[0].set_xlabel(' ')
	ax[1].set_xlabel(' ')
	ax[2].set_xlabel(' ')
	ax[3].set_xlabel('RGI area (km$^2$)',fontsize=fs)
	ax[4].set_xlabel('RGI area (km$^2$)',fontsize=fs)
	ax[5].set_xlabel('RGI area (km$^2$)',fontsize=fs)

	#add the y axis labels 
	ax[0].set_ylabel('GlacierCoverNet area (km$^2$)',fontsize=fs)
	ax[1].set_ylabel(' ')
	ax[2].set_ylabel(' ')
	ax[3].set_ylabel('GlacierCoverNet area (km$^2$)',fontsize=fs)
	ax[4].set_ylabel(' ')
	ax[5].set_ylabel(' ')

	plt.savefig(os.path.join(output_dir,'glacier_wise_comparison_unlabeled_remote_sensing_submission_v3.jpg'),
		dpi=1000, 
		bbox_inches = 'tight',
	    pad_inches = 0.1)
	# plt.show()
	# plt.close('all')

def calc_total_area_dif(pred_df,ref_df): 
	"""Output some stats on differences in total area."""
	return pred_df.r_area.sum(),ref_df.r_area.sum()

def main(input_dir,ref_shp,pickles,output_dir): 
	#output_data = {}
	output_fn = os.path.join(pickles,'rgi_cnn_area_comparison_by_glaciers_08122021_model_run_w_northern_revised.p')
	if not os.path.exists(output_fn): 

		output_list = []
		class_outputs = glob.glob(input_dir+'*buffer.shp')
		for year in range(2000,2013): 
			print(year)
			
			#read in the ref data and get only rows of years that match current year 
			ref_gdf = gpd.read_file(ref_shp)
			ref_gdf = ref_gdf.loc[ref_gdf['start_yr']==year]
			
			#subset the ref data for the year that we want 
			if year % 2 == 0: #even years which have classifications
				pred_file = [file for file in class_outputs if str(year) in file][0]
			else: #odd number years which don't have classifications
				pred_file = [file for file in class_outputs if str(year+1) in file][0]
			#run the comparison-generates a csv right now (6/3/2021)
			if not ref_gdf.empty: 
				output=compare_areas_by_glacier(pred_file,ref_gdf,pickles,year)
				#output_data.update({year:output})
				output_list.append(output)
			else: 
				print(f'There are no RGI data for the year {year}')
		output_df = pd.concat(output_list)
		pickle.dump(output_df, open(output_fn,'wb'))
	else: 
		print('reading from pickle')
		output_df = pickle.load( open(output_fn,'rb'))
	print(output_df.head(100))
	rgi = gpd.read_file("/vol/v3/ben_ak/vector_files/glacier_outlines/rgi_epsg_3338_US_bounds_merged.shp")
	rgi_test = rgi.loc[rgi['RGIId']=='RGI60-01.02216']
	print('the rgi test is: ',rgi_test)
	print(output_df.shape)
	print(output_df.dtypes)
	test = output_df.loc[output_df['RGIId']=='RGI60-01.02216']
	print('test is: ',test)
	#make a plot comparing the areas by glacier id 

	plot_glacier_wise_comparison(output_df,output_dir)

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		classified_dir = variables['classified_dir']
		output_dir = variables['output_dir']
		rgi_dir = variables['rgi_dir']
		pickles = variables['pickles']
		output_dir = variables['output_dir']
		
		main(classified_dir,rgi_dir,pickles,output_dir)