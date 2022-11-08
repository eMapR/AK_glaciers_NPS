import os
import sys 
import json 
from osgeo import gdal
from rasterstats import zonal_stats,point_query
import rasterio
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time
import pickle 
import matplotlib as mpl
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from osgeo.gdalnumeric import *  
from osgeo.gdalconst import * 
import re
import glob
import geopandas as gpd 
from shapely.geometry import Point
import fiona
from geopandas import GeoDataFrame
import seaborn as sns


def calc_zonal_stats(predicted_raster,reference_data,stat,resolution=30,noData=-32768,crs='3338',*args,**kwargs): 
	"""Calculate pixel counts inside polygons."""
	if reference_data.endswith('.shp'): 
		gdf = gpd.read_file(shp)

	elif reference_data.endswith('.csv'):
		try:  
			df = pd.read_csv(reference_data)
			
			#added 8/26/2021
			# print(df.head())
			# df = df.dropna()
			# print('transformed: ')
			# print(df.head())
		except Exception as e: 
			print('Reading data from csv and formatting...')
			df = pd.read_csv(reference_data,sep='\t')
			#create the gdf 
		geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
		crs = f'EPSG:{crs}'
		gdf = GeoDataFrame(df, crs=crs, geometry=geometry)
	else: 
		print('You need to supply either a shapefile (.shp) or a csv with lat/lon fields to calc_zonal_stats func.')

	with rasterio.open(predicted_raster,'r') as src: 
		#construct the transform tuple in the form: top left (x coord), west-east pixel res, rotation (0.0), top left northing (y coord), rotation (0.0), north-south pixel res (-1*res)
		transform = (src.bounds[0],float(resolution),0.0,src.bounds[3],0.0,-1*float(resolution))
		arr = src.read(1).astype('float')
		#this was changed 8/26/2021 for the default GlacierNet output which is a three class raster (0=no glacier, 1=supraglacial debris, 2=debris free ice)
		arr = np.where(arr > 0, 1,0)
		#arr[arr > 0] = 1 
		print(np.max(arr))
		print(np.min(arr))
		print(arr)
		print(arr.shape)

	#rasterstats zonal stats produces a list of dicts, get the value
	stats = zonal_stats(gdf,arr,stats=stat,transform=transform,nodata=noData) #gets the values at the raster pixels for the given points
	#print(stats)
	output_gdf = gdf.join(pd.DataFrame(stats))
	print('shape here is: ')
	print(output_gdf.shape)
	print(output_gdf)
	output_gdf=output_gdf.replace(np.inf, 0)
	output_gdf=output_gdf.replace(np.nan,0)
	return output_gdf
	

def calc_confusion_matrix(zonal_stats_df,stat,reference_col='binary',*args,**kwargs):#actual_source,predicted_source,stat,): 
	"""Calculate a confusion matrix to compare nlcd or rgi and classification."""

	#clean up the df a bit 
	zonal_stats_df.index = np.arange(1,len(zonal_stats_df)+1)


	#format the df 
	#first remove bad values 
	if 'remove_field' in kwargs: 
		zonal_stats_df = zonal_stats_df[zonal_stats_df[kwargs.get('remove_field')]!=kwargs.get('remove_val')]
		#zonal_stats_df = zonal_stats_df[zonal_stats_df[kwargs.get('remove_field')]!='m']
		#print(df.shape)
	try: 
		#make sure this is cast to string
		zonal_stats_df['class'] = zonal_stats_df['class'].astype('str')
		#remove whitespace that might be in there by accident 
		zonal_stats_df['class'] = zonal_stats_df['class'].str.replace(' ','') #field is hardcoded 
		zonal_stats_df = zonal_stats_df[zonal_stats_df['class']!='u'] #remove the undecided points
		zonal_stats_df['binary'] = zonal_stats_df['class'].map({'u': 0, 's': 0,'w':0,'n':0,'g':1,'d':1,'ha':1,'tw':1,'sh':1}) #hardcoded relationships 
	except KeyError as e: 
		print('The column class does not exist and therefore undecided will not be removed \n and the binary col will not be created.')
	
	zonal_stats_df.to_file("/vol/v3/ben_ak/vector_files/neural_net_data/validation/southern_region_pts_revised.shp")

	#zonal_stats_df = zonal_stats_df.dropna()#(np.nan,0)
	zonal_stats_df = zonal_stats_df[(zonal_stats_df[reference_col]==1.0) | (zonal_stats_df[reference_col]==0.0)]
	pd.set_option('display.max_rows', None)
	reference_data = [float(i) for i in list(zonal_stats_df[reference_col])]
	predicted_data = [float(i) for i in list(zonal_stats_df[stat])]
	
	ids = zonal_stats_df.index

	#labels = sorted(list(set(list(actual_col)+list(predicted_col))))

	results = confusion_matrix(reference_data, predicted_data)#,zonal_stats_df.index) 
	print(results)
	#disp = plot_confusion_matrix(None,actual_ls,predicted_ls,display_labels=labels,cmap=plt.cm.Blues)
	#fig,(ax,ax1) = plt.subplots(nrows=1,ncols=2)
	ax=plt.subplot()
	sns.set(font_scale=3)  # crazy big
	sns.heatmap(results,annot=True,ax=ax,fmt='g',cmap='Blues')
	#ax.collections[0].colorbar.ax.set_ylim(0,400)

	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
	#print(results) 
	# print ('Accuracy Score :',accuracy_score(actual_ls, predicted_ls))
	# print ('Report : ')
	# print (classification_report(actual_ls, predicted_ls))
	print(classification_report(reference_data,predicted_data))
	classification_output = pd.DataFrame(classification_report(reference_data,predicted_data,output_dict=True)).transpose().reset_index().rename(columns={'index':'stat'})
	classification_output['model_accuracy'] = accuracy_score(reference_data,predicted_data)
	# print('output is: ', classification_output)
	# print(classification_output.index)
	#print('output is: ',type(classification_report(actual_ls,predicted_ls,output_dict=True)))
	# 	report = classification_report(y_test, y_pred, output_dict=True)
	# and then construct a Dataframe and transpose it:

	#df = pandas.DataFrame(report).transpose()
	plt.show()
	plt.close('all')
	return classification_output#, incorrect, false_positives, false_negatives

def main(predicted_raster,reference_data,stat,**kwargs): 

	######################################################################################################
	#make confusion matrix
	#error_stats=calc_confusion_matrix(None,classified_raster,random_pts,resolution,stat,actual_source,predicted_source,model_run,write_to_pickle,pickle_dir,modifier,uncertainty_layer)
	zonal_stats = calc_zonal_stats(predicted_raster,reference_data,stat,**kwargs)
	output=calc_confusion_matrix(zonal_stats,stat,**kwargs)
	print(output)
	# #get a csv of the points that were incorrectly classified so we can see where they are located- this is the second output of the calc_confusion_matrix func
	# error_stats[1].to_csv(output_dir+modifier.replace(' ','_')+'_incorrect_points.csv')
	# #get csv of false positives
	# error_stats[2].to_csv(output_dir+modifier.replace(' ','_')+'_false_positives.csv')
	# error_stats[3].to_csv(output_dir+modifier.replace(' ','_')+'_false_negatives.csv')
if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		predicted_raster = variables['predicted_raster']
		reference_data = variables['reference_data']
		stat = variables['stat']
	#note that for 2016 generated data there is a remove field 'discard' but for data generated in GEE for 1990 that field does not exist
	main(predicted_raster,reference_data,stat)#,remove_field='discard',remove_val='y')