import geopandas as gpd 
import pandas as pd 
import numpy as np 
import os 
import sys 
import glob 
import json
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from functools import partial
import multiprocessing 
import re

#suppress the settingwithcopy warning from Pandas 
pd.options.mode.chained_assignment = None  # default='warn'

# class Joins(): 
# def __init__(self,shp1, shp2,file_list=None,output_dir=None,year=None,join_type='right',op='touches',epsg='EPSG:3338',col='rgi_label',**kwargs): 
# 	shp1=shp1
# 	shp2=shp2
# 	file_list=file_list
# 	output_dir=output_dir
# 	year=year
# 	join_type=join_type
# 	op=op
# 	epsg=epsg
# 	col=col
# 	for k,v in kwargs.items(): 
# 		setattr(self, k, v)

def pairwise_sj(shp1,shp2,output_dir,join_type='right',op='touches',epsg='EPSG:3338',col='rgi_label'):
	"""Perform a spatial join with two shapefiles. Significantly faster than QGIS."""
	print('Processing ', shp1)
	#read in the shapefiles 
	shp1_df = gpd.read_file(shp1)#.to_crs('EPSG:3338')
	#if you don't want to pass two shapefiles leave second one as None and this will deal with it 
	if not shp2: 
		shp2_df = shp1_df[shp1_df[col]==-1] #hardcoded for the non-labeled default from earlier in the workflow (-1)
	else: 
		shp2_df = gpd.read_file(shp2)
	#do a spatial join such that polygons touching other polygons are given a label of the parent polygon
	gdf = gpd.sjoin(shp1_df, shp2_df, how=join_type, op=op).dropna()
	gdf = gdf.sort_values('id_y', ascending=False).drop_duplicates('id_y').sort_index()
	try: 
		#previous step leaves redundant cols, rename them 
		gdf.rename(columns={f'{col}_x':'rgi_label'},inplace=True) #this assumes an order like left- base vector, right- unlabeled vector 
		#create output name
		subset = os.path.split(shp1)[0].split('/')[-1] #this is very clunky and hardcoded, basically we want the child dir name
		
		sjoin_out_filename = f'{os.path.split(shp1)[1][:-4]}_unlabeled_all_revised.shp' #get the filename from the boundaries 
		sjoin_out_filename = os.path.join(output_dir,subset,sjoin_out_filename)

		#returns a shapefile which is the unlabeled polygons but now with the label of their parent glacier and a shapefile of all the unlabeled polygons
		return gdf,sjoin_out_filename#,unlabeled

	except KeyError: 
		print('Processing general sjoin')
		sjoin_out_filename = os.path.join(output_dir,os.path.split(shp1)[1][:-4]+f'_sjoin_full_area_{year}_{join_type}_{op}.shp')
		return gdf, sjoin_out_filename

def attribute_join(shp1,shp2,output_dir,subset_id,epsg='EPSG:3338',col='rgi_label'): 
	"""Perform an attribute (table) join based on a common column between two shapefiles."""
	#create the filename 
	subset = os.path.split(shp2)[0].split('/')[-1] #this is very clunky and hardcoded, basically we want the child dir name
	sjoin_out_filename = f'{os.path.split(shp2)[1][:-4]}_extended_labels.shp' #get the filename from the boundaries 
	sjoin_out_filename = os.path.join(output_dir,subset,sjoin_out_filename)

	if os.path.exists(sjoin_out_filename): 
		print('The output file exists: ')
		return None
	#read in the shapefiles 
	shp1_df = gpd.read_file(shp1).dropna()#.to_crs('EPSG:3338')
	shp2_df = gpd.read_file(shp2).dropna() #this should be the base shapefile that has the parent glaciers 

	#cast the join cols as int to make sure they're the same 
	shp1_df['rgi_label'] = shp1_df['rgi_label'].astype('int')
	shp2_df['rgi_label'] = shp2_df['rgi_label'].astype('int')

	#testing a revised strategy 8/28/2021
	#first remove the -1 polygons (unlabeled) from the base polygons- these are the ones which remain unlabled after intial processing 
	shp1_df = shp1_df.loc[shp1_df['rgi_label'] != -1]
	shp2_df = shp2_df.loc[shp2_df['rgi_label'] != -1]

	#reset the index of the big dataframe to make sure there aren't issues with non-unique ids below 
	shp2_df.index = range(0,shp2_df.shape[0])
	
	#do the join- this should yield a geodataframe that has only the merged og glaciers and little bits that need to be added 
	select = shp2_df.merge(shp1_df, on='rgi_label',how='inner') #this should be like left: big vector with parent glaciers, right: little bits that need labels 
	#merge the geometries of the polygons that touch- this is based on matching labels from the join above 
	#I think this is where we are getting redundant areas- what we actually want to do is in cases where there are geometries to join, 
	#join them but in cases where there are no geometries to join then just preserve the original column. It seems like we're just losing the 
	#ones that don't have a partner in the join/merge above
	try: 
		select['geometry'] = select.apply(lambda x: cascaded_union([x.geometry_x,x.geometry_y]),axis=1) 
	except ValueError: 
		#there area a few subsets/years that have unlabled vectors but they are not going to get labels. In that case just use the base data  
		shp2_df['subset'] = subset_id
		return shp2_df,sjoin_out_filename
	#get rid of redundant (pre-merge) geometries 
	#output.drop(columns=['geometry_x','geometry_y','id_y','id_x',''],inplace=True)
	try: 
		select = select[['rgi_label','geometry']]
	except KeyError: 
		raise
	#make sure its cast as gdf not just df for export 
	try: 
		select = gpd.GeoDataFrame(select,crs=epsg,geometry=select['geometry'])
	except KeyError: 
		raise
	#dissolve anywhere that two glaciers are touching and the labels agree so there's just one polygon 
	try: 
		select = select.dissolve(by='rgi_label').reset_index()
	except AttributeError: 
		raise
	# #################################################################### added 5/21/2021
	# #there is an issue where polygons that do not have a bit added get dropped from the output, fix that 
	# #in this case shp2 is the original data 
	leftovers = shp2_df.loc[(~shp2_df['rgi_label'].astype(str).isin(select['rgi_label'].astype(str)))]
	output = pd.concat([select,leftovers[['rgi_label','geometry']]])
	#in the last step we need to remove some artifacts. Give a subset id so we can find these 
	output['subset'] = subset_id

	return output,sjoin_out_filename

def multiple_shp_sj(file_list,year,output_dir,model,remove_feats,data): 
	"""Merge a list of shapefiles, this is not explicitly a spatial join but more of a merge. 
	This will take the biggest area at a subset border which is keeping artifacts as of 8/16/2021."""

	#try for multiple shapefiles
	try: 
		#concat a list of shapefiles (list is just paths)
		gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in file_list], 
                        ignore_index=True), crs=gpd.read_file(file_list[0]).crs)
		#remove features that have an artifact from GlacierNet outputs 
		#first remove features from subset 9 
		gdf = gdf.loc[~((gdf['subset']=='subset_9') & (gdf['rgi_label'].isin(remove_feats)))]
		#then remove features from subset 12
		gdf = gdf.loc[~((gdf['subset']=='subset_12') & (gdf['rgi_label'].isin(remove_feats)))]
		#then from subset 7 
		gdf = gdf.loc[~((gdf['subset']=='subset_7') & (gdf['rgi_label'].isin(remove_feats)))]
		gdf = gdf.dissolve(by='rgi_label').reset_index()
		sjoin_out_filename=os.path.join(output_dir,f'{model}_{year}_{data}_output.shp')

	except Exception as e: 
		print(e) 
	return gdf, sjoin_out_filename

def write_sj_to_file(gdf,output_fn): 
	#write to file
	
	if not os.path.exists(output_fn): 
		#print('writing spatial join to file...')
		try: 
			gdf.to_file(output_fn)
		except (ValueError, TypeError):
			print('It appears that subset/year does not have any data.') 

	return output_fn

def run_step1_parallel(input_shp,output_dir): 
	"""Just used as a helper function to instantiate classes so it can be run in parallel."""

	#do the spatial join between base (parent) glacier layer and unlabeled glacier polygons
	newly_labeled = pairwise_sj(input_shp,None,output_dir=output_dir) #returns a tuple that is (gdf,filename)
	write_sj_to_file(newly_labeled[0],newly_labeled[1])

def run_step2_parallel(input_shp,output_dir,data,alt_unlabeled_vects): 
	"""Just used as a helper function to instantiate classes so it can be run in parallel."""
	subset_id = str(input_shp).split('/')[-2]
	year = re.findall('(\d{4})', os.path.split(input_shp)[1][8:])[0] #assumes the file starts with the model date (8 digits)
	try: 
		#this is getting the polygons that have been assigned a label in the previous step but need to be joined to their 
		#'parent' glacier 
		label_shp = [str(shp) for shp in alt_unlabeled_vects if (subset_id in str(shp).split('/')) & (year in str(shp))][0]
		joined_out=attribute_join(label_shp,input_shp,output_dir=output_dir,subset_id=subset_id)
	except IndexError: 
		#there are some instances where there are subsets with no polygons that didn't get an RGI label in the raster step
		#these will throw an index error when you try to index an empty list. In these cases we'd like to just return the original data
		print('There was no label_shp, returning base shape')
		sjoin_out_filename = f'{os.path.split(input_shp)[1][:-4]}_extended_labels.shp' #get the filename from the boundaries 
		sjoin_out_filename = os.path.join(output_dir,subset_id,sjoin_out_filename)
		#read in the original data as gdf and remove any remaining -1 polygons- these are ones that don't have an RGI label and 
		#should not get one because they are not in proximity to an existing labeled glacier 8/30/2021
		gdf = gpd.read_file(input_shp)
		gdf = gdf.loc[gdf['rgi_label'] != -1]
		gdf['subset'] = subset_id
		joined_out = (gdf,sjoin_out_filename)

	try: 
		if joined_out: 
			write_sj_to_file(joined_out[0],joined_out[1])
		else: 
			print('The file exists so we skip it.')
	except (TypeError, UnboundLocalError):
		raise

def add_dir(output_dir,subset_num): 
	output_dir = os.path.join(output_dir,f'subset_{subset_num}')
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)

def main(input_dir,output_dir,**kwargs): 
	"""Assign labels to polygons that should have an RGI label but did not get one in the raster-based labeling process. 
	This also removes polygons that do not have an association with an RGI id. Process should be run in three steps: 
	i) gets polygons that touch an existing labeled polygon and assigns them that RGI id. NOTE that in cases where a polygon
	touchs more than one labeled glacier it will take whichever the last polygon is so areas are not double counted. 
	ii) combines the labeled polygon bits from i) with the larger glaciers that were labeled in the raster step. 
	iii) takes the outputs of ii) which are still at the subset level and makes them into one study-area wide shapefile. 
	"""
	##########################################################################
	#make sure the directory structures are in place for the outputs 
	#check if the output dir exists and if not create one 
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)
	##########################################################################
	# i) first run the spatial join which gives previously unlabeled polygons which touch labeled polygons a label 
	# for x in range(1,17): 
	# 	add_dir(output_dir,x)

	# input_files = list(Path(input_dir).rglob('*0.01_min_size.shp')) #changed to just get the vectors with the highest level of processing
	
	# ##run a directory (all years) in parallel 
	# pool = multiprocessing.Pool(processes=25)
	
	# label=partial(run_step1_parallel, output_dir=output_dir)
	
	# #run it 
	# result_list = pool.map(label, input_files) 
	
	##########################################################################
	# ii) make join polygons which got a label in the previous step with their 'parent glaciers' 
	# for x in range(1,17): 
	# 	add_dir(output_dir,x)
	#print('Working...')
	# # #these should be the files that are the output of the previous vector processing step
	# input_files = list(Path(input_dir).rglob('*0.01_min_size.shp')) #changed to just get the vectors with the highest level of processing
	# # print(len(input_files))
 # # 	#these are the files produced by step i) above 
	# alt_unlabeled_vects = list(Path(kwargs.get('vect_dir')).rglob('*all_revised.shp'))
	# #generate outputs in parallel 
	# pool = multiprocessing.Pool(processes=25)
	# label=partial(run_step2_parallel, output_dir=output_dir,data=kwargs.get('data_type'),alt_unlabeled_vects=alt_unlabeled_vects)
	# result_list = pool.map(label, input_files)

	#do a test for one file
	# test_file = "/vol/v3/ben_ak/vector_files/neural_net_data/outputs/08122021_model_run/combined/subset_16/08122021_model_2004_subset_16_w_0.01_min_size.shp"
	# run_step2_parallel(test_file,output_dir,data=kwargs.get('data_type'),alt_unlabeled_vects=alt_unlabeled_vects)
	#test in a for loop instead of parallel for troubleshooting
	# for file in input_files: 
	# 	run_step2_parallel(file,output_dir,data=kwargs.get('data_type'),alt_unlabeled_vects=alt_unlabeled_vects)	

	##########################################################################
	# iii) do a merge/join to combine all the subsets of a given year. These are the outputs of the attribute join above  
	input_files = list(Path(input_dir).rglob('*extended_labels.shp'))
	print(len(input_files))
	for year in range(1986,2022,2): 
		sjoin_files = [str(file) for file in input_files if str(year) in str(file)]
		print('year is: ', year)
		output = multiple_shp_sj(sjoin_files,year,output_dir,kwargs.get('model'),
			kwargs.get('remove_feats'),kwargs.get('data_type'))
		write_sj_to_file(output[0],output[1])

if __name__ == '__main__':

	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		input_dir = variables['input_dir']
		clip_bounds = variables['clip_bounds']
		output_dir = variables['output_dir']
		model = variables['model_version']
		vect_dir = variables['vect_dir']
		data_type = variables['data_type']
		
	remove_feats = [9044,22342,22341,22335,22349,9164,9421,10232,8115,13830,14216,22889,26715,19814]
	main(input_dir,output_dir,vect_dir=vect_dir,model=model,clip_bounds=clip_bounds,remove_feats=remove_feats,data_type=data_type)#,field='pixelvalue',model=model,left_shp=left_shp)#,right_shp=att_2)
#working
# class Joins(): 
# 	def __init__(self,shp1, shp2,file_list=None,output_dir=None,year=None,join_type='right',op='touches',epsg='EPSG:3338',col='rgi_label',**kwargs): 
# 		shp1=shp1
# 		shp2=shp2
# 		file_list=file_list
# 		output_dir=output_dir
# 		year=year
# 		join_type=join_type
# 		op=op
# 		epsg=epsg
# 		col=col
# 		for k,v in kwargs.items(): 
# 			setattr(self, k, v)

# 	def pairwise_sj(self):
# 		"""Perform a spatial join with two shapefiles. Significantly faster than QGIS."""
# 		print('Processing ', shp1)
# 		#read in the shapefiles 
# 		shp1_df = gpd.read_file(shp1)#.to_crs('EPSG:3338')
# 		#if you don't want to pass two shapefiles leave second one as None and this will deal with it 
# 		if not shp2: 
# 			shp2_df = shp1_df[shp1_df[col]==-1] #hardcoded for the non-labeled default from earlier in the workflow (-1)
# 		else: 
# 			shp2_df = gpd.read_file(shp2)
# 		#do a spatial join such that polygons touching other polygons are given a label of the parent polygon
# 		gdf = gpd.sjoin(shp1_df, shp2_df, how=join_type, op=op).dropna()
# 		gdf = gdf.sort_values('id_y', ascending=False).drop_duplicates('id_y').sort_index()
# 		try: 
# 			#previous step leaves redundant cols, rename them 
# 			gdf.rename(columns={f'{col}_x':'rgi_label'},inplace=True) #this assumes an order like left- base vector, right- unlabeled vector 
# 			#gdf.rename(columns={f'{col}_left':'rgi_label',f'{col}_right':'no_label'},inplace=True)
			
# 			# #cast cols as int because they became float somewhere along the way 
# 			# gdf['rgi_label'] = gdf['rgi_label'].astype('int')
# 			# gdf['no_label'] = gdf['no_label'].astype('int')

# 			# #right now the output that gets RGI labels and those that don't are two files, make them the same one 
# 			# #just using concat makes it so that polygons with RGI labels are overwritten with -1. make it so RGI labeled polygons are preserved by removing them from shp2_df
# 			# leftovers = shp2_df.loc[(shp2_df['rgi_label']==-1)&
# 			# 				(~shp2_df['id'].astype(str).isin(gdf['id_x'].astype(str)))&
# 			# 				(~shp2_df['id'].astype(str).isin(gdf['id_y'].astype(str)))]

# 			# unlabeled = pd.concat([leftovers,gdf])
# 			#create output name
# 			subset = os.path.split(shp1)[0].split('/')[-1] #this is very clunky and hardcoded, basically we want the child dir name
			
# 			sjoin_out_filename = f'{os.path.split(shp1)[1][:-4]}_unlabeled_all_revised.shp' #get the filename from the boundaries 
# 			sjoin_out_filename = os.path.join(output_dir,subset,sjoin_out_filename)

# 			#returns a shapefile which is the unlabeled polygons but now with the label of their parent glacier and a shapefile of all the unlabeled polygons
# 			return gdf,sjoin_out_filename#,unlabeled

# 		except KeyError: 
# 			print('Processing general sjoin')
# 			sjoin_out_filename = os.path.join(output_dir,os.path.split(shp1)[1][:-4]+f'_sjoin_full_area_{year}_{join_type}_{op}.shp')
# 			return gdf, sjoin_out_filename

# 	def multiple_shp_sj(self,remove_feats): 
# 		"""Merge a list of shapefiles, this is not explicitly a spatial join but more of a merge. 
# 		This will take the biggest area at a subset border which is keeping artifacts as of 8/16/2021."""

# 		#try for multiple shapefiles
# 		try: 
# 			#concat a list of shapefiles (list is just paths)
# 			gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in file_list], 
# 	                        ignore_index=True), crs=gpd.read_file(file_list[0]).crs)
# 			#remove features that have an artifact from GlacierNet outputs 
# 			#first remove features from subset 9 
# 			gdf = gdf.loc[~((gdf['subset']=='subset_9') & (gdf['rgi_label'].isin(remove_feats)))]
# 			#then remove features from subset 12
# 			gdf = gdf.loc[~((gdf['subset']=='subset_12') & (gdf['rgi_label'].isin(remove_feats)))]
# 			gdf = gdf.loc[~((gdf['subset']=='subset_7') & (gdf['rgi_label'].isin(remove_feats)))]
# 			gdf = gdf.dissolve(by='rgi_label').reset_index()
# 			sjoin_out_filename=os.path.join(output_dir,f'{model}_{year}_dissolve_test.shp')

# 		except Exception as e: 
# 			print(e) 
# 		return gdf, sjoin_out_filename

# 	def attribute_join(self): 
# 		"""Perform an attribute (table) join based on a common column between two shapefiles."""
# 		#create the filename 
# 		subset = os.path.split(shp2)[0].split('/')[-1] #this is very clunky and hardcoded, basically we want the child dir name
# 		sjoin_out_filename = f'{os.path.split(shp2)[1][:-4]}_extended_labels_testing_again.shp' #get the filename from the boundaries 
# 		sjoin_out_filename = os.path.join(output_dir,subset,sjoin_out_filename)

# 		if os.path.exists(sjoin_out_filename): 
# 			return None 
# 		#read in the shapefiles 
# 		try: 
# 			shp1_df = gpd.read_file(shp1).dropna()#.to_crs('EPSG:3338')
# 		except Exception as e: 
# 			shp1_df = shp1

# 		shp2_df = gpd.read_file(shp2).dropna() #this should be the base shapefile that has the parent glaciers 

# 		#cast the join cols as int to make sure they're the same 
# 		shp1_df['rgi_label'] = shp1_df['rgi_label'].astype('int')
# 		shp2_df['rgi_label'] = shp2_df['rgi_label'].astype('int')

# 		#testing a revised strategy 8/28/2021
# 		#first remove the -1 polygons (unlabeled) from the base polygons- these are the ones which remain unlabled after intial processing 
# 		shp1_df = shp1_df.loc[shp1_df['rgi_label'] != -1]
# 		shp2_df = shp2_df.loc[shp2_df['rgi_label'] != -1]

# 		#reset the index of the big dataframe to make sure there aren't issues with non-unique ids below 
# 		shp2_df.index = range(0,shp2_df.shape[0])
# 		print('shp 2 is: ')
# 		print(shp2_df.head())
# 		print(shp2_df.shape)
# 		# try: 
# 		# 	#remove these cols from either df, they are leftover from spatial joining and will confuse this step 
# 		# 	shp1_df.drop(columns['index_left'],inplace=True)
# 		# 	shp2_df.drop(columns['index_left'],inplace=True)

# 		# except Exception as e: 
# 		# 	print('the index_left col does not appear to exist in this df')
		
# 		#do the join- this should yield a geodataframe that has only the merged og glaciers and little bits that need to be added 
# 		select = shp2_df.merge(shp1_df, on='rgi_label',how='inner') #this should be like left: big vector with parent glaciers, right: little bits that need labels 
# 		print(select.shape)
# 		#merge the geometries of the polygons that touch- this is based on matching labels from the join above 
# 		#I think this is where we are getting redundant areas- what we actually want to do is in cases where there are geometries to join, 
# 		#join them but in cases where there are no geometries to join then just preserve the original column. It seems like we're just losing the 
# 		#ones that don't have a partner in the join/merge above
# 		select['geometry'] = select.apply(lambda x: cascaded_union([x.geometry_x,x.geometry_y]),axis=1) 
# 		# # #get rid of redundant (pre-merge) geometries 
# 		# #output.drop(columns=['geometry_x','geometry_y','id_y','id_x',''],inplace=True)
# 		select = select[['rgi_label','geometry']]
# 		# # #make sure its cast as gdf not just df for export 
# 		select = gpd.GeoDataFrame(select,crs=epsg,geometry=select['geometry'])

# 		#dissolve anywhere that two glaciers are touching and the labels agree so there's just one polygon 
# 		select = select.dissolve(by='rgi_label').reset_index()

# 		# #################################################################### added 5/21/2021
# 		# #there is an issue where polygons that do not have a bit added get dropped from the output, fix that 
# 		# #in this case shp2 is the original data 
# 		leftovers = shp2_df.loc[(~shp2_df['rgi_label'].astype(str).isin(select['rgi_label'].astype(str)))]
# 		output = pd.concat([select,leftovers[['rgi_label','geometry']]])
		
# 		return output,sjoin_out_filename

# 	def write_sj_to_file(self,gdf,output_fn): 
# 		#write to file
# 		print('writing spatial join to file...')
# 		if not os.path.exists(output_fn): 
# 			try: 
# 				gdf.to_file(output_fn)
# 			except (ValueError, TypeError):
# 				raise
# 				print('It appears that subset/year does not have any data.') 

# 		print('output written')

# 		return output_fn