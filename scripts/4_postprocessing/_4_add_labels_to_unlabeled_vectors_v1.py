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

def add_dir(output_dir,subset_num): 
	output_dir = os.path.join(output_dir,f'subset_{subset_num}')
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)

def add_persistence_label(gdf,output_dir): 
	"""Add the actual persistence label of the types laid out above."""
	#gdf = gdf.drop_duplicates('geometry')
	gdf['group_count'] = gdf.groupby('non_rgi_id')['non_rgi_id'].transform('count')

def create_group_ids(input_shps,output_dir): 
	"""Stack vector files for multiple years and assign a unique id to a group.
	NOTE that this is done on a subset basis so there will be overlap in ids across subsets 
	but not in the output product as we're only interested in the frequency."""

	#make a big geodataframe from all the years in a subset area 
	gdf = pd.concat([gpd.read_file(file) for file in input_shps])
	
	#make sure the program understands that is a geodataframe 
	gdf = gpd.GeoDataFrame(gdf,crs='EPSG:3338',geometry=gdf['geometry'])
	
	#union the layers so they form groups across years
	#this approach comes from: 
	#https://gis.stackexchange.com/questions/334459/how-to-dissolve-overlapping-polygons-using-geopandas 
	geoms = gdf.geometry.unary_union

	#assign the unioned features as the geometry of the geodataframe 
	gdf = gpd.GeoDataFrame(geometry=[geoms])

	#convert multipart geometries to single part (multipolygon to polygon)
	gdf = gdf.explode()#.reset_index(drop=True)
	
	#add a unique identifier to each group 
	gdf['non_rgi_id'] = range(-2,((len(gdf)+2)*-1),-1)
	gdf.to_file(os.path.join(output_dir,'base_groups_unioned_test_output.shp'))

	return gdf

def spatial_join(shps,base_shp): #rename
	"""Do a spatial join between each year in a subset area and the unioned shape that was created by the previous function.
	This attaches the unique group ID to each feature in a shapefile and makes it so we can check how many times that id shows up over time."""
 
	gdfs = {}
	#loop through a subdirectory of vector files for one subset (one file per year(s))
	for shp in shps: 
		year = re.findall('(\d{4})', os.path.split(shp)[1][8:])[0] #assumes the file starts with the model date (8 digits)
		
		shp = gpd.read_file(shp)
		
		
		try:
			#if 'index_left' is still there from a previous join get rid of it b/c it will error 
			shp.drop(columns=['index_left'],inplace=True)
		except KeyError: 
			pass

		try: 
			base_shp.drop(columns=['index_left','index_right'],inplace=True)
		except KeyError as e: 
			pass
		#join the year's vector with the base group vector thereby assigning a group id 
		gdf = gpd.sjoin(shp, base_shp, how='left', op='intersects')

		#write the result to disk 
		# gdf.to_file(os.path.join(output_dir,f'merged_w_no_rgi_label_{count}.shp'))

		gdfs.update({year:gdf})
	#return a dict like {'file year':df} so we know where the df came from 
	return gdfs

def pairwise_sj(shp1,shp2,output_dir,col='rgi_label',epsg='EPSG:3338',join_type='right',op='touches'):
	"""Perform a spatial join with two shapefiles. Significantly faster than QGIS."""
	print('Processing ', shp1)
	#read in the shapefiles 
	shp1_df = gpd.read_file(shp1)#.to_crs('EPSG:3338')

	#if you don't want to pass two shapefiles leave second one as None and this will create a subset of polygons w/o labels 
	if not shp2: 
		shp2_df = shp1_df[shp1_df[col]==-1] #hardcoded for the non-labeled default from earlier in the workflow (-1)
	else: 
		shp2_df = gpd.read_file(shp2)#.to_crs('EPSG:3338')

	#do a spatial join such that polygons touching other polygons are given a label of the parent polygon
	gdf = gpd.sjoin(shp1_df, shp2_df, how=join_type, op=op).dropna()

	#previous step leaves redundant cols, rename them 
	gdf.rename(columns={f'{col}_x':'rgi_label',f'{col}_y':'no_label'},inplace=True) #this assumes an order like left- base vector, right- unlabeled vector 
	
	#cast cols as int because they became float somewhere along the way 
	gdf['rgi_label'] = gdf['rgi_label'].astype('int')
	gdf['no_label'] = gdf['no_label'].astype('int')

	#right now the output that gets RGI labels and those that don't are two files, make them the same one 
	#just using concat makes it so that polygons with RGI labels are overwritten with -1. make it so RGI labeled polygons are preserved by removing them from shp2_df
	leftovers = shp2_df.loc[(shp2_df['rgi_label']==-1)&
					(~shp2_df['id'].astype(str).isin(gdf['id_x'].astype(str)))&
					(~shp2_df['id'].astype(str).isin(gdf['id_y'].astype(str)))]

	unlabeled = pd.concat([leftovers,gdf])
	#create output name
	subset = os.path.split(shp1)[0].split('/')[-1] #this is very clunky and hardcoded, basically we want the child dir name
	
	sjoin_out_filename = f'{os.path.split(shp1)[1][:-4]}_unlabeled_all_revised.shp' #get the filename from the boundaries 
	sjoin_out_filename = os.path.join(output_dir,subset,sjoin_out_filename)

	#returns a shapefile which is the unlabeled polygons but now with the label of their parent glacier and a shapefile of all the unlabeled polygons
	return gdf,sjoin_out_filename,unlabeled
		

def multiple_shp_sj(file_list,output_dir): 
	"""Merge a list of shapefiles, this is not explicitly a spatial join but more of a merge. 
	This will take the biggest area at a subset border which is keeping artifacts as of 8/16/2021."""

	#try for multiple shapefiles
	try: 
		#concat a list of shapefiles (list is just paths)
		gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in file_list], 
                        ignore_index=True), crs=gpd.read_file(file_list[0]).crs)
		#dissolve anywhere that two glaciers are touching and the labels agree so there's just one polygon 
		#gdf = gdf.groupby(['rgi_label']).agg({'geometry':'first','sj_area':'min'}).reset_index()
		#gdf = gdf.loc[gdf.groupby("rgi_label")["sj_area"].idxmin()]
		#gdf = gdf.dissolve(by='rgi_label',aggfunc='min').reset_index()
		sjoin_out_filename=os.path.join(output_dir,f'{model}_{year}_no_dissolve.shp')

	except Exception as e: 
		print(e) 
	return gdf, sjoin_out_filename

def attribute_join(shp1,shp2,output_dir,epsg='EPSG:3338'): 
	"""Perform an attribute (table) join based on a common column between two shapefiles."""
	#create the filename 
	subset = os.path.split(shp2)[0].split('/')[-1] #this is very clunky and hardcoded, basically we want the child dir name
	sjoin_out_filename = f'{os.path.split(shp2)[1][:-4]}_extended_labels.shp' #get the filename from the boundaries 
	sjoin_out_filename = os.path.join(output_dir,subset,sjoin_out_filename)

	if os.path.exists(sjoin_out_filename): 
		return None 
	
	#read in the shapefiles 
	try: 
		shp1_df = gpd.read_file(shp1).dropna()#.to_crs('EPSG:3338')
	except Exception as e: 
		shp1_df = shp1

	shp2_df = gpd.read_file(shp2).dropna() #this should be the base shapefile that has the parent glaciers 

	#cast the join cols as int to make sure they're the same 
	shp1_df['rgi_label'] = shp1_df['rgi_label'].astype('int')
	shp2_df['rgi_label'] = shp2_df['rgi_label'].astype('int')


	try: 
		#remove these cols from either df, they are leftover from spatial joining and will confuse this step 
		shp1_df.drop(columns['index_left'],inplace=True)
		shp2_df.drop(columns['index_left'],inplace=True)

	except Exception as e: 
		print('the index_left col does not appear to exist in this df')
	
	#do the join
	output = shp2_df.merge(shp1_df, on='rgi_label',how='inner') #this should be like left: big vector with parent glaciers, right: little bits that need labels 

	#get a subset of the joined gdf that does not include nan (no touching) or -1 (things that are not going to get an RGI label)
	select = output.loc[(output['rgi_label']!=-1)&(output['rgi_label']!=np.nan)]
	if select.empty: 
		return None 
	#merge the geometries of the polygons that touch- this is based on matching labels from the join above 
	select['geometry'] = select.apply(lambda x: cascaded_union([x.geometry_x,x.geometry_y]),axis=1)

	#get rid of redundant (pre-merge) geometries 
	select.drop(columns=['geometry_x','geometry_y'],inplace=True)

	#make sure its cast as gdf not just df for export 
	select = gpd.GeoDataFrame(select,crs=epsg,geometry=select['geometry'])

	#dissolve anywhere that two glaciers are touching and the labels agree so there's just one polygon 
	select = select.dissolve(by='rgi_label').reset_index()

	#################################################################### added 5/21/2021
	#there is an issue where polygons that do not have a bit added get dropped from the output, fix that 
	#in this case shp2 is the original data 
	leftovers = shp2_df.loc[(shp2_df['rgi_label']!=-1)&
					(~shp2_df['rgi_label'].astype(str).isin(select['rgi_label'].astype(str)))]
	
	select = pd.concat([select,leftovers])
	# #create the filename 
	# subset = os.path.split(shp2)[0].split('/')[-1] #this is very clunky and hardcoded, basically we want the child dir name
	
	# sjoin_out_filename = f'{os.path.split(shp2)[1][:-4]}_extended_labels.shp' #get the filename from the boundaries 
	# sjoin_out_filename = os.path.join(output_dir,subset,sjoin_out_filename)

	return select,sjoin_out_filename

def write_sj_to_file(self,gdf,output_fn): 
	#write to file
	print('writing spatial join to file...')
	if not os.path.exists(output_fn): 
		try: 
			gdf.to_file(output_fn)
		except (ValueError, TypeError):
			print('It appears that subset/year does not have any data.') 

	print('output written')

	return output_fn
def run_step1_parallel(input_shp,output_dir): 
	"""Just used as a helper function to instantiate classes so it can be run in parallel."""

	#do the spatial join between base (parent) glacier layer and unlabeled glacier polygons

	newly_labeled = pairwise_sj(input_shp,None,output_dir=output_dir) #returns a tuple that is (gdf,filename)

	write_sj_to_file(newly_labeled[2],newly_labeled[1])


def run_step2_parallel(input_shp,output_dir,alt_unlabeled_vects): 
	"""Just used as a helper function to instantiate classes so it can be run in parallel."""

	#this set up should be considered temporary (5/24/2021) as it requires running two scripts to get the outputs 
	subset_id = str(input_shp).split('/')[-2]
	
	year = re.findall('(\d{4})', os.path.split(input_shp)[1][8:])[0] #assumes the file starts with the model date (8 digits)
	
	try: 
		label_shp = [str(shp) for shp in alt_unlabeled_vects if (subset_id in str(shp)) & (year in str(shp))][0]
		
	except IndexError: 
		print('you are likely looking for a subset which has no data')
	#attribute join to make RGI labeled polygons
	joined_out=attribute_join(gpd.read_file(label_shp),input_shp,output_dir=output_dir)

	try: 
		write_sj_to_file(joined_out[0],joined_out[1])
	except TypeError: 
		print('It looks like that df has no data')

def main(input_dir,output_dir,**kwargs): 
	
	##########################################################################
	##########Use this section to do  spatial joins for two shapefiles or joins on part of the same shapefile (should also work with geojson)
	##########This is the section to use when doing the labeling scheme for GlacierNet outputs with a spatial join and attribute join 
	##########################################################################
	#check if the output dir exists and if not create one 
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)

	#only needs to be run the first time
	for x in range(1,17): 
		add_dir(output_dir,x)

	##########################################################################
	# i) first run the spatial join to get areas which are missing an id
	#running
	# input_files = list(Path(input_dir).rglob('*1_min_size.shp')) #changed to just get the vectors with the highest level of processing
	# #generate outputs	
	# #run it all
	# pool = multiprocessing.Pool(processes=25)
	
	# label=partial(run_step1_parallel, output_dir=output_dir)
	
	# #run it 
	# result_list = pool.map(label, input_files) 
	##########################################################################
	# ii) next add the persistence thing and associated group id. This section is likely depreceated as of 8/15/2021 but is 
	#still being used currently to generate intermediate inputs to the final step. 

	#get subdirs in the parent dir - this is badly setup so the parent dir has to finsih with /*/ for this to iterate through the children
	# dirs = glob.glob(os.path.join(input_dir,'*/'))
	# print(dirs)
	# for child in dirs: 
		
	# 	subset_id = child.split('/')[-2] #hardcoded for the structure of the files and directory 
	# 	print(child)
	# 	#get a list of vector files like: one file/year/subset area with all unlabeled areas, RGI included 
	# 	shps = glob.glob(child+'*all_revised.shp')
		
	# 	#unioned year vectors per subset area with unique ids attached to each group 
	# 	base_union=create_group_ids(shps,output_dir).set_crs('EPSG:3338')
	# 	# print('base')
	# 	#print(base_union) #unique groups across the full time series 

	# 	#assign a unique id to each feature in each vector file by subset 
	# 	gdfs=persistence.spatial_join(shps,base_union)
	 	
	#  	#this is a df that has all the years for a subset with the group ids. use to calculate persistence etc. 
	# 	merged = pd.concat(gdfs.values())
		
	# 	#clean up this df a bit 
	# 	try: 
	# 		merged = merged[['rgi_label','r_area','non_rgi_id','geometry']]
	# 	except KeyError as e: 
	# 		print(f'The error at column selection was: {e}')
	# 		area_col = input('please write the name of the area col you want: ')
	# 		merged = merged[['rgi_label',str(area_col),'non_rgi_id','geometry']]

	# 	##############################
	# 	#add the actual counting labels 
	# 	##############################
	# 	merged = persistence.add_persistence_label(merged,output_dir)
	# 	#just get one number per id group 
	# 	merged = merged.drop_duplicates('non_rgi_id')
	# 	#make a dict of ids and counts to join to the group shapefile 
	# 	count_dict = dict(zip(merged['non_rgi_id'],merged['group_count']))
	# 	base_union['persistence'] = base_union['non_rgi_id'].map(count_dict)
	# 	#make a col which is the number of times that group shows up/18 possible periods 
	# 	base_union['persistence'] = base_union['persistence']/18 #hardcoded for the current (5/23/2021) number of yearly composites 
	# 	#create a new dict that is just ids and persistence to join to each year's vector for each subset 
	# 	persist_dict = dict(zip(base_union['non_rgi_id'],base_union['persistence']))
		
	# 	for k,v in gdfs.items(): 
	# 		#remove some junk 
	# 		try: 
	# 			gdf = v[['rgi_label','r_area','non_rgi_id','geometry']]
	# 		except KeyError as e: 
	# 			print(f'Error is: {e}')
	# 			raise
		
	# 		#add the persistence col to each year/subset 
	# 		gdf['persistence']=gdf['non_rgi_id'].map(persist_dict)

	# 		output_file = os.path.join(child,f'{kwargs.get('model')}_model_{k}_{subset_id}_group_id_added.shp')

	# 		if not os.path.exists(output_file): 
	# 			print('Making file')
	# 			gdf.to_file(output_file)
	# 		else: 
	# 			print('That file already exists')

	##########################################################################
	# iii) finally make the final output with the attribute join and write to disk 
	#running
	#these should be the files that are the output of the previous vector processing step 
	input_files = list(Path(input_dir).rglob('*0.01_min_size.shp')) #changed to just get the vectors with the highest level of processing
	print(len(input_files))
 	#these are the files produced by step two above 
	alt_unlabeled_vects = list(Path(kwargs.get('vect_dir')).rglob('*all_revised.shp'))
	print(len(alt_unlabeled_vects))

	#generate outputs
	#test run one file
	#run_parallel(input_files[0],output_dir)

	#run it all
	pool = multiprocessing.Pool(processes=25)
	
	label=partial(run_step2_parallel, output_dir=output_dir,alt_unlabeled_vects=alt_unlabeled_vects)
	
	#run it 
	result_list = pool.map(label, input_files)

	##########################################################################
	# iv) do a merge/join to combine all the subsets of a given year. These are the outputs of the attribute join above  
	input_files = list(Path(input_dir).rglob('*extended_labels.shp'))
	print(len(input_files))
	for year in range(1986,2022,2): 
		sjoin_files = [str(file) for file in input_files if str(year) in str(file)]
		print('year is: ',year)
		sjoin = Joins(None,None,file_list=sjoin_files,year=year,output_dir=output_dir,**kwargs)
		output = sjoin.multiple_shp_sj()
		sjoin.write_sj_to_file(output[0],output[1])

if __name__ == '__main__':

	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		input_dir = variables['input_dir']
		clip_bounds = variables['clip_bounds']
		output_dir = variables['output_dir']
		model = variables['model_version']
		vect_dir = variables['vect_dir']
		
	#uncomment to run some of the other functionality (e.g. joins on multiple files)
	main(input_dir,output_dir,vect_dir=vect_dir,model=model,clip_bounds=clip_bounds,)#,field='pixelvalue',model=model,left_shp=left_shp)#,right_shp=att_2)
