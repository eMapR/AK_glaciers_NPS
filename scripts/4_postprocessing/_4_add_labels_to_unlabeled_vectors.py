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
# import _5_assign_new_label_scheme as persistence #8/16/2021 this will create an error when it runs because this module will be missing 
pd.options.mode.chained_assignment = None  # default='warn'

class Joins(): 
	def __init__(self,shp1, shp2,file_list=None,output_dir=None,year=None,join_type='right',op='touches',epsg='EPSG:3338',col='rgi_label',**kwargs): 
		self.shp1=shp1
		self.shp2=shp2
		self.file_list=file_list
		self.output_dir=output_dir
		self.year=year
		self.join_type=join_type
		self.op=op
		self.epsg=epsg
		self.col=col
		for k,v in kwargs.items(): 
			setattr(self, k, v)

	def pairwise_sj(self):
		"""Perform a spatial join with two shapefiles. Significantly faster than QGIS."""
		print('Processing ', self.shp1)
		#read in the shapefiles 
		shp1_df = gpd.read_file(self.shp1)#.to_crs('EPSG:3338')
	
		#if you don't want to pass two shapefiles leave second one as None and this will deal with it 
		if not self.shp2: 
			shp2_df = shp1_df[shp1_df[self.col]==-1] #hardcoded for the non-labeled default from earlier in the workflow (-1)
		else: 
			shp2_df = gpd.read_file(self.shp2)#.to_crs('EPSG:3338')

		#do a spatial join such that polygons touching other polygons are given a label of the parent polygon
		 
		gdf = gpd.sjoin(shp1_df, shp2_df, how=self.join_type, op=self.op).dropna()

		try: 
			#previous step leaves redundant cols, rename them 
			gdf.rename(columns={f'{self.col}_x':'rgi_label',f'{self.col}_y':'no_label'},inplace=True) #this assumes an order like left- base vector, right- unlabeled vector 
			#gdf.rename(columns={f'{self.col}_left':'rgi_label',f'{self.col}_right':'no_label'},inplace=True)
			
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
			subset = os.path.split(self.shp1)[0].split('/')[-1] #this is very clunky and hardcoded, basically we want the child dir name
			
			sjoin_out_filename = f'{os.path.split(self.shp1)[1][:-4]}_unlabeled_all_revised.shp' #get the filename from the boundaries 
			sjoin_out_filename = os.path.join(self.output_dir,subset,sjoin_out_filename)

			#returns a shapefile which is the unlabeled polygons but now with the label of their parent glacier and a shapefile of all the unlabeled polygons
			return gdf,sjoin_out_filename,unlabeled

		except KeyError: 
			print('Processing general sjoin')
			sjoin_out_filename = os.path.join(self.output_dir,os.path.split(self.shp1)[1][:-4]+f'_sjoin_full_area_{self.year}_{self.join_type}_{self.op}.shp')
			return gdf, sjoin_out_filename

	def multiple_shp_sj(self,remove_feats): 
		"""Merge a list of shapefiles, this is not explicitly a spatial join but more of a merge. 
		This will take the biggest area at a subset border which is keeping artifacts as of 8/16/2021."""

		#try for multiple shapefiles
		try: 
			#concat a list of shapefiles (list is just paths)
			gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in self.file_list], 
	                        ignore_index=True), crs=gpd.read_file(self.file_list[0]).crs)
			#remove features that have an artifact from GlacierNet outputs 
			#first remove features from subset 9 
			gdf = gdf.loc[~((gdf['subset']=='subset_9') & (gdf['rgi_label'].isin(remove_feats)))]
			#then remove features from subset 12
			gdf = gdf.loc[~((gdf['subset']=='subset_12') & (gdf['rgi_label'].isin(remove_feats)))]
			gdf = gdf.loc[~((gdf['subset']=='subset_7') & (gdf['rgi_label'].isin(remove_feats)))]
			gdf = gdf.dissolve(by='rgi_label').reset_index()
			sjoin_out_filename=os.path.join(self.output_dir,f'{self.model}_{self.year}_dissolve_test.shp')

		except Exception as e: 
			print(e) 
		return gdf, sjoin_out_filename

	def attribute_join(self): 
		"""Perform an attribute (table) join based on a common column between two shapefiles."""
		#create the filename 
		subset = os.path.split(self.shp2)[0].split('/')[-1] #this is very clunky and hardcoded, basically we want the child dir name
		sjoin_out_filename = f'{os.path.split(self.shp2)[1][:-4]}_extended_labels.shp' #get the filename from the boundaries 
		sjoin_out_filename = os.path.join(self.output_dir,subset,sjoin_out_filename)

		if os.path.exists(sjoin_out_filename): 
			return None 
		
		#read in the shapefiles 
		try: 
			shp1_df = gpd.read_file(self.shp1).dropna()#.to_crs('EPSG:3338')
		except Exception as e: 
			# print('It looks like your primary input is already a geodataframe.')
			# check = input('Is that correct? (y/n)')
			# if check.lower()=='y': 
			shp1_df = self.shp1

			# else: 
			# 	print('Ok stopping...')
			# 	return None 

		shp2_df = gpd.read_file(self.shp2).dropna() #this should be the base shapefile that has the parent glaciers 

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
		select = gpd.GeoDataFrame(select,crs=self.epsg,geometry=select['geometry'])

		#dissolve anywhere that two glaciers are touching and the labels agree so there's just one polygon 
		select = select.dissolve(by='rgi_label').reset_index()

		#################################################################### added 5/21/2021
		#there is an issue where polygons that do not have a bit added get dropped from the output, fix that 
		#in this case shp2 is the original data 
		leftovers = shp2_df.loc[(shp2_df['rgi_label']!=-1)&
						(~shp2_df['rgi_label'].astype(str).isin(select['rgi_label'].astype(str)))]
		
		select = pd.concat([select,leftovers])
		select['subset'] = self.subset_id
		# #create the filename 
		# subset = os.path.split(self.shp2)[0].split('/')[-1] #this is very clunky and hardcoded, basically we want the child dir name
		
		# sjoin_out_filename = f'{os.path.split(self.shp2)[1][:-4]}_extended_labels.shp' #get the filename from the boundaries 
		# sjoin_out_filename = os.path.join(self.output_dir,subset,sjoin_out_filename)

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
	joined = Joins(input_shp,None,output_dir=output_dir)

	newly_labeled = joined.pairwise_sj() #returns a tuple that is (gdf,filename)

	joined.write_sj_to_file(newly_labeled[2],newly_labeled[1])


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
	#joined = Joins(newly_labeled[0],input_shp,output_dir=output_dir)
	joined = Joins(gpd.read_file(label_shp),input_shp,output_dir=output_dir,subset_id=subset_id)

	joined_out=joined.attribute_join()

	try: 
		joined.write_sj_to_file(joined_out[0],joined_out[1])
	except TypeError: 
		print('It looks like that df has no data')

def add_dir(output_dir,subset_num): 
	output_dir = os.path.join(output_dir,f'subset_{subset_num}')
	if not os.path.exists(output_dir): 
		os.mkdir(output_dir)

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
	# i) first run the spatial join to incorporate missing areas 
	#running
	# input_files = list(Path(input_dir).rglob('*1_min_size.shp')) #changed to just get the vectors with the highest level of processing

	# #input_files = [str(file) for file in input_files if 'subset_8' in str(file)] #just in for testing 
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
	# 	base_union=persistence.create_group_ids(shps,output_dir).set_crs('EPSG:3338')
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

	# 		output_file = os.path.join(child,f'07042021_model_{k}_{subset_id}_group_id_added.shp')

	# 		if not os.path.exists(output_file): 
	# 			print('Making file')
	# 			gdf.to_file(output_file)
	# 		else: 
	# 			print('That file already exists')

	##########################################################################
	# iii) finally make the final output with the attribute join and write to disk 
	#running
	#these should be the files that are the output of the previous vector processing step 
	# input_files = list(Path(input_dir).rglob('*0.01_min_size.shp')) #changed to just get the vectors with the highest level of processing
	# print(len(input_files))
 # 	#these are the files produced by step two above 
	# alt_unlabeled_vects = list(Path(kwargs.get('vect_dir')).rglob('*id_added.shp'))
	# print(len(alt_unlabeled_vects))

	# #generate outputs
	# #test run one file
	# #run_parallel(input_files[0],output_dir)

	# #run it all
	# pool = multiprocessing.Pool(processes=25)
	
	# label=partial(run_step2_parallel, output_dir=output_dir,alt_unlabeled_vects=alt_unlabeled_vects)
	
	# #run it 
	# result_list = pool.map(label, input_files)

	##########################################################################
	# iv) do a merge/join to combine all the subsets of a given year. These are the outputs of the attribute join above  
	input_files = list(Path(input_dir).rglob('*extended_labels.shp'))
	print(len(input_files))
	for year in range(1986,2022,2): 
		sjoin_files = [str(file) for file in input_files if str(year) in str(file)]
		print('year is: ',year)
		sjoin = Joins(None,None,file_list=sjoin_files,year=year,output_dir=output_dir,**kwargs)
		output = sjoin.multiple_shp_sj(kwargs.get('remove_feats'))
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
		
	remove_feats = [9044,22342,22341,22335,22349,9164,9421,10232,8115,13830,14216,22889,26715,19814]
	main(input_dir,output_dir,vect_dir=vect_dir,model=model,clip_bounds=clip_bounds,remove_feats=remove_feats)#,field='pixelvalue',model=model,left_shp=left_shp)#,right_shp=att_2)
