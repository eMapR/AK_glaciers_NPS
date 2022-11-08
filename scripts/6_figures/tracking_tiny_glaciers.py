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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


classified=gpd.read_file('/vol/v3/ben_ak/vector_files/neural_net_data/outputs/03272021_model_run/labeled_subsets_basic/03272021_1990_dissolved.shp')
rgi=gpd.read_file('/vol/v3/ben_ak/vector_files/glacier_outlines/rgi_epsg_3338_clipped_to_US_southern_region.shp')

missing = rgi.loc[~rgi['rgi_id'].isin(classified['rgi_label'])]

missing['area'] = missing['geometry'].area / 10**6 #this assumes that the field that is being created is in meters and you want to change to km2

print(missing['area'].sum())
print(missing['area'].mean())
print(missing['area'].max())
#gdf = gdf.loc[gdf['area']>=min_size] 

#clipped = gpd.clip(classified,missing)
output_fn = os.path.join('/vol/v3/ben_ak/vector_files/neural_net_data/testing/missing_glaciers/','missing_rgi_labels.shp')

if not os.path.exists: 
	missing.to_file(output_fn)

#clipped.read_file(os.path.join('/vol/v3/ben_ak/vector_files/neural_net_data/testing/missing_glaciers/','2010_classification_areas_where_labels_are_missing.shp'))
print(missing.shape)

print(classified.columns)
print(rgi.columns)

