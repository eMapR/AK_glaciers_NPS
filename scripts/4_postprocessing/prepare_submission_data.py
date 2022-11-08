import os 
import sys
import geopandas as gpd
import pandas as pd 
import glob 
import fiona 
from shapely.geometry import shape
from osgeo import ogr 
import fiona
from shapely.geometry import Polygon, shape, mapping

# overall = "/vol/v3/ben_ak/vector_files/neural_net_data/NSIDC_dataset_submission/final_dataset/overall_2020.shp"
# debris = "/vol/v3/ben_ak/vector_files/neural_net_data/NSIDC_dataset_submission/final_dataset/debris_2020.shp"

# fn = "/vol/v3/ben_ak/vector_files/neural_net_data/NSIDC_dataset_submission/final_dataset/2020_gpd_test1.shp"

# poly1 = fiona.open(overall)
# poly2 = fiona.open(debris)
# # transformation of geometry to shapely geometry (one element )
# geom1 = [shape(feat['geometry']) for feat in poly1][0]
# geom2 = [shape(feat['geometry']) for feat in poly2][0]
# # creation of the resulting shapefile
# with fiona.open(fn, 'w',driver='ESRI Shapefile', schema=poly1.schema) as output:
#    prop = {'id': 1}
#    output.write({'geometry': mapping(geom1.difference(geom2)), 'properties':prop})
# #out = ogr.Difference(overall,debris)#overall.difference(debris, align=False)
# #out.to_file(fn)


# overall = gpd.read_file("/vol/v3/ben_ak/vector_files/neural_net_data/NSIDC_dataset_submission/final_dataset/overall/AK_2018_overall_glacier_covered_area_fixed.shp")

# error = overall.loc[overall['RGIId']=='RGI60-01.11624']['geometry']
# print(error)
# good = overall.loc[overall['RGIId']=='RGI60-01.03865']['geometry']
# print(good)

# print(overall.head())


files = glob.glob("/vol/v3/ben_ak/vector_files/neural_net_data/NSIDC_dataset_submission/final_dataset/AK_glaciers_G10040-V001/debris_free_area/*") 

print(files)

count = 0 
for file in files:
#    if file.endswith('.qpj'): 
#       os.remove(file)
#       count += 1 
#    else: 
#       pass
# print(f'Finished and removed {count} files')
   # start = file[:-4]
   # ext = file[-4:]
   # insert = '_area'
   # new_fn = start + insert + ext
   # print(start)
   # print(ext)
   # print(new_fn)
   new_fn = file.replace('_buffer','')
   os.rename(file,new_fn) 
   # print('file renamed!')

