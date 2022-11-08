from osgeo import ogr
import os
import sys 
import json 
import glob
from pathlib import Path

def merge_shapefiles(input_files,output_fn): 
    #set some local variables 
    file_ext = '.shp'
    drvrname = 'ESRI Shapefile'
    geomtype = ogr.wkbMultiPolygon
    outputdriver = ogr.GetDriverByName('ESRI Shapefile')
    out_ds = outputdriver.CreateDataSource(output_fn) 
    out_layer = out_ds.CreateLayer(output_fn, geom_type = geomtype)
    
    #filelist = os.listdir(directory)
    
    for file in input_files:
        print(f'processing {file}')
        ds = ogr.Open(file)
        if ds is None:
            print("ds is empty")
        lyr = ds.GetLayer()
        for feat in lyr:
            out_feat = ogr.Feature(out_layer.GetLayerDefn())
            out_feat.SetGeometry(feat.GetGeometryRef().Clone())
            out_layer.CreateFeature(out_feat)
            out_layer.SyncToDisk()

def main(input_dir,output_dir,model):
    # outputMergefn = 'merged_features.shp'
    # directory = "./"
    # file_start = 'tmp'
    input_files = list(Path(input_dir).rglob('*extended_labels.shp'))
    for year in range(1986,2022,2): 
        merge_files = [str(file) for file in input_files if str(year) in str(file)]
        output_fn = os.path.join(output_dir,f'{model}_{year}_merged.shp')
        if not os.path.exists(output_fn): 
            merge_shapefiles(merge_files,output_fn)

if __name__ == '__main__':
    params = sys.argv[1]
    with open(str(params)) as f:
        variables = json.load(f)
        input_dir = variables['input_dir']
        output_dir = variables['output_dir']
        model_version = variables['model_version']
        main(input_dir,output_dir,model_version) 