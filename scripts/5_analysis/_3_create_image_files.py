import glob
import os 
import sys
import matplotlib.pyplot as plt 
import pandas as pd 
import geopandas as gpd 
import re
import rasterio
import rasterio.plot
import json

def create_pngs(input_shp,input_raster,output_dir,year,band,bounds,year_field='year',**kwargs): 
    """Create a matplotlib figure with multiple layers and save as a PNG file.
    One file per multi-annual composite and these serve as inputs to make gifs for change over time. 
    """
    #check if the output dir exists and if not make it 
    if 'focus_area' in kwargs: 
        source_file = kwargs.get('focus_area')
    else: 
        source_file = (os.path.split(input_shp)[1][:-4]).split(year)[1] #hardcoded for file naming convention
    
    dir_name = os.path.join(output_dir,source_file)
    #print(dir_name)
    if not os.path.exists(dir_name):
        print('making dir ',dir_name) 
        os.mkdir(dir_name)
    #get the shapefile
    gdf = gpd.read_file(input_shp)
    #get the bounds of the first year and use that to contstrain all of the other years 
    aoi_bounds = gpd.read_file(bounds).geometry.total_bounds
    #print(aoi_bounds)
    xmin, ymin, xmax, ymax = aoi_bounds
    #get the raster 
    raster = rasterio.open(input_raster,'r')
    #arr = raster.read(band)
    fig, ax = plt.subplots(figsize=(15, 15))
    
    #set the aoi to the clipped shapefile 
    ax.set_xlim(xmin+1,xmax+2000) #changed 5/17/2021 to add one to each val
    ax.set_ylim(ymin-1000,ymax) 

    #add the base raster 
    #rasterio.plot.show((raster,band), ax=ax,cmap='Greys_r',vmin=0,vmax=8000) #changed from 15000 5/17/2021
    
    #add any ancillary data (e.g. other boundaries)
    gdf.plot(color ='#a5b4c2', linewidth=1, edgecolor='black',ax=ax,alpha=0.5)
    if 'ancillary_vect' in kwargs: 
        #changed 1/28/2022 for a dir instead of one file
        other_files = glob.glob(kwargs.get('ancillary_vect')+'*.shp')
        #other_files = gpd.read_file(kwargs.get('ancillary_vect'))
        try:  
            #other_df = other_df[(other_df[year_field]==year) | (other_df[year_field]==str(year))]
            other_df = [f for f in other_files if str(year) in os.path.split(f)[1]][0]
            other_df = gpd.read_file(other_df)
        except KeyError as e: 
            year_field = input('The year field does not exist. If there is another field to use for year please enter it: ')
            try: 
                other_df = other_df[other_df[year_field]==year]
            except KeyError as e: 
                print('Sorry that did not work. Passing on adding an extra boundary to the figure.')
        #changed these 1/28/2022 to make them filled in 
        other_df.plot(color='#683f00',linewidth=1,edgecolor='black',ax=ax,alpha=0.6)
    
    #gdf.plot(color ='#cfe2f3', linewidth=1, edgecolor='black',ax=ax)

    #add the RGI for reference 
    if 'rgi' in kwargs: 
        rgi_df = gpd.read_file(kwargs.get('rgi'))
        rgi_df.boundary.plot(linewidth=2, edgecolor='darkblue',ax=ax)

    ax.annotate(year,xy=(0.85,0.05),
                xycoords='axes fraction',
                fontsize=24, 
                bbox=dict(boxstyle="square,pad=0.3", fc="#a6a6a6", ec="black", lw=1)
                ) #moved 5/17/2021
    # plt.show()
    # plt.close('all')

    # this will save the figure as a high-res png in the output path. you can also save as svg if you prefer.
    output_filepath = os.path.join(dir_name, year+'_w_rgi.png')
    if not os.path.exists(output_filepath): 
        fig.savefig(output_filepath, dpi=500)
        return 'done'
    else: 
        print(f'The file {output_filepath} already exists,')
        overwrite = input('Would you like to overwrite? (y/n)')
        if (overwrite.lower() == 'y') | (overwrite.lower() == 'yes'): 
            fig.savefig(output_filepath, dpi=500)
            return 'done'
        else: 
            return None 


def main(input_dir,input_raster,output_dir,**kwargs): 
    
    files = sorted(glob.glob(input_dir+'*.shp'))
    #files = sorted([file for file in files if 'buffer' in file])
    count = 1 
    for file in files: 
        year = re.findall('(\d{4})', os.path.split(file)[1])[0] #gets a list with the start and end of the water year, take the second one. expects files to be formatted a specific way from GEE 
        print(f'year is: {year}')
        check=create_pngs(file,input_raster,output_dir,year,count,bounds=files[0],year_field='GN_comp_ye',**kwargs)
        #in an instance where overwrite is not given exit the for loop 
        if not check: 
            break 
        count +=1 
if __name__ == '__main__':
    params = sys.argv[1]
    with open(str(params)) as f:
        variables = json.load(f)
        input_dir = variables['input_dir']
        base_raster = variables['base_raster']
        rgi = variables['rgi']
        output_dir = variables['output_dir']
        focus_area = variables['focus_area']
        ancillary_vect = variables['ancillary_vect']

    main(input_dir,base_raster,output_dir,rgi=rgi,focus_area=focus_area,ancillary_vect=ancillary_vect)


#     import fiona
# import rasterio
# import rasterio.plot
# import matplotlib as mpl
# from descartes import PolygonPatch

# src = rasterio.open("tests/data/RGB.byte.tif")

# with fiona.open("tests/data/box.shp", "r") as shapefile:
#     features = [feature["geometry"] for feature in shapefile]

# rasterio.plot.show((src, 1))
# ax = mpl.pyplot.gca()

# patches = [PolygonPatch(feature) for feature in features]
# ax.add_collection(mpl.collections.PatchCollection(patches))


# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# fig = plt.figure() 
# # create figure window

# gs = gridspec.GridSpec(a, b)
# # Creates grid 'gs' of a rows and b columns 


# ax = plt.subplot(gs[x, y])
# # Adds subplot 'ax' in grid 'gs' at position [x,y]


# ax.set_ylabel('Foo') #Add y-axis label 'Foo' to graph 'ax' (xlabel for x-axis)


# fig.add_subplot(ax) #add 'ax' to figure