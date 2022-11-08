Create GlacierCoverNet Training Dataset
=======================================

GlacierCoverNet is a type of Convolutional Neural Network (CNN) and as such, requires a large training dataset to 'learn' about its classification target.
To build the training dataset for this model we created a decision-tree based classification of debris free glacier using the Spatiotemporal 
Exploratory Model (STEM) (`Hooper and Kennedy, 2018 <https://doi.org/10.1016/j.rse.2018.03.032>`_). We then differenced that with the existing Randolph Glacier Inventory (RGI)
6.0 dataset as has been done by a variety of authors in the past. This difference area was considered supraglacial debris for the purposes of this model. 
We created a layer that covered the full study area (state of Alaska) and then systematically subsampled that area into hundreds of thousands of image chips. 
For more details on the process see the associated manuscript. 

Code location: `GitHub <https://github.com/eMapR/AK_glaciers_NPS/tree/main/scripts/1_collect_training_data>`_

Scripts:   
* _1_stack_predictors_to_multiband.py 
* _2_prep_train_locations.py  
* _3_fix_no_data_areas_in_partitions_rev.py   
* _4_extract_image_chips_moving_window.py   
   
* _4_remove_chips_without_enough_important_class.py   

================
Code description
================

**_1_stack_predictors_to_multiband**  

This is the main working script to generate the data needed for training data selection. These will be multiband rasters that cover the full study area.
Collect all of your data for your predictor variables for the study area you want to predict. In our case it is:   

* Five bands of optical data  
* Four bands of topographic data  
* The class label   

**10 bands total**

*NOTE data needs to be in the same dtype (i.e. int16, float32 etc.) if some of your data are in float and some in int it might make sense to 
scale your float (multiply by 100, 1000 etc.) and then convert to int. This will save a lot of space in the long term.*

**General Steps**

1. Convert tif files in subdirectories of optical data for each year and band (60 files for 2000-2020)- this was predicated on the way data were stored on a local directory 
and they were originally organized this way due to the use for STEM (above). This could vary on another system. 
2. Convert topo data to int and then stack/mosaic to make a big vrt file for topo data (one band per predictor)
3. Right now topo and class label were being stored in directories that mimic the directory structure for the optical data but if the data are not being used for STEM this is not necessary. 
It would be easier to just have a tif of the study area and make that into a vrt 
4. Make a large vrt for the study area of the class label layer 
5. Make a large vrt for the study area of the distance raster (if using)
6. Combine all of these big vrts into one 10 (or 11 with the distance raster) band vrt file for each yearly composite 

*NOTE that these things are all being done twice. Due to modeling paramaters, training data constraints and biophysical differences, the project
treated northern Alaska glaciers (Brooks Range) differently than those in proximity to the Gulf of Alaska (southern region).* 

Functions: 

* run_cmd - helper for parallelization   
* flatten_lists - flatten some lists   
* write_list_to_txt_file - as it sounds 
* reordering - as it sounds   
* get_image_tiles_from_subdirectories - used to get files from a specific directory structure that was specific to what was required to run the STEM model.    
* get_image_files - collect and organize files in a similar fashion to the previous function. This is also specific to the way files were stored for STEM.
* make_singleband_vrts - precisely what it sounds like    
* make_multiband_rasters - actually outputs a vrt file but with multiple bands.    

**_2a_convert_data_types**    

This script is used for pre-processing of data before building the VRTs (outlined in previous script). This script came about because the 
topographic data was by default in format Float32 and the optical data was Int16. We convert the optical data to Float32 to keep the 
decimals in the topographic data (this could be changed). It depends on the _2_stack_predictors_to_multiband script above. 
Script is setup to take a directory with subdirectories and make a copy of every file that is not negated with the specified modifier.    

Functions:    
* Convert_data_type- takes a list of paths, the specified band number you want to convert, a modifier that will be added to the output file name and a negate argument which allows you to pass files   
* Translate_data_type- same thing as above but for multiband raster- note that this will not work with the gdal buildvrt function if separate is specified as true.    
* Tif_to_vrt- what it sounds like, convert a tif to a vrt file    

**_2_prep_train_locations**
Creates a vrt file for every tile in the study area that is specified in the grid shapefile arg. Requires as input the full study area vrt 
created as the final output of _2_stack_predictors_to_multiband (above) and a grid of tiles. These tiles are created using three GEE scripts:     
https://code.earthengine.google.com/549d99456cfb28ddaf546e3eeee46a05    
https://code.earthengine.google.com/a00ddc82e138250108b5ae51ef2d67b1    
https://code.earthengine.google.com/601ba51aa58a6b3b296852214975a0c6    

These will create and compare grid sizes from 128x128 pixels to 512x512 pixels. Currently using 256 as we find that there is not a huge 
difference and these were optimized for GPU use. 
*NOTE that this script has to be run twice, once for the train set and once for the DEV set. These grids output two shapefiles (test,train) 
and you need to add bounding box cols to the shapefiles in QGIS (or elsewhere). 
Its easy to do with: https://gis.stackexchange.com/questions/79248/calculate-bounding-box-coordinates-of-a-selected-polygon-with-qgis*

*NOTES: This was revised before the April 2021 training data generation so that it selects data from different years of imagery/predictors. If you run it in this mode you have 
to supply a grid shapefile in the script that is a spatial join of the RGI boundaries and the train/test grid. This is important because it 
inherits the year of the RGI from that shapefile and assigns it to a tile so the program knows where to pull imagery from for partition 
creation. Make sure you are running everything for the southern or northern region. This means you have to change inputs and outputs in the param file before generating partitions.* 

Functions:  

* run_cmd - helper function for parallelization.    
* make_single_clip_command - clip a chip out of a larger raster.    
* get_geom_dict - prepare a geopandas df for processing.    
* get_partition_year - function used to figure out what year in the imagery should be used to match with the RGI stated year.    
* create_grid_image_cmds - set up a parallel version of the clipping functions to create the dataset iteratively.   

**_3_fix_no_data_areas_in_partitions_rev**  
This checks the outputs of the previous script to make sure there are no noData issues that are going to cause issues in modeling down the line. 
It is possible this script is deprecated depening on the approach you take.    

General steps and notes:   

1. Run after you generate the partitions with the previous script. This one takes a vrt and makes sure all of the noData areas are homogenous 
and correct. It uses a masking strategy that is based on the band numbers/structure and the noData values in current data. These values are adjustable but have defaults so it will run that way if its not changed. 
2. Script outputs adjusted tifs in the same directory as the VRT files that were generated by the previous script. 
3. This has to be run twice (once for the train set and once for the dev set) 
4. These are the inputs for the next script below which extracts training data chips 

Functions:   

* cast_dtype - as it sounds   
* make_masks - as it sounds   
* read_raster_to_np_and_write_to_tif - do the actual work of masking or changing the mask on the no data areas. 

**_4_extract_image_chips_moving_window**    

Create the actual image chips (128x128 pixels) from the previous 256x256 partitions. You just need to make sure you change the script to run for the appropriate region and train/dev set.    

Functions:   

* run_cmd - parallelization function
* extract_chips - this does the actual extraction, it walks across a partition by a given step and extracts an image chip. 
* run_moving_window_for_one_partition - implements the previous function 
* log_metadata - as it sounds 


