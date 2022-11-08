Post processing of glacier vectors
==================================

After converting the GlacierCoverNet rasters to vectors we have to do some post processing to remove background values/shps and then do the labeling. 

Required param file: `GitHub <https://github.com/eMapR/AK_glaciers_NPS/blob/main/params/GlacierNet_post_processing.txt>`_    

Scripts: 
_3_vector_post_processing.py
_4_add_labels_to_unlabeled_polygons.py
_5_clean_attribute_tables.py

Param file args:   

* Input_dir- this is different depending on the step below but should be whatever directory of subsets you are working with   
* Output_dir- also changes depending on the step below. In general, scripts should be set up to make subset subdirs in cases where that should happen   
* Model_version- some of the scripts use this to append the model version to the filename   
* Classified_rasters- used in the first script as the dir where the GlacierNet outputs reside   
* Id_rasters- these are the rasterized RGI with pixel values the RGI label as an int. This is used in the first step below.   
* Binary_val is for the first script and denotes the class you are interested in. Setting binary_val to a number larger than 10 will select all areas with a class higher than zero. This will produce a map of all glacier covered areas irrespective of supraglacial debris. In the final version of GlacierNet there are three output classes:   
    * 0- no glacier   
    * 1- debris covered glacier   
    * 2- debris free glacier   

==================
Code description
=================

**_3_vector_post_processing**    

Clean up the vector outputs. 

1. Remove the background data- when the data are converted from raster to vector they retain the 0 (no glacier) area in the background. This has to be removed. 
2. Buffer- there is probably a better way to do this but this is the fastest way to deal with topological errors that arise from the conversion. Buffer with a distance of zero. 
*NOTE that in some cases this seems to leave some errors, particularly self intersection errors* 
3. Add an area column and then remove any individual glacier entity smaller than 0.01 km2 as per the RGI. This can be amended to 1km as per Herreid et al and other examples by adjusting the function argument.

**_4_add_labels_to_unlabeled_polygons**     

Deal with glaciers that remain unlabeled. These are generally vectors that touch labeled glaciers but do not themselves have a label.
This script is done in four steps which as of 8/15/2021 are commented/uncommented in the main function.    

1. Doing a spatial join using touches arg to get the RGI label attached to the small bit    
2. Selecting all the bits that are -1 (no label) and then doing an attribute join with the parent glacier vector. Adding a persistence label
and group to the data- this is not relevant as of 8/15/2021 but has not yet been removed because a fairly significant rewrite is required to make that work    
3. Combining labeled bits with the outputs of the GlacierNet model    
4. Merge all of the subset areas to make one map for the state for every biannual composite in the collection     

*Notes:* 

1. In the attribute join function we merge shapes with the same RGI id and then dissolve them so theres just one feature   
2. These may need to be buffered again or have something else done to deal with topological errors    
3. This script must be run in a conda env with the newest version of Geopandas otherwise youll get an error that touches is not a legit command for op   
4. It is necessary to run this process three times to cover the full dataset:   
    * Combined supraglacial debris/clean ice  
    * Debris-free ice   
    * Supraglacial debris   

Functions: 
pairwise_sj - does a spatial join between GlacierCoverNet outputs and the RGI glaciers
attribute_join - migrates RGI metadata over to the GlacierCoverNet data
multipl_shp_sj - Merge a list of shapefiles, this is not explicitly a spatial join but more of a merge. This will take the biggest area at a subset border which is keeping artifacts as of 8/16/2021.
write_sj_to_file - what it sounds like 
run_step1_parallel - helper function to do the spatial joins
run_step2_parellel - deal with joining
add_dir - checks if output dir exists and if not it creates it 

**5_clean_attribute_tables**

Move the RGI metadata over to the outputs with a table join on the RGI ID and remove erroneous columns 

1. Takes a combined file from northern and southern processing regions from the RGI and does a table join with all of the GlacierCoverNet outputs so they inherit all of the information from the RGI shapefiles
2. Cols to add or drop are hardcoded in this script as of 8/15/2021 and should therefore be adjusted accordingly to make changes

Functions: 
calculate_area_col - as it sounds, adds an area column to new shapefiles 


