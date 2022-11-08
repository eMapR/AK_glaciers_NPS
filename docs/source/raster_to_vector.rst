Raster to vector conversion and vector 
=======================================

The primary outputs of the GlacierCoverNet model are three class rasters, one for each of the two year composites in the time series (18 composites).
Each of these three class rasters has classes for supraglaical debris, non-debris covered glaciers and non-glacier cover. To be more useful to the NPS
partners, researchers and other end users, we convert these rasters to vectors to enable labeling and then tracking of individual glaciers over time. 
The naming structure for these glaciers is derived from existing naming structures/conventions in the RGI and GLIMS databases. There are a series of 
steps involved in first making this conversion then doing the labeling and dealing with some of the associated problems that arise. 

Required param file: `GitHub <https://github.com/eMapR/AK_glaciers_NPS/blob/main/params/GlacierNet_post_processing.txt>`_    

Scripts: 
_1_assign_glacier_labels.py
_2_convert_raster_to_vector.py
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

================
Code description
=================
   
**_1_assign_glacier_labels.py**   

1. Get outputs of GlacierNet model which will be organized as 18 two-year composites with the label year being the second of the two years (e.g. the 1986 
composite includes data from 1985 and 1986). The data will be organized by overlapping subsets of the overall study area. These are used to increase efficiency and decrease processing time. 
.. image:: /imgs/image_areas_fig.jpg
2. Transfer glacier IDs from RGI - 
Required script(s) and param files: 
 
* Make a raster of RGI IDs which matches the extent of the GlacierNet raster you want to label   
* *Note that this has to be just a simple numeric code- the generic RGI id is a mix of string and int so a unique numeric code field has to be created*    
* Basically the script uses an iterative process where it grows the existing glacier labels using a 3x3 max matrix and re-labels pixels that are subsumed by that growth. Predictably, this process reaches a point of diminishing returns as it gets the majority of pixels in the first 5-10 iterations.    

.. image:: /imgs/asymptote.jpg

3. Process is set up to stop when there is less than a 1% change in the percent of pixels that move from unlabeled to labeled between iterations
*Note that you need to change the binary_val arg if you have more than two classes you are concerned with (e.g. not glacier and glacier). 
This is set up so that we output one set of maps for debris cover and a separate set of maps for debris free glacier*

Functions/classes

LabelSupport - (class) makes sure inputs (specific glaciers) have matching dimensions and creates the max filter function. 
LabelRaster - (class) does the actual pixel labeling with the naming approach outlined above and in the manuscript 
run_parallel - simple helper function to implement in parallel 

**2_convert_raster_to_vector.py**

Basically does what the script name infers.   
*Notes* 

This process could take 8-10 hours
This script hits an error at the end of the script when it tries to pickle something to disk? This does not seem to do anything to the outputs but rather happens at the end of the processing    
This script will output subdirs like the GlacierNet subset outputs   

Functions 
gdal_raster_to_vector - implements a version of GDALs `Polygonize <https://gdal.org/programs/gdal_polygonize.html>`_ function. 







