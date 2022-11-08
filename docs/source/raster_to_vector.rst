Raster to vector conversion and vector 
=======================================

The primary outputs of the GlacierCoverNet model are three class rasters, one for each of the two year composites in the time series (18 composites).
Each of these three class rasters has classes for supraglaical debris, non-debris covered glaciers and non-glacier cover. To be more useful to the NPS
partners, researchers and other end users, we convert these rasters to vectors to enable labeling and then tracking of individual glaciers over time. 
The naming structure for these glaciers is derived from existing naming structures/conventions in the RGI and GLIMS databases. There are a series of 
steps involved in first making this conversion then doing the labeling and dealing with some of the associated problems that arise. 