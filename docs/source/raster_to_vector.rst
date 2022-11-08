Converting GlacierCoverNet outputs to vector 
============================================

The default outputs of the GlacierCoverNet model are three class rasters, one for each two year composite in the time series. However, to be useful
to researchers and the NPS these need to be in vector format. By converting to vector format we can track individual glaciers and how they are 
changing over time. To do this, we first convert rasters to vectors and then apply a series of steps that deal with labeling the individual glaciers 
based on existing norms/structures from the Randolph Glacier Inventory (RGI). 

