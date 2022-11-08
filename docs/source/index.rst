

Documentation for mapping of AK glaciers using DL
=================================================

This project aims to create composite raster images and vector datasets for the period 1984-2020 for all of the glaciated area in Alasaka. We use images from the Landsat 
archive and we use the `Landtrendr <https://emapr.github.io/LT-GEE/>`_ algorithm to create a spectrally homogeous time series of optical data. We combine spectral indices 
from that time series with topographic indices calculated from the `Alaska IfSAR 5m DEM <https://elevation.alaska.gov/>`_ in a convolutional neural network (CNN). We use 
the CNN to make predictions about glacier and debris covered glacier areas across the state of Alaska for bi or tri annual composites.   

Dependencies:   
	*numpy  
	*pandas  
	*geopandas  
	*`pytorch <https://pytorch.org/>`_  
	*multiprocessing  
	*subprocess  

.. toctree::
   :maxdepth: 2
   :caption: Script examples:
   Create training dataset <training_data.rst>
   GlacierCoverNet <modeling.rst>
   Convert raster to vector <raster_to_vector.rst>
   Postprocess vectors <vector_processing.rst>
   Create figures <figures.rst>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
