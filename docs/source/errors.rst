Methodology and code for conducting GlacierCoverNet error analysis 
===================================================================

The error analysis for GlacierCoverNet outputs was multifacited and leveraged a group of existing datasets to characterize the uncertainty in all of the
areas of the output products. For more details on the process see the manuscript where the logic, products and results were laid out in detail.
Much of the error analysis was just based on specific layers so a lot of it was conducted in QGIS and therefore does not have associated python scripts that can be elucidated. 

Calculation of overall glacier covered area: 
Comparison of RGI and GlacierNet for 2010 this has to be done somewhat manually as everything needs to be clipped to the 20 percent test 
grid that was generated when we were making the train/test set. This is largely done in QGIS and is specific to the version of the dataset you are using. 

*Notes:*    
* In the original version of the paper we used 2008 and 2010. This will only include 2010 by default.   
* In the original version we removed some partitions from the border with AK because the classification was bleeding over the border and needed to be constrained somehow. 
* Currently (8/24/2021) set up to do clipping and pre-processing in QGIS. The following are required: 
    * 2010 combined GlacierNet output clipped to northern and southern test sets   
    * 2016 debris cover GlacierNet output clipped to southern region test    
    * Scherler debris cover clipped to southern region test set    
    * Herreid debris cover clipped to southern region test set    

**calculate_mk_stats_areal_change**
This thing requires that you update the file paths and then you can calculate updated area columns and generate a csv or fill one in 
manually. This also calculates the Mann-Kendell trend for the data. 

Functions:   

* run_mk_test - runs the Mann Kendell test 





