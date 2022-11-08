import pandas as pd 
import pymannkendall as mk
import os 
import sys 
import numpy as np 


def run_mk_test(data): 
	return mk.original_test(data)


if __name__ == '__main__':
	#main()
	output_dir = '/vol/v3/ben_ak/excel_files/areal_change_stats/'
	overall_area = pd.read_csv("/vol/v3/ben_ak/excel_files/areal_change_stats/overall_glacier_area_all_years_full_draft1.csv").iloc[:,1:]
	debris = pd.read_csv("/vol/v3/ben_ak/excel_files/areal_change_stats/debris_cover_glacier_area_all_years_full_draft1.csv").iloc[:,1:]

	overall_dict = {}
	debris_dict = {}
	for col in overall_area.columns: 
		overall_dict.update({col:[run_mk_test(overall_area[col])[0]]}) #will just put the increasing, decreasing or no trend in there 
		debris_dict.update({col:[run_mk_test(debris[col])[0]]}) #will just put the increasing, decreasing or no trend in there 

	#print(debris_dict)
	debris_out = pd.DataFrame.from_dict(debris_dict)
	overall_out = pd.DataFrame.from_dict(overall_dict)

	output = pd.concat([debris_out,overall_out])
	output.index = ['debris','overall']
	output.replace('increasing','+',inplace=True)
	output.replace('decreasing','-',inplace=True)
	output.replace('no trend',' ',inplace=True)
	
	output_fn = os.path.join(output_dir,'areal_change_mann_kendell_stats_recoded_draft1.csv')
	if not os.path.exists(output_fn): 
		output.to_csv(output_fn) 