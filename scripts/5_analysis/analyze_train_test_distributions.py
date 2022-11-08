import os 
import sys 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import json 
import glob 
from functools import reduce

def file_list_to_df_list(input_list,col_of_interest): 
	"""
	Helper function. 
	"""
	if col_of_interest: 
		output_list = [pd.read_csv(df)[col_of_interest] for df in input_list]
	else: 
		output_list = [pd.read_csv(df) for df in input_list]

	return output_list


def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)	
		#construct variables from param file
		csv_dir = variables['csv_dir'] 
		split = variables['split']
		output_dir = variables['output_dir']
		buff_type = variables['buff_type']
		train_files = glob.glob(csv_dir+f"*train*{buff_type}*")

		test_files = glob.glob(csv_dir+f"*test*{buff_type}*")
		#print(train_files)
		#print(test_files)
		#organize data 
		#first get train data
		tile_128_train=pd.melt(pd.concat(file_list_to_df_list([file for file in train_files if '128' in file],'percentage'),axis=1)) #these are all organized to go from file list to df list to concat df of all dfs to melted df 
		tile_256_train=pd.melt(pd.concat(file_list_to_df_list([file for file in train_files if '256' in file],'percentage'),axis=1))
		tile_512_train=pd.melt(pd.concat(file_list_to_df_list([file for file in train_files if '512' in file],'percentage'),axis=1))

		#then get test data 
		tile_128_test=pd.melt(pd.concat(file_list_to_df_list([file for file in test_files if '128' in file],'percentage'),axis=1))
		tile_256_test=pd.melt(pd.concat(file_list_to_df_list([file for file in test_files if '256' in file],'percentage'),axis=1))
		tile_512_test=pd.melt(pd.concat(file_list_to_df_list([file for file in test_files if '512' in file],'percentage'),axis=1))

		train_dfs=[tile_128_train,tile_256_train,tile_512_train]
		test_dfs=[tile_128_test,tile_256_test,tile_512_test]
		titles = ['128x128','256x256','512x512']
		ylabels= ['Train','Test']

		# font = {'family' : 'normal',
  #       'weight' : 'bold',
  #       'size'   : 18}

		# plt.rc('font', **font)
		
		fig,axes = plt.subplots(nrows=2,ncols=3,sharex=True,sharey='row',figsize=(10,10))

		for row in range(2): 
			for col in range(3): 
				if row < 1: 
					print('train')
					#print(type(train_dfs[col].mean().get(0)))
					train_dfs[col].hist(ax=axes[row][col],color='darkblue',bins=100)
					axes[row][col].set_title(titles[col])
					axes[row][col].set_ylabel(ylabels[row])
					axes[row][col].annotate(f'Mean:{round(train_dfs[col].mean().get(0),3)} \n Variance: {round(train_dfs[col].var().get(0),3)} \n STD: {round(train_dfs[col].std().get(0),3)}',xy=(0.7,0.7),xycoords='axes fraction')
				else: 
					print('test')
					test_dfs[col].hist(ax=axes[row][col],color='darkblue',bins=100)
					axes[row][col].set_title(' ')
					axes[row][col].set_ylabel(ylabels[row])
					axes[row][col].annotate(f'Mean:{round(test_dfs[col].mean().get(0),3)} \n Variance: {round(test_dfs[col].var().get(0),3)} \n STD: {round(test_dfs[col].std().get(0),3)}',xy=(0.7,0.7),xycoords='axes fraction')
		axes[1][1].set_xlabel('Tile percent glacier')

		#plt.savefig(os.path.join(output_dir,f'split_{split}_{buff_type}_buff_type.png'))
		plt.tight_layout()
		plt.show()
		plt.close('all')

if __name__ == '__main__':
	main()