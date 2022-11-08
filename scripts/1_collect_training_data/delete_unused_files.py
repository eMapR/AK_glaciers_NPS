import os 
import glob
import sys
import json

def remove_unwanted_files(parent_directory,modifier,file_ext): 
	'''
	Helper function to remove files from subdirectories.
	'''
	print(f'We are going to delete files which contain the value {modifier}.')
	double_check = input('Should we proceed? (y/n)')
	if (double_check.lower() == 'y') or (double_check.lower()=='yes'): 
		for filename in glob.iglob(parent_directory+f'/**/*{modifier}*',#.{file_ext}', 
		                           recursive = True): 
			
			if modifier in filename: 
				if 'revised' in os.path.split(filename)[1]: 
					pass
				else: 	
					print(filename)
					os.remove(filename) 
			else: 
				continue
	else: 
		print('You either specified you did not want to delete files or there were no files to delete. Finishing.')
		print('Done deleting files...')

def main(): 
	# params = sys.argv[1]
	# with open(str(params)) as f:
	# 	variables = json.load(f)
	# 	input_dir = variables['input_dir']
	# 	image_tiles_dir = variables["image_tiles_dir"]
	# 	topo_tiles_dir = variables["topo_tiles_dir"]
	# 	vrt_dir = variables["vrt_dir"]
	# 	train_class = variables["train_class"]
	# 	test_dir = variables["test_dir"]
		
	remove_unwanted_files("/vol/v3/ben_ak/vector_files/neural_net_data/outputs/08122021_model_run/debris_free/",'min_size','tif')
if __name__ == '__main__':

    main()