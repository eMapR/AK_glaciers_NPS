import os
import sys 
from osgeo import gdal 
import json
import subprocess
import multiprocessing
import time 
import glob 

def run_cmd(cmd):
  print(cmd)  
  return subprocess.call(cmd, shell=True)

def make_clip_commands(file_dir): 
	"""Convert a directory of vrts to tifs."""
	cmd_list = []
	for file in glob.glob(file_dir+'*.vrt'): 
		output_filename = file[:-4]+'.tif'
		cmd = 'gdal_translate '+file+' '+output_filename
		cmd_list.append(cmd)

	return cmd_list
def convert_single_file(input_vrt,output_tif): 
	"""Convert single vrt to tif."""
	if not os.path.exists(output_tif): 
		subprocess.call('gdal_translate '+input_vrt+' '+output_tif,shell=True)
	else: 
		return None

def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		test_dir = variables['test_dir']
		
		# run the image creation commands in parallel 
		start_time = time.time()
		pool = multiprocessing.Pool(processes=20)
		pool.map(run_cmd, make_clip_commands(test_dir)) 
		pool.close()

		print(f'Time elapsed for image chip extraction is: {((time.time() - start_time))/60} minutes')
		
		
if __name__ == '__main__':
    main()