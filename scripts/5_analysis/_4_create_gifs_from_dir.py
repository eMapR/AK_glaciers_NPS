import json 
import os
import glob
import sys 
from PIL import Image

class MakeGif(): 

	def __init__(self,image_dir,output_dir,focus_area,fp_out=None): 
		self.image_dir=image_dir
		self.output_dir=output_dir
		self.focus_area=focus_area
		self.fp_out = os.path.join(self.output_dir,f"{self.focus_area}.gif")

	def make_gif(self): 
		"""Create the actual gif and write to disk.
		Comes from: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
		"""
		if not self.image_dir.endswith('/'): 
			files = self.image_dir+"/*.png"
		else: 
			files = self.image_dir+"*.png"
		img, *imgs = [Image.open(f) for f in sorted(glob.glob(files))]
		img.save(fp=self.fp_out, format='GIF', append_images=imgs,
		         save_all=True, duration=1500, loop=0)
		return None 

	def run(self): 
		#check if the output dir exists and if not make it 
	    
		if not os.path.exists(self.fp_out): 
			self.make_gif()
		else: 
			overwrite = input(f'The gif file {self.fp_out} already exists, would you like to overwrite? (y/n)')
			overwrite = overwrite.lower().replace(' ','')
			if (overwrite=='y') or (overwrite=='yes'):
				self.make_gif()
			else: 
				return None  


def main(image_dir,output_directory): 
	subdirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]
	for subdir in subdirs: 
		focus_area = os.path.basename(os.path.normpath(subdir))
		print(f'Generating gif for {focus_area}')
		ins=MakeGif(subdir,output_directory,focus_area)
		ins.run()

if __name__ == '__main__':
	# params = sys.argv[1]
	# with open(str(params)) as f:
	# 	variables = json.load(f)
	input_dir = "/vol/v3/ben_ak/visualizations/final_outputs/pngs/nebesna/"
	output_dir = "/vol/v3/ben_ak/visualizations/final_outputs/gifs/"
	#focus_area = variables['focus_area']
	main(input_dir,output_dir)