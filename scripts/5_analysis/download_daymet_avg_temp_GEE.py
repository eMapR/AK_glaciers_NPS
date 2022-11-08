import ee

#deal with authentication
try: 
	ee.Initialize()
except Exception as e: 
	print('Not authenticated')
	authenticate = ee.Authenticate()
	ee.Initialize()

class GetDaymet(): 
	"""Define a class for getting the Daymet imageCollection and do some formatting."""
	def __init__(self, aoi, year, isotherm):  
		self.aoi=aoi
		self.year = year
		self.isotherm = isotherm

	def get_data(self): 
		daymet_ic = (ee.ImageCollection("NASA/ORNL/DAYMET_V4").filterBounds(self.aoi)
															  .filterDate(f'{self.year}-10-01', f'{self.year+1}-09-01'))
		return daymet_ic

	def average_temp(self,img): 
		"""Take the average of tmax and tmin."""
		tavg = (img.select('tmax').add(img.select('tmin'))).divide(ee.Image(2))#daymet_ic.map(lambda img: ((img.select('tmax').add(img.select('tmin'))).divide(ee.Image(2))))

		return img.addBands(ee.Image(tavg)) 


	def get_ics(self): 
		"""Create a one year imageCollection, add the avg temp band and return a composite."""
		ic = self.get_data()
		#add a tavg band to every day in the year of interest  
		output = ic.map(lambda img: self.average_temp(img))
		#get a list of the band names in an image of the tavg ic
		bands=output.first().bandNames()
		
		try: 
			if bands.contains('tmin_1').getInfo(): 
				output=output.select(bands,bands.replace('tmin_1','tavg'))
			elif bands.contains('tmax_1').getInfo(): 
				output=output.select(bands,bands.replace('tmax_1','tavg'))
		except KeyError as e: 
			print('The default col tmin_1 does not exist.')
			print(f'The bands in the tavg ic look like: {bands}')

		return ee.Image(output.mean()).set('year',str(self.year))

	def create_isotherms(self): 
		"""Take an input avg temp image and get above and below 0."""
		year_img = self.get_ics()
		if (self.isotherm.lower() == 'pos') | (self.isotherm.lower() == 'positive'): 
			above_zero = year_img.select('tavg').gt(0)
			return year_img.updateMask(above_zero)
		elif (self.isotherm.lower() == 'neg') | (self.isotherm.lower() == 'negative'): 
			below_zero = year_img.select('tavg').lte(0)
			return year_img.updateMask(below_zero)
		else: 
			print('You need to select either a negative or positive isotherm.')
			return None 

class ExportImages(): 
	"""Define a class for exporting the images that were defined in the GetDaymet class."""
	def __init__(self,export_img,year,isotherm,aoi,scale=1000):
		self.export_img=export_img
		self.scale=scale
		self.year=year
		self.isotherm=isotherm
		self.aoi=aoi

	def run_exports(self): 
		"""Export some data."""

		task=ee.batch.Export.image.toDrive(
			image = self.export_img,
			description= f'daymet_{self.year}_ak_{self.isotherm}_isotherm_nonbinary_annual',
			folder="ak_daymet",
			fileNamePrefix=f'daymet_{self.year}_ak_{self.isotherm}_isotherm_nonbinary_annual',
			region = self.aoi.first().geometry(), 
			scale = self.scale, 
			crs = 'EPSG:3338', 
			maxPixels = 1e13, 
			fileFormat = 'GeoTIFF'
			)
		#start the task in GEE 
		print(task)
		task.start()


def main(ak,isotherm): 

	for year in range(1980,2021): #years: this is exclusive 
		try: 
			image_inst = GetDaymet(ak,year,isotherm)
			image = image_inst.get_ics()
			#image = image_inst.create_isotherms()
		except IndexError as e: 
			pass 
		exports = ExportImages(image,year,isotherm,ak).run_exports()

if __name__ == '__main__':
	
	ak = ee.FeatureCollection("TIGER/2018/States").filter(ee.Filter.eq('NAME','Alaska'))

	main(ak,'all')