import rasterio 
import os
import sys 
import pandas as pd 
import geopandas as gpd 
import json 
import numpy as np 
from osgeo import gdal
import fiona 
import rasterio.mask
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from colour import Color
import matplotlib as mpl
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

def plot_3d_lines(df,output_dir): 

	fig = plt.figure(figsize=(8,6))
	ax = fig.add_subplot(111, projection='3d')
	#fig,ax = plt.subplots(1,projection='3d',figsize=(8,6))

	black = Color("black")
	start_color = Color('darkblue')
	color_ls = list(black.range_to(Color("#ffffff"),19))
	color_ls = [c.hex_l for c in color_ls]
	colors_dict = dict(zip(range(1986,2022,2),color_ls))

	for year,c in zip(range(1986,2022,2),color_ls): 
		plot_df = df.loc[df['year'] == year]
		ax.plot(plot_df['mean_temp'], 
				plot_df['area'], 
				plot_df['elev_band_max'], 
				c=c,
				linewidth=2)

	# make a color map of fixed colors
	cmap = colors.ListedColormap(color_ls)
	bounds = list(colors_dict.keys())#.append(2022)
	print(bounds)
	norm = colors.BoundaryNorm(bounds, cmap.N)

	cbar_ax = fig.add_axes([0.87, 0.3, 0.02, 0.5]) #formatted like:  [left, bottom, width, height].
	fig.colorbar(mpl.cm.ScalarMappable(norm=norm, 
				 cmap=cmap), 
		         cax=cbar_ax, 
		         orientation='vertical',
		         ticks=[1990,2000,2010,2020])
	#set the 'camera angle' using elevation and azmiuth 
	ax.view_init(elev=28., azim=-80)
	ax.set_xlabel('Mean annual temp (deg C)')
	ax.set_ylabel('Area (km2)')
	ax.set_zlabel('Elev band max (m)')

	plt.show()
	plt.close('all')
	# print('ax.azim {}'.format(ax.azim))
	# print('ax.elev {}'.format(ax.elev))
	# plot_fn = os.path.join(output_dir,'3d_temp_area_elev_test_plot_draft9.jpg')
	# if not os.path.exists(plot_fn): 
	# 	plt.savefig(plot_fn, 
	# 				dpi=500,
	# 				bbox_inches = 'tight',
	# 				pad_inches = 0.1
	# 				)

def plot_side_by_side(df,output_dir): 

	black = Color("black")
	start_color = Color('darkblue')
	color_ls = list(start_color.range_to(Color("darkred"),19))
	color_ls = [c.hex_l for c in color_ls]
	colors_dict = dict(zip(range(1986,2022,2),color_ls))

	fig,(ax1,ax2) = plt.subplots(2,figsize=(8,6),sharex=True,gridspec_kw={'wspace':0,'hspace':0})
	print(df)
	#for year,c in zip(range(1986,2022,2),color_ls): 
	#	plot_df = df.loc[df['year'] == year]
		# ax1.plot(plot_df['elev_band_max'], 
		# 		plot_df['area'],  
		# 		c=c,
		# 		linewidth=2)

	sns.lineplot(x='elev_band_max', 
				 y='area',  
				 data=df,
				 hue='year', 
				 palette='BrBG',
				 linewidth=2.5,
				 ax=ax1, 
				 legend=False
				 #legend='full'
				 )

	sns.lineplot(x='elev_band_max', 
				 y='mean_temp',  
				 data=df,
				 hue='year', 
				 palette='BrBG',
				 linewidth=2.5,
				 ax=ax2, 
				 legend='full'
				 )
	#add an inset plot 
	#first create some data, restrict to the higher elevations 
	# high_elev_df = df.loc[df['elev_band_max']>=3000]
	# ax3 = plt.axes([0,0,1,1])
	# # Manually set the position and relative size of the inset axes within ax1
	# ip = InsetPosition(ax1, [0.5,0.4,0.45,0.5])
	# ax3.set_axes_locator(ip)
	# # Mark the region corresponding to the inset axes on ax1 and draw lines
	# # in grey linking the two axes.
	# #mark_inset(ax1, ax3, loc1=2, loc2=4, fc="none", ec='0.5')
	# sns.lineplot(x='elev_band_max', 
	# 			 y='area',  
	# 			 data=high_elev_df,
	# 			 hue='year', 
	# 			 palette='BrBG',
	# 			 linewidth=2.5,
	# 			 ax=ax3, 
	# 			 legend=False
	# 			 )

	ax2.legend(loc='lower left',ncol=2)
	ax1.set_ylabel('Glacier area (km2)')
	ax2.set_ylabel('Mean annual \ntemp (deg C)')
	ax2.set_xlabel('Elevation band max (m)')
	ax1.grid(axis='y',alpha=0.25)
	ax2.grid(axis='y',alpha=0.25)
	ax1.set_xlim(500,5000)
	ax2.set_xlim(500,5000)

	plt.show()
	plt.close('all') 

	# plot_fn = os.path.join(output_dir,'revised_temp_area_elev_plot_draft2.jpg')
	# if not os.path.exists(plot_fn): 
	# 	plt.savefig(plot_fn, 
	# 				dpi=500,
	# 				bbox_inches = 'tight',
	# 				pad_inches = 0.1
	# 				)


if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		output_dir = variables['output_dir']
		dem = variables['dem']
		pickles = variables['pickles']
		ecoregions = variables['ecoregions']
		fig_dir = variables['fig_dir']
		data = "/vol/v3/ben_ak/excel_files/stats_for_isotherm_figs/area_temp_by_250m_elev_band_and_year_08122021_model_revised_resampling.csv"
		# test = pd.read_csv(data)
		# test = test.loc[test.year == 1988]
		# print(test)
		#plot_3d_lines(pd.read_csv(data),fig_dir)
		plot_side_by_side(pd.read_csv(data),fig_dir)

# """
# =============================================
# Generate polygons to fill under 3D line graph
# =============================================

# Demonstrate how to create polygons which fill the space under a line
# graph. In this example polygons are semi-transparent, creating a sort
# of 'jagged stained glass' effect.
# """

# # from mpl_toolkits.mplot3d import Axes3D
# # from matplotlib.collections import PolyCollection
# # import matplotlib.pyplot as plt
# # from matplotlib import colors as mcolors
# # import numpy as np


# # fig = plt.figure()
# # ax = fig.gca(projection='3d')


# # def cc(arg):
# #     return mcolors.to_rgba(arg, alpha=0.6)

# # xs = np.arange(0, 10, 0.4)
# # verts = []
# # zs = [0.0, 1.0, 2.0, 3.0]
# # for z in zs:
# #     print(z)
# #     ys = np.random.rand(len(xs))
# #     print('xs is: ')
# #     print(xs)
# #     print('ys is: ')
# #     print(ys)
# #     ys[0], ys[-1] = 0, 0
# #     verts.append(list(zip(xs, ys)))
# # print('here verts is: ')
# # print(verts)
# # poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),
# #                                          cc('y')])
# # poly.set_alpha(0.7)
# # ax.add_collection3d(poly, zs=zs, zdir='y')

# # ax.set_xlabel('X')
# # ax.set_xlim3d(0, 10)
# # ax.set_ylabel('Y')
# # ax.set_ylim3d(-1, 4)
# # ax.set_zlabel('Z')
# # ax.set_zlim3d(0, 1)

# # plt.show()
