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
from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler

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
	

	#for year,c in zip(range(1986,2022,2),color_ls): 
	#	plot_df = df.loc[df['year'] == year]
		# ax1.plot(plot_df['elev_band_max'], 
		# 		plot_df['area'],  
		# 		c=c,
		# 		linewidth=2)

	#df = df.loc[df['year'].isin(range(1990,2022,2))]
	#add a new col that puts something different in the 1986 and 1988 rows than everything else to change their styles below 
	df['styles'] = np.where(df['year'] >= 1990,1,0)
	line_styles = ['-','--']
	styles_dict = {1: '', 0:(4,1.5)}
	cmap = plt.cm.RdGy_r
	N=18
	rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
	#cycler(color=cmap(np.linspace(0, 1, N)))
	colors = plt.cm.RdGy_r(np.linspace(0.0,1.0,N)) # This returns RGBA; convert:
	# hexcolor = map(lambda rgb:'#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255),
 #               tuple(color[:,0:-1]))
	
	#create a custom legend
	labels = sorted([str(y) for y in df['year'].unique()])
	# for n in range(N): 
	# 	if (n <= 1): 
	# 		linestyle = '--'
	# 	else: 
	# 		linestyle = '-'
	# 	legend_lines.append((Line2D([0], [0], c=colors[n][:-1], lw=2,label=labels[n]),ls=linestyle))
	#create a custom legend so we can get the dashed lines in there
	custom_lines = [Line2D([0], [0], c=colors[0][:-1], lw=2,label=labels[0],ls='--'),
	                Line2D([0], [0], c=colors[1][:-1], lw=2,label=labels[1],ls='--'),
	                Line2D([0], [0], c=colors[2][:-1], lw=2,label=labels[2]),
	                Line2D([0], [0], c=colors[3][:-1], lw=2,label=labels[3]),
	                Line2D([0], [0], c=colors[4][:-1], lw=2,label=labels[4]),
	                Line2D([0], [0], c=colors[5][:-1], lw=2,label=labels[5]),
	                Line2D([0], [0], c=colors[6][:-1], lw=2,label=labels[6]),
	                Line2D([0], [0], c=colors[7][:-1], lw=2,label=labels[7]),
	                Line2D([0], [0], c=colors[8][:-1], lw=2,label=labels[8]),
	                Line2D([0], [0], c=colors[9][:-1], lw=2,label=labels[9]),
	                Line2D([0], [0], c=colors[10][:-1], lw=2,label=labels[10]),
	                Line2D([0], [0], c=colors[11][:-1], lw=2,label=labels[11]),
	                Line2D([0], [0], c=colors[12][:-1], lw=2,label=labels[12]),
	                Line2D([0], [0], c=colors[13][:-1], lw=2,label=labels[13]),
	                Line2D([0], [0], c=colors[14][:-1], lw=2,label=labels[14]),
	                Line2D([0], [0], c=colors[15][:-1], lw=2,label=labels[15]),
	                Line2D([0], [0], c=colors[16][:-1], lw=2,label=labels[16]),
	                Line2D([0], [0], c=colors[17][:-1], lw=2,label=labels[17])
	                ]

	# print(custom_lines)
	# fig, ax = plt.subplots()
	# lines = ax.plot(data)
	# print([str(y) for y in df['year'].unique()])
	
	# custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
 #                Line2D([0], [0], color=cmap(.5), lw=4),
 #                Line2D([0], [0], color=cmap(1.), lw=4)]

	# # fig, ax = plt.subplots()
	# # lines = ax.plot(data)
	# ax1.legend(custom_lines, ['Cold', 'Medium', 'Hot'])


	ax1.legend(handles=custom_lines, loc='upper right',ncol=2)

	#add a couple of bolder grid lines 
	ax2.plot(df['elev_band_max'].unique(),
		np.zeros(shape=len(df['elev_band_max'].unique())),
		linewidth=2,
		c='gray', 
		ls = '--', 
		alpha=0.5
		)

	#add some vertical elevation lines 
	for xi in range(0,5000,500): 
		ax1.axvline(x=xi, color='gray',ls='--',lw=1,alpha=0.25)
		ax2.axvline(x=xi, color='gray',ls='--',lw=1,alpha=0.25)

	sns.lineplot(x='elev_band_max', 
				 y='area',  
				 data=df,
				 hue='year', 
				 style='styles',
				 dashes=styles_dict,
				 palette=cmap,
				 linewidth=1.5,
				 ax=ax1, 
				 legend=False
				 )

	sns.lineplot(x='elev_band_max', 
				 y='mean_temp',  
				 data=df,
				 hue='year', 
				 palette=cmap,
				 linewidth=1.5,
				 ax=ax2, 
				 legend=False
				 )
	
	ax1.set_ylabel('Glacier area ($km^2$)',fontsize=10)
	ax2.set_ylabel('Mean annual \ntemp ($^\circ$C)',fontsize=10)
	ax2.set_xlabel('Elevation band max (m)',fontsize=10)
	ax1.grid(axis='both',alpha=0.25)
	ax2.grid(axis='both',alpha=0.25)
	ax1.set_xlim(200,5000)
	ax2.set_xlim(200,5000)
	ax1.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',') if x >= 10000 else int(x)))
	# plt.show()
	# plt.close('all') 

	plot_fn = os.path.join(output_dir,'revised_200m_bands_temp_area_elev_plot_amended_lines_v4.jpg')
	
	if not os.path.exists(plot_fn): 
		plt.savefig(plot_fn, 
					dpi=500,
					bbox_inches = 'tight',
					pad_inches = 0.1
					)


if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		fig_dir = variables['fig_dir']
		data = "/vol/v3/ben_ak/excel_files/stats_for_isotherm_figs/area_temp_by_200m_elev_band_and_year_08122021_model_revised_resampling.csv"
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
