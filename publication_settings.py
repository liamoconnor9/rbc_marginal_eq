import numpy as np
import matplotlib

# set up various things for postscript eps plotting.
golden_mean = (np.sqrt(5)-1.0)/2.0
fig_width = 3.4 # in column width
# fig_width = 7.1 # in page width
fig_height = fig_width * golden_mean
fig_size =  [fig_width,fig_height]
colors = ['#a8ddb5',
  '#7bccc4',
  '#4eb3d3',
  '#2b8cbe',
  '#08589e']

params = {#'backend': 'eps',
          'axes.prop_cycle' : matplotlib.cycler(color=colors),
          'axes.labelsize': 10,
          'font.family':'serif',
          #'text.fontsize': 8,
          'font.size':8,
          'font.serif': [],
          'font.sans-serif': ['DejaVu Sans'],
          'mathtext.fontset' : 'stix',
        #   'text.usetex' : True,
          'legend.fontsize': 8,
          'legend.markerscale': 1,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
#          'text.usetex': True,
          'figure.figsize': fig_size,
          'figure.dpi': 600,
          'lines.markersize':3,
          'lines.linewidth': 1,
          'lines.markeredgewidth':1,
#          'lines.dashes':(),
          'figure.subplot.left': 0.20,
          'figure.subplot.bottom': 0.20,
	  'figure.subplot.right': 0.95,
	  'figure.subplot.top': 0.90,
	  'figure.subplot.hspace': 0.1
          }

    
#def setup_plot():
#
#    import publication_settings
#
#    rcParams.update(publication_settings.params)
