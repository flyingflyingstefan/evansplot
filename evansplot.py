
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cm = 1/2.54

DEFAULT_LIGHTNESS = 0.8
DEFAULT_ASPECT = 0.25

__all__ = ["evans_plot"]

def _cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.    
    cmap: colormap instance 
    N: number of colors.    
    """
    if type(cmap) == str:
        cmap = mpl.pyplot.get_cmap(cmap)
    start_buff = (1./ N) * 0.5
    colors_i = np.linspace(start_buff, 1 - start_buff, N)
    colors_i = np.concatenate((colors_i, (0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
    return mpl.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

class _EvansPlotter:
    
    def __init__(self, 
                 data : xr.Dataset, 
                 hue_var : str, 
                 sat_var : str,
                 lightness = DEFAULT_LIGHTNESS,
                 num_hue_levels : int = None,
                 num_sat_levels : int = 5,
                 cmap = "hsv",
                 cbar = "horizontal",
                 hue_value_labels : list[str] = None,   
                 hue_offset_deg : float = 0.0,
                 min_sat = None,
                 max_sat = None
                 #xticklabels : list[str] = None,
                 #yticklabels: list[str] = None
                 ):
          
        self.hue_data = data[hue_var] # hue (HSL)
        self.sat_data = data[sat_var] # satruation (HS)
        self.lightness = lightness # lightness (HSL)
        self.lat = self.hue_data.lat
        self.lon = self.hue_data.lon
        self.num_hue_levels = num_hue_levels
        self.num_sat_levels = num_sat_levels
        self.cmap = cmap
        
        # cartography
        self.projection = ccrs.PlateCarree()
        self.feature_alpha = 0.3 # map cartopy features transparency
        self.cartopy_features = [cfeature.COASTLINE, cfeature.BORDERS, cfeature.LAKES,  cfeature.RIVERS, cfeature.OCEAN, cfeature.LAND]
        
        # at the moment we assume hue data is discrete categories, and sat data is continuous with threshholds
        self._hue_data_discrete = True 
        self._sat_data_discrete = False 
        self.hue_data_min = int(self.hue_data.min().item())
        self.hue_data_max = int(self.hue_data.max().item())
        self._unique_hue_data_values = self._get_unique_data_values(self.hue_data)
        self.hue_value_labels = hue_value_labels #[str(x) for x in self._unique_hue_data_values]
        if num_hue_levels is  None:
            # default to number of unique values in hue_var
            self.num_hue_levels = len(self._unique_hue_data_values)
              
        # remaining attributes mimic the NCL implementation   
        
        # maximum value of the saturation
        self.max_sat = max_sat 
        if max_sat is None:
            self.max_sat = self.sat_data.max().item() 
            
        # minimum value of the satruation
        self.min_sat = min_sat 
        if min_sat is None: 
            self.min_sat = self.sat_data.min().item() 
        
        self.hue_label = "Hue label" #TODO
        self.sat_label = "Saturation label" #TODO
        self.hue_offset_deg = hue_offset_deg #0.0 # degrees 
        self.cbar = cbar 
        self.show_cyclie_car = False # suggested
        
        # formatting
        #self.xticklabels = xticklabels
        #self.yticklabels = yticklabels
        self.font_size_title = 16
        #self.cyclic = False  # TODO # not really a logical thing, more a display thing?
        #self.cbar_orientation = 'horizontal' 
        #self.cbar_font_size = 5
        # self.explicit_hue_levels = []  # replace with num_hue_levels? and explicit colors?
        # self.max_hue_level = 1.0
        # self.min_hue_level = 0.0
        # self.hue_level_spacing = 0.25
        # self.min_intensity = 0.8         # TODO: what is this
        # self.num_max_sat_levels = 1     # TODO: clarify what is this? really max sat for more than 1 value?
        # self.reverse_sat_colors = False # TODO: what is this? reverse saturation: 0==color, 1==grey?
        # self.reverse_hues = False       # TODO
     
     
    def get_hue_value_labels(self):
        if self.hue_value_labels is None:
            #return [str(x) for x in self._unique_hue_data_values]
            return [str(x+1) for x in range(self.num_hue_levels)]
        return self.hue_value_labels
    
    def get_hue_bins(self):
        return np.linspace(self.hue_data_min, self.hue_data_max, self.num_hue_levels + 1)[1:]
        # assume hue values are discrete integers
        #return range(self.hue_data_min, self.hue_data_max+1)
        
    def _get_hue_data_quantized(self): 
        bins = self.get_hue_bins()
        #print("bins", bins)
        #data = np.nan_to_num(self.hue_data)
        data = np.digitize(self.hue_data, bins) # quantize or digitze data into these bins
        data = data - self.hue_data_min # colors start at index 0
        return data
        
    def _get_unique_data_values(self, data):
        """Return sorted list of all possible unique values in supplied data object, ignoring NaNs"""
        unique = data.to_dataframe().iloc[:, 0].unique()
        unique.sort()
        if np.isnan(unique[-1]):
            unique = unique[:-1]
        return unique
        
    def _cmap(self):
        if self.cmap is None:
            #if self.cyclic:
            cmap = mpl.pyplot.get_cmap("hsv")
            #cmap = mpl.pyplot.get_cmap("coolwarm")
        else:
            if isinstance(self.cmap, str):
                cmap = mpl.pyplot.get_cmap(self.cmap)
            else:
                cmap = self.cmap
        if self.hue_offset_deg:
            cmap = self._rotate_cmap(self.hue_offset_deg, cmap)
        return _cmap_discretize(cmap, self.num_hue_levels)

    def _rotate_cmap(self, deg, cmap): 
        # rotate the colormap by degrees
        n = cmap.N
        deg = -deg % 360
        if deg < 0:
            deg += 360
        cutpoint = n * deg // 360
        new_col_arr = [cmap(i) for i in range(cutpoint, n)] + [cmap(i) for i in range(cutpoint)]
        return mpl.colors.ListedColormap(new_col_arr)

    def _extent(self):
        # Return the bounding box of lat & lon
        return [self.hue_data.lon.min(), 
                self.hue_data.lon.max(), 
                self.hue_data.lat.min(), 
                self.hue_data.lat.max()]
    
    def _get_bins(self):
        return np.linspace(0, 1, self.num_sat_levels)[1:]
    
    def _saturation_factor(self):
        """Return a matrix of saturation values [0,1] quantized"""
        # scale saturation data to [0, 1]
        a = (self.sat_data - self.min_sat) / (self.max_sat - self.min_sat)
        
        # account for mininum saturation
        #a = a * (1 - self.min_sat) + self.min_sat  
        
        #a = a - (1 - self.value_hsv)
        # convert NaNs to 0
        #a = np.nan_to_num(a).astype(np.float32) 
        
        # break alpha into discrete levels
        bins = self._get_bins()
        data = np.nan_to_num(self.sat_data)
        q_data = np.digitize(data, bins) / self.num_sat_levels
        return q_data 
        
    def get_colors(self):
        """Return array of rgb values to use for the hues"""
        buffer = 0.5 / self.num_hue_levels 
        col_locs = np.linspace(buffer , 1.0-buffer, self.num_hue_levels)
        cmap = self._cmap()
        colors = list(map(cmap, col_locs))
        # TODO allow list of user specified colors
        return np.array(colors)            
        
    def color_bar_horz(self, ax):
        """Render a color bar key into the supplied axes."""
        colors = self.get_colors()
        hue_vals = list(range(0, self.num_hue_levels))
        sat_vals = np.linspace(self.min_sat, self.max_sat, self.num_sat_levels)
        hue_boxes = np.array([hue_vals])
        hue_boxes = np.repeat(hue_boxes, len(sat_vals), axis=0)
        sat_boxes = np.array([sat_vals])
        sat_boxes = np.repeat(sat_boxes, len(hue_vals), axis=0)
        sat_boxes = sat_boxes.transpose()

        # convert to rgb grids
        L = self.lightness
        grey = (L,L,L)
        rgba = np.ones((hue_boxes.shape[0], hue_boxes.shape[1], 4))
        for i in (0,1,2): # r, g, b
            rgba[:, :, i] = colors[:,i][hue_boxes] * sat_boxes + grey[i] * (1-sat_boxes)
  
        # render the rgba data in the axes
        ax.pcolormesh(rgba)
        
        # stretch the color bar so it's nice a long, not square
        ax.set_aspect(DEFAULT_ASPECT)
        
        # set x labels
        xtic_locs = [0.5 + x for x in range(self.num_hue_levels)]
        ax.set_xticks(xtic_locs, 
                      labels=self.get_hue_value_labels(), 
                      ha="center")
       
        # set y labels
        label_vals = np.linspace(self.min_sat, self.max_sat, self.num_sat_levels+1)
        print("label_vals", label_vals)
        labels = [f"{x:.2g}" for x in list(label_vals)]
        ax.set_yticks(range(len(labels)), labels=[str(x) for x in labels],
                      fontsize=5,
                      ha='right')
        
    def color_bar_cyclic(self, ax):
        """Render a color bar key into the supplied axes."""
        colors = self.get_colors()
        hue_vals = list(range(0, self.num_hue_levels))
        #sat_vals = np.linspace(self.min_sat, self.max_sat, self.num_sat_levels)
        #sat_vals = self._get_bins()
        sat_vals = np.linspace(0, 1, self.num_sat_levels)
        hue_boxes = np.array([hue_vals])
        hue_boxes = np.repeat(hue_boxes, len(sat_vals), axis=0)
        sat_boxes = np.array([sat_vals])
        sat_boxes = np.repeat(sat_boxes, len(hue_vals), axis=0)
        sat_boxes = sat_boxes.transpose()
        print("hue boxed shape", hue_boxes.shape)

        # convert to rgb grids
        L = self.lightness
        grey = (L,L,L)
        rgba = np.ones((hue_boxes.shape[0], hue_boxes.shape[1], 4))
        for i in (0,1,2): # r, g, b
            rgba[:, :, i] = colors[:,i][hue_boxes] * sat_boxes + grey[i] * (1-sat_boxes)
  
        theta = np.linspace(0, 2*np.pi, self.num_hue_levels+1, endpoint=True)
        print("theta is ", theta)
        #r = sat_vals * self.num_sat_levels
        #r = np.linspace(1, self.num_sat_levels, num=self.num_sat_levels)
        r = np.linspace(0, self.num_sat_levels, num=self.num_sat_levels+1)
        #r = self._get_bins()
        print("r is ", r)
        ax.set_rlim(0, r[-1])
        # render the rgba data in the axes
        ax.pcolormesh(theta, r, rgba, shading='flat')
        ax.grid(visible=None)
        ax.set_frame_on(False)
        
        # chaneg from polygon to circle
        ax.set_aspect('equal')
        
        #ax.set_aspect(DEFAULT_ASPECT)
        
        # set circumference labels
        delta = 2 * np.pi / self.num_hue_levels
        xtic_locs = [ x * delta + delta/2 for x in range(self.num_hue_levels)]
        
        for i in range(self.num_hue_levels):
            if xtic_locs[i] > np.pi:
                xtic_locs[i] = xtic_locs[i] - 2 * np.pi
        #ax.set_thetalim(0, 2*np.pi)
        ax.set_xticks(xtic_locs, 
                      labels=self.get_hue_value_labels(), 
                      ha="center")
        
        # prevent matplotlib from not showing full theta of the plot
        ax.set_thetalim(-np.pi, np.pi)
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
       
        # set labels along the radius
        label_vals = np.linspace(self.min_sat, self.max_sat, self.num_sat_levels+1)
        print("label_vals", label_vals)
        labels = [f"{x:.2g}" for x in list(label_vals)]
        ax.set_rticks(range(len(labels)), labels=[str(x) for x in labels],
                      fontsize=7,
                      ha='center')
        ax.set_rlabel_position(0)
        
        # ax.set_rticks([])
        # for i in range(len(labels)):
        #     ax.text(0,i, labels[i])
        
   
    def plot_data(self, ax, **kwargs):
        """Render the data variables as hue and saturation pixels into the supplied axes."""
        # convert hue_data array to RGB
        hue_data_q = self._get_hue_data_quantized()
        colors = self.get_colors() # list of rgba tuples to map hue index to
        # initialise rgba
        rgba = np.ones((len(self.lat), len(self.lon), 4))
        S = self._saturation_factor() # matrix
        L = self.lightness # scalar
        grey = (L,L,L)
        # use hue_data as index to convert values to colors
        for i in (0,1,2):
            rgba[:, :, i] = colors[:,i][hue_data_q] * S + grey[i] * (1-S)
        # render nan as transparent
        rgba[:, :, 3] = ~np.isnan(self.sat_data) 
        # render the color mesh
        ax.pcolormesh(self.lon, 
                      self.lat, 
                      rgba, 
                      **kwargs)
    
    def plot(self, ax, cbar_ax=None, **kwargs):
        """Plot the data and color bar to a figure."""
        # if ax is None:
        #     ax, cbar_ax = self.get_axes()
       
        # plot main axis
        ax.set_extent(self._extent(), crs=self.projection)
        # This line generates the actual plot
        self.plot_data(ax, **kwargs)
        
        # add cartopy features
        for f in self.cartopy_features:
            ax.add_feature(f, alpha=self.feature_alpha)
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=self.feature_alpha)
        gl.top_labels = False
        gl.right_labels = False
        
            
        # show color bar
        if self.cbar is None:
            pass
        elif self.cbar == 'cyclic':
            self.color_bar_cyclic(cbar_ax)
        elif self.cbar == 'vertical':
            print("Warning: vertical colorbar not supported")
        else: # self.cbar == 'horizontal':
            # default, horizontal cbar
            self.color_bar_horz(cbar_ax)
        #return fig
    
    def get_axes(self, fig=None, title=None, figsize=None):
        if fig is None:
            fig = plt.figure(figsize=figsize)
            # Add title
            if title is not None:
                fig.suptitle(title, fontsize=self.font_size_title)
        
        if self.cbar == 'cyclic':
            gs = GridSpec(1, 2, width_ratios=[5, 1]) 
            ax1 = fig.add_subplot(gs[0], projection=self.projection)
            ax2 = fig.add_subplot(gs[1], projection="polar")
        elif self.cbar == 'vertical':
            gs = GridSpec(1, 2, width_ratios=[5, 1]) 
            ax1 = fig.add_subplot(gs[0], projection=self.projection)
            ax2 = fig.add_subplot(gs[1])
        elif self.cbar is not None: # == "horizontal":
            print("horizontal...")
            gs = GridSpec(2, 1, height_ratios=[7, 1]) 
            ax1 = fig.add_subplot(gs[0], projection=self.projection)
            ax2 = fig.add_subplot(gs[1])
        else:
            ax1 = fig.add_subplot(projection=self.projection)
            ax2 = None
        return ax1, ax2
    
    
def evans_plot(data, 
               hue_var, 
               sat_var, 
               num_hue_levels=None, 
               num_sat_levels=5,
               min_sat=None,
               max_sat=None,
               fig=None,
               ax=None, 
               cbar_ax=None,
               figsize=None,
               cmap=None, 
               cbar=True, 
               #xticklabels=None, 
               #yticklabels=None,
               hue_value_labels=None,
               title=None,
               hue_offset_deg = 0,
               lightness = DEFAULT_LIGHTNESS,
               **kwargs):
    """Plot two variables of rectangular data as hue and saturation. 
    
    The datasets are assumed to be aligned and the same shape.
    
    If no Axes is provided to the ``ax`` argument, use the currently active Axes.
    Part of the Axes space will be used to plot a colormap, unless ``cbar`` is False
    or a separate Axes is provided to ``cbar_ax``.
    
    Parameters
    ----------
    hue_data : rectangular dataset
        2D dataset that can be coerced into an ndarray. 
    
    sat_data: same
    
    num_hue_levels: int, optional
        The number of different hue/color categories to use. If not specified,
        defaults to number of unique values in hue_data, if reasonable (<20).
        
    cbar : bool, optional
        Whether to draw a colorbar.
        
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar, otherwise take space from the
        main Axes.
        
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
        
    kwargs : other keyword arguments
        All other keyword arguments are passed to
        :meth:`matplotlib.axes.Axes.pcolormesh`.
        
    """
    # Initialize the plotter object
    # plotter = _EvansPlotter(data, vmin, vmax, cmap, center, robust, annot, fmt,
    #                       annot_kws, cbar, cbar_kws, xticklabels,
    #                       yticklabels, mask)
    
    plotter = _EvansPlotter(data, 
                            hue_var, 
                            sat_var, 
                            num_hue_levels=num_hue_levels,
                            num_sat_levels=num_sat_levels,
                            min_sat=min_sat,
                            max_sat=max_sat,
                            cmap=cmap, 
                          cbar=cbar, 
                          hue_value_labels=hue_value_labels,
                          lightness=lightness,
                          hue_offset_deg=hue_offset_deg
                          )

    
    if ax is None:
        #    fig = plt.gcf()?
        #    ax = plt.gca()?
        ax, cbar_ax = plotter.get_axes(fig, title, figsize)
    else:
        # otherwise make sure the figure belongs to the axes we're working with
        fig = ax.figure

    # Draw the plot and return the Axes
    plotter.plot(ax=ax, cbar_ax=cbar_ax, **kwargs)
   
    return fig #ax, cbar_ax
       

