
import math
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

DEFAULT_LIGHTNESS = 0.8
DEFAULT_ASPECT = 0.35 # 0.25
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

__all__ = ["evans_plot", "EvansPlotter", "MONTHS"]

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

class EvansPlotter:
    """
    Class for generating Evans plots from two variables of rectangular data.
    See documentation of `evans_plot` function for explanations of parameters.
    """
    
    def __init__(self, 
                 data : xr.Dataset, 
                 hue_var : str, 
                 sat_var : str,
                 lightness = DEFAULT_LIGHTNESS,
                 hue_threshhold_step_size: int = None,
                 num_sat_levels : int = 5,
                 min_sat = None,
                 max_sat = None,
                 hue_threshholds = None,
                 cmap = "hsv",
                 cbar = "horizontal",
                 hue_value_labels : list[str] = None,   
                 hue_offset_deg : float = 0.0,
                 font_size : float = None,
                 ):
          
        self.hue_data = data[hue_var] # hue (HSL)
        self.sat_data = data[sat_var] # satruation (HS)
        self.lightness = lightness # lightness (HSL)
        self.lat = self.hue_data.lat
        self.lon = self.hue_data.lon
        self.num_sat_levels = num_sat_levels
        self.cmap = cmap
        
        # cartography
        self.projection = ccrs.PlateCarree()
        self.feature_alpha = 0.3 # map cartopy features transparency
        self.cartopy_features = [cfeature.COASTLINE, cfeature.BORDERS, cfeature.LAKES,  cfeature.RIVERS, cfeature.OCEAN, cfeature.LAND]
        
        self._hue_data_discrete = True 
        self._sat_data_discrete = False 
        self.hue_data_min = int(self.hue_data.min().item())
        self.hue_data_max = int(self.hue_data.max().item())
        self.hue_threshhold_step_size = hue_threshhold_step_size or self._suggest_step_size(self.hue_data)
        #print("hue data step size:", self.hue_threshhold_step_size)
        
        self.hue_threshholds = hue_threshholds  
            
        # digitize data now so we know if bins are exceeded
        self._hue_data_q = self._hue_data_quantized() 
        
        # determine if hu-date is discrete or continuous
        self.hue_data_is_discrete = len(self._get_unique_data_values(self.hue_data)) - len(self._get_hue_bins()) <= 1  
        
        # maximum value of the saturation
        self.max_sat = max_sat 
        if max_sat is None:
            self.max_sat = self.sat_data.max().item() 
            
        # minimum value of the satruation
        self.min_sat = min_sat 
        if min_sat is None: 
            self.min_sat = self.sat_data.min().item() 
        
        # labels
        self.hue_value_labels = hue_value_labels 
        self.hue_label = "" 
        self.sat_label = "" 
        self.font_size_title = 18
        self.font_size = font_size or 8.0
        
        self.hue_offset_deg = hue_offset_deg #0.0 # degrees 
        self.cbar = cbar # None, 'horizontal', 'vertical', 'cyclic'
        
        # formatting
        # self.max_hue_level = 1.0
        # self.min_hue_level = 0.0
        # self.hue_level_spacing = 0.25
        # self.min_intensity = 0.8        # TODO: what is this
        # self.num_max_sat_levels = 1     # TODO: clarify what is this? really max sat for more than 1 value?
        # self.reverse_sat_colors = False # TODO: what is this? reverse saturation: 0==color, 1==grey?
        # self.reverse_hues = False       # TODO
     
     
    @property
    def num_hue_levels(self):
        return len(self._get_hue_bins()) + 1
        
    @property
    def num_displayed_hue_levels(self):
        """
        How many hue levels to display on the color bar.
        This can be equal to number of bins, eg if discrete data, 
        or +1 if continuous and max bin is exceeded."""
        # as we're using digitize with right=True then we only need to check whether
        # last bin value has been reached or exceeded
        data_range = self._hue_data_q.max() - self._hue_data_q.min() + 1
        n = max(data_range, len(self._get_hue_bins()))

        return n
     
    # Utilitiy functions 
     
    def _extent(self):
        # Return the bounding box of lat & lon
        return [self.lon.min(), 
                self.lon.max(), 
                self.lat.min(), 
                self.lat.max()]
        
    def _get_unique_data_values(self, data):
        # Return sorted list of the unique values in supplied data object, 
        # ignoring NaNs
        unique = data.to_dataframe().iloc[:, 0].unique()
        unique.sort()
        if np.isnan(unique[-1]):
            unique = unique[:-1]
        return unique
        
    def _suggest_step_size(self, data):
        # Return a good sugestino for step size for the data
        # TODO: make this smarter
        min = data.min().item()
        max = data.max().item()
        #bigger = math.max(abs(min), abs(max))
        delta = max - min
        if delta < 20:
            return 1
        elif delta < 100:
            return 10
        elif delta < 200:
            return 10
        elif delta < 1000:
            return 100
        else:
            return delta // 10
        
    # Hue Data processing 
     
    def get_hue_value_labels(self):
        if self.hue_value_labels is not None:
            # is user has provided hue value labels, use them
            return self.hue_value_labels
        labels = list(self._get_hue_bins())
        while len(labels) < self.num_displayed_hue_levels:
            labels.append(labels[-1] + self.hue_threshhold_step_size)
        return map(str, labels)
        
    
    def _get_hue_bins(self):
        if self.hue_threshholds is not None:
            # if user has provided hue threshholds, use them
            return self.hue_threshholds
        else:
            step = self.hue_threshhold_step_size
            bin_min = (self.hue_data_min // step) * step 
            bin_max = (self.hue_data_max // step) * step
            if self.hue_data_max > bin_max:
                bin_max += step
            threshholds = np.arange(bin_min, bin_max, self.hue_threshhold_step_size)
            # don't need first threshhold unless it is the minimum value
            if threshholds[0] < self.hue_data_min:  
                threshholds = threshholds[1:] # remove first element
            return threshholds
        
    def _hue_data_quantized(self): 
        bins = self._get_hue_bins()
        #data = np.nan_to_num(self.hue_data)
        data = np.nan_to_num(self.hue_data, nan=0) # convert nans to 0
        data = np.digitize(data, bins, right=True) # quantize or digitze data into these bins
        return data
        
    # Saturation Data processing 
    
    def _saturation_data(self):
        """Return a matrix of saturation values [0,1] quantized"""
        # crop data that falls outside the min and max saturation
        normalised_sat_data = self.sat_data.clip(min=self.min_sat, max=self.max_sat)
        
        # Normalise saturation data to [0, 1]
        normalised_sat_data = (normalised_sat_data - self.min_sat) / (self.max_sat - self.min_sat)
        normalised_sat_data = np.nan_to_num(normalised_sat_data)
        
        # Break satruation into discrete levels
        # n categories have n+1 (outer) threshholds, but we
        # don't need the first one as it should be the minimum.
        threshholds = np.linspace(0, 1, self.num_sat_levels + 1)[1:]
        
        # Quantize/digitize the data into bins
        q_data = np.digitize(normalised_sat_data, threshholds, right=True) 
        
        # Convert to [0,1] range. 
        # max value is (num_sat_levels - 1) which should map to 1.0
        q_data = q_data / (self.num_sat_levels - 1)
        return q_data 
    
    # Color functions
    
    def _cmap(self):
        if self.cmap is None:
            cmap = mpl.pyplot.get_cmap("hsv")
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
        
    def get_colors(self):
        """Return array of rgb values to use for the hues"""
        buffer = 0.5 / self.num_hue_levels 
        col_locs = np.linspace(buffer , 1.0-buffer, self.num_hue_levels)
        cmap = self._cmap()
        colors = list(map(cmap, col_locs))
        # TODO allow list of user specified colors
        return np.array(colors)     
    
    # Color bar functions    
    
    def _show_hue_bins_as_threshholds(self):
        # Return boolean, whether to
        # show bin vales as thresholds, between pixels
        # (best if hue data is continuous)
        # or else as categories, centered on the pixel
        # (best if hue data is discrete, such as month)
        if self.hue_threshholds:
            # if explicit hue levels are provided, then assume hue data is continuous
            return True
        if self.hue_value_labels is not None and len(self.hue_value_labels) > 0:
            # if hue labels are set, then assume they are discrete categories
            return False
        if self.hue_data_is_discrete:
            return False
        return True # default to continuous data
    
    def _get_cbar_boxes(self):
        sat_vals = np.linspace(0, 1, self.num_sat_levels)
        hue_vals = list(range(0, self.num_displayed_hue_levels))
        hue_boxes = np.array([hue_vals])
        hue_boxes = np.repeat(hue_boxes, len(sat_vals), axis=0)
        sat_boxes = np.array([sat_vals])
        sat_boxes = np.repeat(sat_boxes, len(hue_vals), axis=0)
        sat_boxes = sat_boxes.transpose()
        return sat_boxes, hue_boxes
    
    def _get_cbar_rgba(self):
        colors = self.get_colors()
        sat_boxes, hue_boxes = self._get_cbar_boxes()

        # convert to rgb grids
        L = self.lightness
        grey = (L,L,L)
        rgba = np.ones((hue_boxes.shape[0], hue_boxes.shape[1], 4))
        for i in (0,1,2): # r, g, b
            rgba[:, :, i] = colors[:,i][hue_boxes] * sat_boxes + grey[i] * (1-sat_boxes)
        return rgba
        
    def color_bar_horz(self, ax):
        """Render a horizontal color bar into the supplied axes."""
        rgba = self._get_cbar_rgba()
  
        # render the rgba data in the axes
        ax.pcolormesh(rgba)
        
        # stretch the color bar so it's nice a long, not square
        ax.set_aspect(DEFAULT_ASPECT)
        
        # set x labels
        if self._show_hue_bins_as_threshholds():
            delta = 1.0
        else:
            # center ticks on the pixel
            delta = 0.5
        xtic_locs = [delta + x for x in range(self.num_displayed_hue_levels)]
        ax.set_xticks(xtic_locs, 
                      labels=self.get_hue_value_labels(), 
                      fontsize=self.font_size,
                      ha="center")
        ax.set_xlabel(self.hue_label, fontsize=self.font_size)
       
        # set y labels
        label_vals = np.linspace(self.min_sat, self.max_sat, self.num_sat_levels+1)
        labels = [f"{x:.2g}" for x in list(label_vals)]
        ax.set_yticks(range(len(labels)), labels=[str(x) for x in labels],
                      fontsize=self.font_size - 2,
                      ha='right')
        ax.set_ylabel(self.sat_label, fontsize=self.font_size)
        
    def color_bar_vert(self, ax):
        """Render a vertical color bar into the supplied axes."""
        # get rgbc and transpose x,y to make it vertical
        rgba = self._get_cbar_rgba().transpose((1, 0, 2)) 
  
        # render the rgba data in the axes
        ax.pcolormesh(rgba)
        
        # stretch the color bar so it's nice a long, not square
        ax.set_aspect(1/DEFAULT_ASPECT)
        
        # set y labels
        if self._show_hue_bins_as_threshholds():
            delta = 1.0
        else:
            # center ticks on the pixel
            delta = 0.5
        ytic_locs = [delta + y for y in range(self.num_displayed_hue_levels)]
        ax.set_yticks(ytic_locs, 
                      labels=self.get_hue_value_labels(), 
                      fontsize=self.font_size,
                      ha="left")
        ax.set_ylabel(self.hue_label, fontsize=self.font_size)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
       
        # set x labels
        label_vals = np.linspace(self.min_sat, self.max_sat, self.num_sat_levels+1)
        labels = [f"{x:.2g}" for x in list(label_vals)]
        ax.set_xticks(range(len(labels)), labels=[str(x) for x in labels],
                      fontsize=self.font_size - 2,
                      ha='right')
        ax.set_xlabel(self.sat_label, fontsize=self.font_size)
        
    def color_bar_cyclic(self, ax):
        """Render a color bar key into the supplied axes."""
        colors = self.get_colors()
        sat_boxes, hue_boxes = self._get_cbar_boxes()
        
        # convert to rgb grids
        L = self.lightness
        grey = (L,L,L)
        rgba = np.ones((hue_boxes.shape[0], hue_boxes.shape[1], 4))
        for i in (0,1,2): # r, g, b
            rgba[:, :, i] = colors[:,i][hue_boxes] * sat_boxes + grey[i] * (1-sat_boxes)
  
        theta = np.linspace(0, 2*np.pi, self.num_displayed_hue_levels + 1, endpoint=True)
        r = np.linspace(0, self.num_sat_levels, num=self.num_sat_levels+1)
        ax.set_rlim(0, r[-1])
        
        # render the rgba data in the axes
        ax.pcolormesh(theta, r, rgba, shading='flat')
        ax.grid(visible=None)
        ax.set_frame_on(False)
        ax.set_aspect('equal')
        
        # set circumference labels
        theta = 2 * np.pi / self.num_displayed_hue_levels
        if self._show_hue_bins_as_threshholds():
            delta = theta
        else:
            # center ticks on the pixel
            delta = theta / 2
        
        xtic_locs = [ x * theta + delta for x in range(self.num_displayed_hue_levels)]
        for i in range(self.num_displayed_hue_levels):
            if xtic_locs[i] > np.pi:
                xtic_locs[i] = xtic_locs[i] - 2 * np.pi
        
        ax.set_xticks(xtic_locs, 
                      labels=self.get_hue_value_labels(), 
                      fontsize=self.font_size,
                      ha="center")
        
        # prevent matplotlib from not showing full theta of the plot
        ax.set_thetalim(-np.pi, np.pi)
        #ax.set_thetalim(0, 2*np.pi)
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
       
        # set labels along the radius
        label_vals = np.linspace(self.min_sat, self.max_sat, self.num_sat_levels+1)
        labels = [f"{x:.2g}" for x in list(label_vals)]
        ax.set_rticks(range(len(labels)), labels=[str(x) for x in labels],
                      fontsize=self.font_size - 1,
                      ha='center')
        ax.set_rlabel_position(0)
        # or do it as text overlays...
        # ax.set_rticks([])
        # for i in range(len(labels)):
        #     ax.text(0,i, labels[i])
          
    # Main plotting functions 
    
    def get_axes(self, fig=None, title=None, figsize=None):
        if fig is None:
            fig = plt.figure(figsize=figsize)
            # Add title
            if title is not None:
                fig.suptitle(title, fontsize=self.font_size_title, y=0.93)
        
        if self.cbar == 'cyclic':
            gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.1)
            ax1 = fig.add_subplot(gs[0], projection=self.projection)
            ax2 = fig.add_subplot(gs[1], projection="polar")
            
        elif self.cbar == 'vertical':
            gs = GridSpec(1, 2, width_ratios=[6, 1], wspace=0.1) 
            ax1 = fig.add_subplot(gs[0], projection=self.projection)
            ax2 = fig.add_subplot(gs[1])
        elif self.cbar is not None: # == "horizontal":
            gs = GridSpec(2, 1, height_ratios=[5, 1], wspace=0.05, hspace=0.2) 
            ax1 = fig.add_subplot(gs[0], projection=self.projection)
            ax2 = fig.add_subplot(gs[1])
        else:
            ax1 = fig.add_subplot(projection=self.projection)
            ax2 = None
        # get rid of the degrees symbol on the axes
        ax1.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
        ax1.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='')) 
        return ax1, ax2    
   
    def plot_data(self, ax, **kwargs):
        """Render the data variables as hue and saturation pixels into the supplied axes."""
        # convert hue_data array to RGB
        hue_data_q = self._hue_data_q # use cached quantized data
        colors = self.get_colors() # list of rgba tuples to map hue index to
        # initialise 4 layer matrix rgba - Red, Green, Blue, Alpha
        rgba = np.ones((len(self.lat), len(self.lon), 4))
        S = self._saturation_data() # matrix
        L = self.lightness # scalar
        grey = (L,L,L)
        # use hue_data as index to convert values to colors
        for i in (0,1,2):
            rgba[:, :, i] = colors[:,i][hue_data_q] * S + grey[i] * (1-S)
        # render nan as transparent
        rgba[:, :, 3] = ~np.isnan(self.sat_data) 
        # render the color mesh, pass on extra key word args to pcolormesh
        ax.pcolormesh(self.lon, 
                      self.lat, 
                      rgba, 
                      **kwargs)
    
    def plot(self, ax, cbar_ax=None, **kwargs):
        """Plot the data and color bar to a figure."""
        # plot main axis
        ax.set_extent(self._extent(), crs=self.projection)
        
        # This line generates the actual plot
        self.plot_data(ax, **kwargs)
        
        # add cartopy features
        for f in self.cartopy_features:
            ax.add_feature(f, alpha=self.feature_alpha)
        # Add gridlines
        lon_fmt=cartopy.mpl.ticker.LongitudeFormatter(degree_symbol='')
        lat_fmt=cartopy.mpl.ticker.LatitudeFormatter (degree_symbol='')
        gl = ax.gridlines(draw_labels=True, alpha=self.feature_alpha, xformatter=lon_fmt, yformatter=lat_fmt)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': self.font_size}
        gl.ylabel_style = {'size': self.font_size}
        
        # show color bar
        if self.cbar is None or self.cbar == False:
            pass # no color bar
        elif self.cbar == 'cyclic':
            self.color_bar_cyclic(cbar_ax)
        elif self.cbar == 'vertical':
            self.color_bar_vert(cbar_ax)
        elif self.cbar == 'horizontal' or self.cbar == True:
            self.color_bar_horz(cbar_ax)
        else:
            raise ValueError("cbar must be 'horizontal', 'vertical', 'cyclic' or boolean or None")
    
    
def evans_plot(data, hue_var, sat_var, 
               num_sat_levels=5, min_sat=None, max_sat=None,
               hue_threshhold_step_size=None, hue_threshholds=None,
               fig=None, ax=None, cbar_ax=None, figsize=None,
               cbar=True,
               title=None, hue_label=None, sat_label=None,
               cmap=None, 
               hue_value_labels=None,
               hue_offset_deg = 0,
               lightness = DEFAULT_LIGHTNESS,
               font_size=None,
               **kwargs):
    """Plot two variables of rectangular data as hue and saturation. 
    
    The datasets are assumed to be aligned and the same shape.
    
    If no Axes is provided to the ``ax`` argument, use the currently active Axes.
    Part of the Axes space will be used to plot a colormap, unless ``cbar`` is False
    or a separate Axes is provided to ``cbar_ax``.
    
    Parameters
    ----------
    data: xarray.DataSet with at least two variables of rectangular data that are aligned
    
    hue_var : str
        Variable to use for the hue.
        
    sat_var: str
        Variable to use for  the saturation.
        
    num_sat_levels : int, optional
        Number of saturation levels to use. Default is 5.
        
    min_sat : float, optional
        Value to use as the minimum saturation. If None, use the minimum value from the data.
        
    max_sat : float, optional
        Value to use as the maximum saturation. If None, use the maximum value from the data.
        
    hue_threshhold_step_size :  int, optional
        Hue thresholds, which will be multiples of this step size.
        e.g. if step size is 10, and min hue data value is -17 and max hue data value is 22
        then threshholds will be [-20, -10, 0, 10, 20, 30].
        If None, a "good" step size is suggested. For discrete data, this will generally be 1.
        
    hue_threshholds : list[int], optional
        List of explicit hue thresholds to use. Overrides the hue_threshhold_step_size.
        
    fig : matplotlib Figure, optional
        Figure in which to draw the plot, otherwise use the currently-active
        Figure. If None, a new figure is created.
        
    figsize : tuple, optional
        Size of the figure in inches, e.g. (8, 6). 
    
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
        
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar.
        
    cbar : str, optional
        How and whether to draw a color bar. 
        Can be 'horizontal', 'vertical', 'cyclic' or boolean or None

    title : str, optional
        Title of the plot. If None, no title is set.
        
    hue_label : str, optional
        Label for the hue variable. If None, use the hue_var name.
        
    sat_label : str, optional
        Label for the saturation variable. If None, use the sat_var name.
        
    cmap : str or matplotlib colormap, optional 
        Colormap to use for the hue variable. If None, use 'hsv'.
        
    hue_value_labels :  list[str], optional
        List of labels for the hue values. If None, use the hue variable values.
        If the hue data is discrete, then these labels will be used as categories.
        If the hue data is continuous, then these labels will be used as thresholds
        between pixels.
        
    hue_offset_deg : float, optional
        Offset in degrees to apply to the hues, to effectively rotate the hues.
        
    lightness : float, optional
        Lightness value to use for the colors. Default is 0.8.
        This is the lightness in HSL color space.
        It should be a value between 0 and 1, where 0 is black and 1 is white.
        
    font_size : float, optional
        Font size to use for the labels. Default is 8.0.
      
    kwargs : other keyword arguments
        All other keyword arguments are passed to
        :meth:`matplotlib.axes.Axes.pcolormesh`.
        
    """
    plotter = _EvansPlotter(data, hue_var, sat_var, 
                            hue_threshhold_step_size=hue_threshhold_step_size,
                            num_sat_levels=num_sat_levels,
                            min_sat=min_sat, max_sat=max_sat,
                            hue_threshholds=hue_threshholds,
                            cbar=cbar, 
                            cmap=cmap, 
                            lightness=lightness,
                            hue_value_labels=hue_value_labels,
                            hue_offset_deg=hue_offset_deg,
                            font_size=font_size
                          )

    plotter.hue_label = hue_label #or hue_var
    plotter.sat_label = sat_label #or sat_var
    if ax is None:
        #    fig = plt.gcf()? #    ax = plt.gca()?
        # if fig is None:
        #     fig = plt.gcf()
        ax, cbar_ax = plotter.get_axes(fig, title, figsize)
    else:
        # otherwise set figure to return from the supplied axes
        fig = ax.figure

    
    # Draw the plot and return the Axes
    plotter.plot(ax=ax, cbar_ax=cbar_ax, **kwargs)
    
    return fig 
       

