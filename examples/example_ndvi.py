
import evansplot
import matplotlib.pyplot as plt
import xarray as xr
import sys

ndvi = xr.open_dataset("ndvi_time.nc")
f = evansplot.evans_plot(ndvi, 
                     hue_var="ndvi_timing", 
                     sat_var="ndvi_max", 
                     min_sat=0, max_sat=0.7,
                     title = "AVHRR NDVImax Timing",
                     hue_offset_deg = 30,
                     hue_value_labels=evansplot.LABELS_MONTHS, 
                     cbar="cyclic")

plt.savefig("AVHRR_NDVImax_Timing.png", dpi=300)
