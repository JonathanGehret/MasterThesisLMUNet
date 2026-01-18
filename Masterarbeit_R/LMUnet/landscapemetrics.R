# get landscape metrics

library(landscapemetrics)
library(landscapetools)
library(raster)

getwd()


# import rasters
ufc2001 = raster('FRAGSTATS-EDIS-Data/UrbanForestCover2001.tif')
ufc2011 = raster('FRAGSTATS-EDIS-Data/UrbanForestCover2011.tif')

# Plot the rasters
#spplot(ufc2001)
#spplot(ufc2011)
show_landscape(ufc2001)
#show_landscape(ufc2001)

# inspect landscapes for validity
check_landscape(ufc2001)
check_landscape(ufc2011)

# calculate the Euclidean nearest-neighbor distance on patch level
lsm_p_enn(ufc2001)

# calculate the total area (landscape lvl)
lsm_l_ta(ufc2001)
# pretty big area

# calculate total class edge length 
lsm_c_te(ufc2001)
# also pretty big

# calculate *all* metrix on patch level
# gonna take a loooooooooooong time, lets see :D
all_patch_metrics = calculate_lsm(ufc2001, level = "patch")

# calculate moving window anaylis
window = matrix(1, nrow=5,ncol=5)
moving_window = window_lsm(ufc2001, window = window, what=c("lsm_l_pr", "lsm_l_joinent"))

# list all available metrics
list_lsm()

#saving and loading
saveRDS(moving_window, file="D:/Documents/Goettingen/Masterarbeit/Masterarbeit_R_landscape_metrics/data/ufc2011_moving_window.RData")
#load again:
readRDS("D:/Documents/Goettingen/Masterarbeit/Masterarbeit_R_landscape_metrics/data/ufc2011_moving_window.RData")

saveRDS(all_patch_metrics, file="D:/Documents/Goettingen/Masterarbeit/Masterarbeit_R_landscape_metrics/data/all_patch_metrix.RData")

# finding the right metrics! try out:
# largest patch index (=large patch dominance? c#6/l#2)
ufc2001_c_lpi = lsm_c_lpi(ufc2001)
ufc2001_l_lpi = lsm_l_lpi(ufc2001)


