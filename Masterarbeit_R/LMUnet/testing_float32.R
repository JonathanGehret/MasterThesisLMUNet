
library(NLMR)
library(landscapemetrics)
library(landscapetools)
library(raster)
library(tidyverse)

library(reticulate)
np = import("numpy")

landscape = NLMR::nlm_randomcluster(nrow = 128,ncol = 128,p = 0.5,
                                    ai = c(0.1,0.3,0.2,0.3,0.1), rescale=FALSE)



# visualize landscape
landscapetools::show_landscape(landscape |> raster::stack())


# save landscapes as NumPy array
#ls_name = paste("ls_", name, ls_format, sep = "")

# create metric file names
#metr_dir = paste("data/metrics/", dir, sep = "")
#metr_dir = paste("/home/uni01/UFOM/jgehret/data/metrics/", dir, sep = "")
#metr_dir = paste("/scratch/users/jgehret/data/metrics/", dir, sep = "")
#metr_name = paste("metr_", name, ls_format, sep = "")
#metr_file = paste(metr_dir, metr_name, sep = "")
#dir.create(file.path(metr_dir), showWarnings = FALSE) #create metric directory if doesnt exist

# create moving window metric of landscape
# https://r-spatialecology.github.io/landscapemetrics/reference/window_lsm.html
# https://docs.google.com/spreadsheets/d/1bU5SIYuFBJMne0J-a5CxNS_WfCIpv5NHflHaf3kUCH4/edit#gid=1696753557
#window_size = sample(c(3,5),1) # window size  either 3 or 5, to spice things up!
window = matrix(1, nrow = 5, ncol = 5)
#landscape_raster = raster(landscape)

lsm_metric = "lsm_l_sidi"

metric = landscapemetrics::window_lsm(landscape, window = window, what = lsm_metric) # simpsons diversity index, #1 for ls lvl


metric_matrix = as.matrix(metric[[1]][[1]])

plot(raster(metric_matrix))


# try this: 
# set to float32
metric_matrix_np = np_array(metric_matrix, dtype = "float32")
metric_matrix_64 = np_array(metric_matrix, dtype = "float64")

metric_matrix_64$dtype
metric_matrix_np$dtype

# save metric matrix as numpy array
np$save(metr_file, metric_matrix)