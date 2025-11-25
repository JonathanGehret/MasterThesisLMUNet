# generate diversity
# transform to .md? -> yess

library(furrr)
future::plan(multisession) # parallel computing


library(NLMR)
library(landscapemetrics)
library(landscapetools)
library(raster)
library(ggplot2)
library(recolorize) # for reading in images/landscapes again
library(reticulate)
np = import("numpy")

# 1. generate n landscapes and save them as .png
# completely random, simpson diversity of 1
# https://ropensci.github.io/NLMR/articles/articles/overview_tips.html
generate_landscapes = function(s = 1, n = 9999, nrow = 128, ncol = 128, dir = "10000_random/", img_format = ".png", calc_metr = F) {
  n_len = nchar(as.character(n)) # length of n, ie. if n == 9999, n_len == 4
  for (i in s:n) {
    #for (i in 6760:n) {
    print(paste("ls:", i))
    #landscape = nlm_fun(nrow=nrow,ncol=ncol,p=0.5,
    #                              ai = c(0.3,0.6,0.1), rescale=FALSE)
    
    # introduce some randomness:
    # generate number of classes 3 <= # <= 6
    n_c = floor(runif(1, min=3, max=7))
    
    # generate n_c ai values between 0 and 1, adding up to 1
    ai_values = runif(n_c, min=0.1, max=0.9)
    ai_normalized = ai_values / sum(ai_values)
    ai_normalized_rounded = round(ai_normalized, digits= 1)
    
    # generate p values between 0.1 and 0.6
    p_value = round(runif(1, min=0.1, max=0.6), digits=1)
    
    #landscape = nlm_randomcluster(nrow = nrow,ncol = ncol,p = 0.5,
    #                              ai = c(0.3,0.6,0.1), rescale=FALSE)
    
    landscape = nlm_randomcluster(nrow = nrow,ncol = ncol,p = p_value,
                                  ai = ai_normalized_rounded, rescale=FALSE)
    
    # save landscapes as NumPy array
    i_len = nchar(as.character(i))
    i_just = paste(rep("0", n_len - i_len),  collapse = "") # rjust function fo some nice zeroes
    ls_dir = paste("data/landscapes/", dir, sep = "")
    file_name = paste(ls_dir, "ls_", i_just, i, ".npy", sep = "")
    
    # create ls directory if doesnt exist
    dir.create(file.path(ls_dir), showWarnings = FALSE) 
    
    # convert to matrix as np cant work with raster
    landscape_matrix = as.matrix(landscape)
    
    # save landscape as numpy array
    np$save(file_name, landscape_matrix)
    
    # directly calculate metric if so desired
    if (calc_metr == T) {
      generate_metrics(n = n, i = i, dir = dir, landscape = landscape)
    }
    
  }
}


# 2. loop over all landscapes and calculate metric:

generate_metrics = function(n = 9999, i = 1, dir = "10000_random/", img_format = ".npy", landscape = NA, read_img = F, lsm_metric = "lsm_l_sidi") {
  
  # naming with inserting some zeros for even-length file names
  n_len = nchar(as.character(n)) # length of n, ie. if n == 9999, n_len == 4
  print(paste("metr:", i))
  i_len = nchar(as.character(i))
  i_just = paste(rep("0", n_len - i_len),  collapse = "") # rjust function for some nice zeroes
  ls_name = paste("ls_", i_just, i, img_format, sep = "")
  
  if (read_img == T) {
    #create landscape file names
    ls_dir = paste("data/landscapes/", dir, sep = "")
    ls_path = paste(ls_dir, ls_name, sep = "")
    
    # read in images
    #ls = load.image("test_images/land_example_loseless_07.png")
    landscape = np$load(ls_path)

    # s_matr = data.matrix(ls_df)
    # as.matrix(ls[0,1])
    
    # convert to calculable landscape
    #landscape = as.matrix(landscape)
    landscape = raster(landscape)
  }
  
  plot(landscape)
  
  # create metric file names
  metr_name = paste("metr_", i_just, i, img_format, sep = "")
  metr_dir = paste("data/metrics/", dir, sep = "")
  dir.create(file.path(metr_dir), showWarnings = FALSE) #create metric directory if doesnt exist
  metr_path = paste(metr_dir, metr_name, sep = "")
  
  # create moving window metric of landscape
  # https://r-spatialecology.github.io/landscapemetrics/reference/window_lsm.html
  # https://docs.google.com/spreadsheets/d/1bU5SIYuFBJMne0J-a5CxNS_WfCIpv5NHflHaf3kUCH4/edit#gid=1696753557
  window = matrix(1, nrow = 5, ncol = 5)
  metric = window_lsm(landscape, window = window, what = lsm_metric) # simpsons diversity index, #1 for ls lvl
  

  
  # convert to matrix as np cant work with raster
  #metric_matrix = as.matrix(metric$layer_1$lsm_l_sidi)
  
  #metric_matrix = as.matrix(paste("metric$layer_1$", lsm_metric, sep = ""))
  metric_matrix = as.matrix(metric$layer_1$lsm_l_sidi) # how to make dependant on input metric? paste doesn't help here yet...
  
  #plot(raster(metric_matrix))
  
  # save metric matrix as numpy array
  np$save(metr_path, metric_matrix)
  
  # create indexing list
  ls_metr_line = paste(ls_name, metr_name, sep = ",")
  metric_list_dir = paste("data/metric_list/", dir, sep = "")
  dir.create(file.path(metric_list_dir), showWarnings = FALSE) #create metric directory if doesnt exist
  write.table(ls_metr_line, file = paste(metric_list_dir, "metric_list.csv", sep = ""),
              row.names=F, col.names=F, append = T, quote = F)
}


# generate metrics for n things:
s = 1
n = 1000
for (i in s:n) {
  generate_metrics(n = n, i, dir = "10000_random_npy/", read_img = T) 
}


# generate landscapes
# generate_landscapes(s = 1, n = 1000, dir = "window_sidi/", calc_metr = T)
# generate landscpaes reticulate
generate_landscapes(s = 11, n = 12, dir = "testing_reticulate/", calc_metr = T)

generate_landscapes(s = 3, n = 4, dir = "testing_reticulate/", calc_metr = T)

ls_path = "data/landscapes/10000_random_npy/ls_0001.npy"
lsm_metric = "lsm_l_sidi"
plot(metric$layer_1$lsm_l_sidi)
metric_matrix = as.matrix(metric$layer_1$lsm_l_sidi)
plot(raster(metric_matrix))
