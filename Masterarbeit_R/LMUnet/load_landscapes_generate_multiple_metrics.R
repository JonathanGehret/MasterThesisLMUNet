library(NLMR)
library(landscapemetrics)
library(landscapetools)
library(raster)
library(tidyverse)
#library(recolorize) # for reading in images/landscapes again
#library(reticulate)
#np = import("numpy")

library(furrr)
future::plan(multisession) # parallel computing

LS_FORMAT = ".npy"
#all_metrics = read.csv("all_lsm_list.csv")
#all_metrics = read.csv("all_working_metrics.csv")
all_metrics = read.csv("final_metrics.csv")
LSM_METRIC = c()
for (i in all_metrics) {
  
  LSM_METRIC = append(LSM_METRIC, i)
  
}


#number_of_landscapes = 100
number_of_landscapes = 1
#number_of_metrics = 66 
number_of_metrics = length(all_metrics[[1]]) 

number_total = number_of_metrics * number_of_landscapes

LSM_METRIC = rep(LSM_METRIC,number_of_landscapes)


# proportion of individual 3,4,5 or 6 classes
#number_classes = floor(runif(100000, min = 3, max = 7)) 
number_classes = rep(3,number_total) 
#number_classes = rep(5,number_total) 


# which landscapes to load in
#landscape_list_dir = "data/landscapes/npy_3_class_lsm_l_sidi_5_mov_win/"
#landscape_list = read.csv("data/metric_list/npy_3_class_lsm_l_sidi_5_mov_win/metric_list.csv", header = F)
#landscape_list_dir = "data/landscapes/npy_5_class_no_metric_ls_only_5_mov_win/"
#landscape_list = read.csv("data/metric_list/npy_5_class_no_metric_ls_only_5_mov_win/metric_list.csv", header = F)
#landscape_list_01 = landscape_list$V1[1:number_of_landscapes]
#landscape_list_01 = landscape_list$V1[2]
#landscape_list_01 = landscape_list$V1
#metric_list_01 = landscape_list$V2

#name = c()

#name = "example_landscape"
name = "example_corine"

#for (i in landscape_list_01) {

#  name = append(name, rep(substr(i, 4,11), number_of_metrics))
#name = append(name, substr(i, 4,11))


#}

# window size  either 3 or 5, to spice things up!
#window_size = sample(c(3,5),100000, replace = TRUE) 
window_size = 5

DATA_DIRECTORY = paste(substr(LS_FORMAT,2,4), 
                       number_classes, "class", 
                       name,
                       window_size, "mov_win/", 
                       sep = "_")

generate_landscapes_metrics = function(dir = DATA_DIRECTORY, ls_format = LS_FORMAT, 
                                       lsm_metric = LSM_METRIC,
                                       number_classes = number_classes,
                                       name = name, 
                                       window_size = 5){
  
  #inside here because of furrr shenanigans
  library(reticulate)
  np = import("numpy")
  
  # landscape name for saving the metric
  ls_name = paste("ls_", name, ls_format, sep = "")
  
  # load landscape
  #landscape_dir = paste(landscape_list_dir, "/", ls_name, sep = "")
  #landscape_dir = "example_landscape.npy"
  landscape_dir = paste(name, ".npy", sep = "")
  landscape <- np$load(landscape_dir)
  
  # specify metric
  print(lsm_metric)
  lsm_metric = lsm_metric
  #print(lsm_metric)
  
  # create metric file names
  metr_dir = paste("data/metrics/", dir, sep = "")
  metr_name = paste("metr_", name, "_", lsm_metric, ls_format, sep = "")
  #metr_name_metr = paste(metr_name, lsm_metric, sep = "")
  metr_file = paste(metr_dir, metr_name, sep = "")
  #metr_file_metr = paste(metr_dir, metr_name, lsm_metric, sep = "")
  dir.create(file.path(metr_dir), showWarnings = FALSE) #create metric directory if doesnt exist
  
  # create moving window metric of landscape
  window = matrix(1, nrow = window_size, ncol = window_size)
  landscape = raster(landscape)
  
  metric = landscapemetrics::window_lsm(landscape, window = window, what = lsm_metric) # simpsons diversity index, #1 for ls lvl
  
  #for saving we need to extract the matrix
  metric_matrix = as.matrix(metric[[1]][[1]])
  
  # save metric matrix as numpy array
  #np$save(metr_file_metr, metric_matrix)
  np$save(metr_file, metric_matrix)
  
  
  # save number classes! to the list as 3rd column, 
  # and also the metric used as 4th column, 
  # and window size as 5th. cool
  # and gernation functino as 6th. nice.
  
  # create indexing list
  ls_metr_line = paste(ls_name, metr_name, sep = ",")
  ls_metr_line_meta = paste(ls_name, metr_name, 
                            number_classes, lsm_metric, window_size, 
                            "nlm_randomcluster", sep = ",")
  metric_list_dir = paste("data/metric_list/", dir, sep = "")
  dir.create(file.path(metric_list_dir), showWarnings = FALSE) #create metric directory if doesnt exist
  write.table(ls_metr_line, file = paste(metric_list_dir, "metric_list.csv", sep = ""),
              row.names=F, col.names=F, append = T, quote = F)
  write.table(ls_metr_line_meta, file = paste(metric_list_dir, "metric_list_meta.csv", sep = ""),
              row.names=F, col.names=F, append = T, quote = F)
}

param_df = tibble( 
  dir = DATA_DIRECTORY, ls_format = LS_FORMAT, 
  lsm_metric = LSM_METRIC, number_classes = number_classes, 
  name = name)


nlm_randomcluster_landscapes <- param_df |> 
  furrr::future_pmap(generate_landscapes_metrics, .options = furrr_options(seed = TRUE))
