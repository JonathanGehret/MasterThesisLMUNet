library(NLMR)
library(landscapemetrics)
library(landscapetools)
library(raster)
library(tidyverse)
#library(ggplot2)
#library(recolorize) # for reading in images/landscapes again
#library(reticulate)
#np = import("numpy")

library(furrr)
future::plan(multisession) # parallel computing



#DATA_DIRECTORY = "npy_3_class_sidi_5_mov_win/"
#LANDSCAPE_DIRECTORY = paste(DATA_DIRECTORY, "variable_classes/", sep =)
#METRICS_DIRECTORY = 
DIM = 128
LS_FORMAT = ".npy"
LSM_METRIC = "lsm_l_sidi"
#LSM_METRIC = "lsm_l_pd"
#LSM_METRIC = "lsm_l_area_cv"
#LSM_METRIC = "lsm_l_frac_mn"
#LSM_METRIC = "lsm_l_iji"


# proportion of individual 3,4,5 or 6 classes
#number_classes = floor(runif(100000, min = 3, max = 7)) 
number_classes = rep(4,100000) 


# ai values
param_df = tibble(n = number_classes, min = 0.1, max = 0.9)
ai_values <- param_df |> 
  furrr::future_pmap(runif, .options = furrr_options(seed = TRUE))
ai_normalized = lapply(ai_values, function(x){x/sum(x)} )
# this doesn't always add up to 1, but i guess it works anyways.
ai_normalized_rounded = lapply(ai_values, function(x){round(x/sum(x), digits = 1)} )

# generate cluster element propertion between 0.1 and 0.6
p_value = round(runif(100000, min = 0.1, max = 0.6), digits = 1)

# file names
name = as.character(round(runif(100000, min=10000000, max=99999999), 0))

# window size  either 3 or 5, to spice things up!
#window_size = sample(c(3,5),100000, replace = TRUE) 
window_size = 5

DATA_DIRECTORY = paste(substr(LS_FORMAT,2,4), 
                       number_classes, "class", 
                       LSM_METRIC, 
                       window_size, "mov_win/", 
                       sep = "_")


generate_landscapes_metrics = function(nrow = DIM, ncol = DIM, 
                                       dir = DATA_DIRECTORY, ls_format = LS_FORMAT, 
                                       lsm_metric = LSM_METRIC,
                                       number_classes = number_classes,
                                       ai_normalized_rounded = ai_normalized_rounded,
                                       p_value = p_value, name = name, 
                                       windpw_size = window_size) {
  
  #inside here because of furrr shenanigans
  library(reticulate)
  np = import("numpy")
  
  # proportion of individual 3,4,5 or 6 classes
  #number_classes = floor(runif(1, min = 3, max = 7)) 
  #ai_values = runif(number_classes, min = 0.1, max = 0.9)
  #ai_normalized = ai_values / sum(ai_values)
  #ai_normalized_rounded = round(ai_normalized, digits = 1)
  
  # generate cluster element propertion between 0.1 and 0.6
  #p_value = round(runif(1, min = 0.1, max = 0.6), digits = 1)
  
  landscape = NLMR::nlm_randomcluster(nrow = nrow,ncol = ncol,p = p_value,
                                      ai = ai_normalized_rounded, rescale=FALSE)
  
  # visualize landscape
  #landscapetools::show_landscape(landscape |> raster::stack())
  
  # random name...could probably use index in futrue somehow, but as of know i don't know how
  #name = as.character(round(runif(1, min=0, max=100000000), 0))
  
  # save landscapes as NumPy array
  #i_len = nchar(as.character(i))
  #i_just = paste(rep("0", n_len - i_len),  collapse = "") # rjust function fo some nice zeroes
  ls_dir = paste("data/landscapes/", dir, sep = "")
  ls_name = paste("ls_", name, ls_format, sep = "")
  ls_file = paste(ls_dir, ls_name, sep = "")
  dir.create(file.path(ls_dir), showWarnings = FALSE)   # create ls directory if doesnt exist
  
  # convert to matrix as np cant work with raster
  landscape_matrix = as.matrix(landscape)
  
  # save landscape as numpy array
  np$save(ls_file, landscape_matrix)
  
  # create landscapes only indexing list
  ls_line_meta = paste(ls_name, number_classes, 
                       "nlm_randomcluster", sep = ",")
  ls_list_dir = paste("data/ls_list/", dir, sep = "")
  dir.create(file.path(ls_list_dir), showWarnings = FALSE) #create ls directory if doesnt exist
  write.table(ls_line_meta, file = paste(ls_list_dir, "ls_list.csv", sep = ""),
              row.names=F, col.names=F, append = T, quote = F)

  

}

#generate_landscapes_metrics()


param_df = tibble(nrow = DIM, ncol = DIM, 
                  dir = DATA_DIRECTORY, ls_format = LS_FORMAT, 
                  lsm_metric = LSM_METRIC, number_classes = number_classes, 
                  ai_normalized_rounded = ai_normalized_rounded,
                  p_value = p_value, name = name)

nlm_randomcluster_landscapes <- param_df |> 
  furrr::future_pmap(generate_landscapes_metrics, .options = furrr_options(seed = TRUE))
