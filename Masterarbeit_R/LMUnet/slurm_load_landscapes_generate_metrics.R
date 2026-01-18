#jonathan@e5a63f8597c6:~/data$ df -BG
#Filesystem     1G-blocks  Used Available Use% Mounted on
#overlay             158G  102G       50G  67% /

library(NLMR)
library(landscapemetrics)
library(landscapetools)
library(raster)
library(tidyverse)

library(rslurm)

#dont need this because reasons!
#library(furrr)
#future::plan(multisession) # parallel computing

DIM = 128
LS_FORMAT = ".npy"

##########################################################
##########################################################

#LSM_METRIC = "lsm_l_sidi" #100k 3 class, 100 3 classs, 10k 5 class, 100k 5 class
#LSM_METRIC =  "lsm_l_joinent" #100k 5 class PC1
#LSM_METRIC =  "lsm_l_dcad" #100k 5 class PC4
#LSM_METRIC = "lsm_l_area_cv" #100k 5 class PC2
#LSM_METRIC = "lsm_l_frac_mn" #20k 5 class +100k 5 class PC3
#LSM_METRIC = "lsm_l_pladj" #10k 5 class #this one next? 100k 5 class
#LSM_METRIC =  "lsm_l_ed" #100k 5 class PC4
#LSM_METRIC =  "lsm_l_ndca" #100k 5 class 
#LSM_METRIC =  "lsm_l_para_mn" #100k 5 class 
#LSM_METRIC =  "lsm_l_mutinf" #100k 5 class 
#LSM_METRIC = "lsm_l_contag" #10k 5 class PC2 #100k 5 class 
#LSM_METRIC = "lsm_l_condent" #100k 5 class 
#LSM_METRIC = "lsm_l_area_sd" #100k 5 class
#LSM_METRIC = "lsm_l_cai_cv" #100k 5 class
#LSM_METRIC = "lsm_l_dcore_sd" #100k 5 class
#LSM_METRIC = "lsm_l_frac_cv" #100k 5 class
#LSM_METRIC = "lsm_l_frac_sd" #100k 5 class
#LSM_METRIC = "lsm_l_para_cv" #100k 5 class
#LSM_METRIC = "lsm_l_shape_cv" #100k 5 class
#LSM_METRIC = "lsm_l_area_mn" #100k 5 class
#LSM_METRIC = "lsm_l_cai_mn" #100k 5 class
LSM_METRIC = "lsm_l_cai_sd" #100k 5 class

#LSM_METRIC = "lsm_l_circle_cv" #100k 5 class
#LSM_METRIC = "lsm_l_circle_mn" #100k 5 class
#LSM_METRIC = "lsm_l_circle_sd" #100k 5 class
#LSM_METRIC = "lsm_l_cohesion" #100k 5 class
#LSM_METRIC = "lsm_l_core_cv" #100k 5 class
#LSM_METRIC = "lsm_l_core_mn" #100k 5 class
#LSM_METRIC = "lsm_l_core_sd" #100k 5 class
#LSM_METRIC = "lsm_l_dcore_cv" #100k 5 class
#LSM_METRIC = "lsm_l_dcore_mn" #100k 5 class
#LSM_METRIC = "lsm_l_division" #100k 5 class
#LSM_METRIC = "lsm_l_enn_cv" #100k 5 class
#LSM_METRIC = "lsm_l_enn_mn" #100k 5 class
#LSM_METRIC = "lsm_l_enn_sd" #100k 5 class
#LSM_METRIC = "lsm_l_ent" #100k 5 class
#LSM_METRIC = "lsm_l_iji" #100k 5 class
#LSM_METRIC = "lsm_l_lpi" #100k 5 class
#LSM_METRIC = "lsm_l_lsi" #100k 5 class
#LSM_METRIC = "lsm_l_mesh" #100k 5 class
#LSM_METRIC = "lsm_l_msidi" #100k 5 class
#LSM_METRIC = "lsm_l_msiei" #100k 5 class
#LSM_METRIC = "lsm_l_np" #100k 5 class
#LSM_METRIC = "lsm_l_pafrac" #100k 5 class
#LSM_METRIC = "lsm_l_para_sd" #100k 5 class
#LSM_METRIC = "lsm_l_pd" #100k 5 class
#LSM_METRIC = "lsm_l_pr" #100k 5 class
#LSM_METRIC = "lsm_l_prd" #100k 5 class
#LSM_METRIC = "lsm_l_relmutinf" #100k 5 class
#LSM_METRIC = "lsm_l_rpr" #100k 5 class
#LSM_METRIC = "lsm_l_shape_mn" #100k 5 class
#LSM_METRIC = "lsm_l_shape_sd" #100k 5 class
#LSM_METRIC = "lsm_l_shdi" #100k 5 class
#LSM_METRIC = "lsm_l_shei" #100k 5 class
#LSM_METRIC = "lsm_l_siei" #100k 5 class
#LSM_METRIC = "lsm_l_split" #100k 5 class
#LSM_METRIC = "lsm_l_ta" #100k 5 class
#LSM_METRIC = "lsm_l_tca" #100k 5 class
#LSM_METRIC = "lsm_l_te" #100k 5 class

###############################################
###############################################



number_of_metrics = 1
#number_of_metrics = 100000
number_classes = rep(5,number_of_metrics) # 5 classes

# file names
#name = as.character(round(runif(100000, min=10000000, max=99999999), 0))
#name = as.character(round(runif(100000, min=10000000, max=99999999), 0))

#landscape_list_dir = "data/landscapes/npy_3_class_lsm_l_sidi_5_mov_win/"
#landscape_list = read.csv("data/metric_list/npy_3_class_lsm_l_sidi_5_mov_win/metric_list_100.csv", header = F)

#working_dir = "/home/uni01/UFOM/jgehret/_rslurm_metric/_rslurm_metric/"
#working_dir = "/home/uni01/UFOM/jgehret/"
working_dir = "/scratch/users/jgehret/"


#landscape_list_dir = "data/landscapes/npy_5_class_no_metric_ls_only_100k_5_mov_win/"
#landscape_list_dir = "/home/uni01/UFOM/jgehret/_rslurm_metric/slurm_data/landscapes/npy_5_class_1_landscape/"
#landscape_list_dir = paste(working_dir, "slurm_data/landscapes/npy_5_class_1_landscape/", sep = "")
landscape_list_dir = paste(working_dir, "data/landscapes/npy_5_class_no_metric_ls_only_100k_5_mov_win/", sep = "")
landscape_list = read.csv(paste(working_dir, "data/metric_list/npy_5_class_no_metric_ls_only_100k_5_mov_win/metric_list.csv", sep = ""), header = F)
#landscape_list = read.csv("data/metric_list/npy_5_class_no_metric_ls_only_100k_5_mov_win/metric_list.csv", header = F)
landscape_list_01 = landscape_list$V1
#metric_list_01 = landscape_list$V2

name = c()

for (i in landscape_list_01) {
  
  name = append(name, substr(i, 4,11))
  
}



# window size  either 3 or 5, to spice things up!
#window_size = sample(c(3,5),100000, replace = TRUE) 
window_size = 5

DATA_DIRECTORY = paste(substr(LS_FORMAT,2,4), 
                       number_classes, "class", 
                       LSM_METRIC, "100k",
                       window_size, "mov_win/", 
                       sep = "_")


generate_landscapes_metrics = function(nrow = DIM, ncol = DIM, 
                                       landscape_list_dir = landscape_list_dir,
                                       dir = DATA_DIRECTORY, ls_format = LS_FORMAT, 
                                       lsm_metric = LSM_METRIC,
                                       number_classes = number_classes,
                                       name = name, 
                                       window_size = 5) {
  
  #inside here because of furrr shenanigans
  library(reticulate)
  np = import("numpy")
  
  #print(i)
  #landscape_list_dir = "slurm_data/landscapes/npy_5_class_1_landscape/"
  #landscape_list_dir = "/home/uni01/UFOM/jgehret/_rslurm_metric/_rslurm_metric/slurm_data/landscapes/npy_5_class_1_landscape/"
  #name = "one_landscape"
  landscape_dir = paste(landscape_list_dir, "ls_", name, ".npy", sep = "")
  #landscape_dir = paste(landscape_list_dir, name, ".npy", sep = "")
  #print(ls_dir)
  landscape <- np$load(landscape_dir)
  #plot(raster(test_ls))
  
  
  
  # visualize landscape
  #landscapetools::show_landscape(landscape |> raster::stack())
  
  
  # save landscapes as NumPy array
  ls_name = paste("ls_", name, ls_format, sep = "")
  
  # create metric file names
  #metr_dir = paste("data/metrics/", dir, sep = "")
  #metr_dir = paste("/home/uni01/UFOM/jgehret/data/metrics/", dir, sep = "")
  metr_dir = paste("/scratch/users/jgehret/data/metrics/", dir, sep = "")
  metr_name = paste("metr_", name, ls_format, sep = "")
  metr_file = paste(metr_dir, metr_name, sep = "")
  dir.create(file.path(metr_dir), showWarnings = FALSE) #create metric directory if doesnt exist
  
  # create moving window metric of landscape
  # https://r-spatialecology.github.io/landscapemetrics/reference/window_lsm.html
  # https://docs.google.com/spreadsheets/d/1bU5SIYuFBJMne0J-a5CxNS_WfCIpv5NHflHaf3kUCH4/edit#gid=1696753557
  #window_size = sample(c(3,5),1) # window size  either 3 or 5, to spice things up!
  window = matrix(1, nrow = window_size, ncol = window_size)
  landscape = raster(landscape)
  
  metric = landscapemetrics::window_lsm(landscape, window = window, what = lsm_metric) # simpsons diversity index, #1 for ls lvl
  
  metric_matrix = as.matrix(metric[[1]][[1]])
  
  # try this: 
  # set to float32
  #metric_matrix = np_array(metric_matrix, dtype = "float32")
  
  # save metric matrix as numpy array
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
  metric_list_dir = paste("/scratch/users/jgehret/data/metric_list/", dir, sep = "")
  dir.create(file.path(metric_list_dir), showWarnings = FALSE) #create metric directory if doesnt exist
  write.table(ls_metr_line, file = paste(metric_list_dir, "metric_list.csv", sep = ""),
              row.names=F, col.names=F, append = T, quote = F)
  write.table(ls_metr_line_meta, file = paste(metric_list_dir, "metric_list_meta.csv", sep = ""),
              row.names=F, col.names=F, append = T, quote = F)
}

#generate_landscapes_metrics()



#loading landscapes and running metric calculation

#landscape_list_dir = "data/landscapes/npy_3_class_lsm_l_sidi_5_mov_win/"
#landscape_list = read.csv("data/metric_list/npy_3_class_lsm_l_sidi_5_mov_win/metric_list_100.csv", header = F)
#landscape_list_01 = landscape_list$V1
#metric_list_01 = landscape_list$V2


par_df = tibble(nrow = DIM, ncol = DIM, 
                landscape_list_dir = landscape_list_dir,
                dir = DATA_DIRECTORY, ls_format = LS_FORMAT, 
                lsm_metric = LSM_METRIC, number_classes = number_classes,
                name = name)


#nlm_randomcluster_landscapes <- param_df |> 
#  furrr::future_pmap(generate_landscapes_metrics, .options = furrr_options(seed = TRUE))


#par_df <- data.frame(x = 1:5)

sjob <- slurm_apply(f = generate_landscapes_metrics, 
                    params = par_df, 
                    #y = 10,
                    #jobname = 'metrics_100k_area_sd',
                    jobname = LSM_METRIC,
                    #nodes = nrow(par_df), 
                    nodes = 1042, 
                    #nodes = 2084, # "invalid job array specification"
                    #nodes = 4168, # "invalid job array specification"
                    #cpus_per_node = 1,
                    cpus_per_node = 96,
                    #cpus_per_node = 24,
                    #cpus_per_node = 48,
                    rscript_path = "/opt/sw/rev/21.12/haswell/gcc-9.3.0/r-4.2.2-2nuddo/rlib/R/bin/Rscript", # adjust to your Rscript path
                    slurm_options = list("time" = "00:10:00", 
                                         "mem-per-cpu" = "1G", # 500 MB is enough
                                         "mail-type" = "ALL",
                                         "mail-user" = "jonathan.gehret@forst.uni-goettingen.de"),
                    submit = FALSE)