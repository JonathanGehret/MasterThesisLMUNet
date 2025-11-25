# generate diversity
# transform to .md? -> yess

library(NLMR)
#library(igraph)
library(landscapemetrics)
library(landscapetools)
library(raster)
#library(rgdal)
#library(sets)


# 1. generate n landscapes and save them as .csv, separated by " "

# number of landscapes
n = 10000

# image size 128
nrow = 128
ncol = 128

# completely random, simpson diversity of 1
#for (i in 1:n) {
for (i in 6760:n) {
  print(i)
  landscape = nlm_randomcluster(nrow=nrow,ncol=ncol,p=0.5,
                                ai = c(0.3,0.6,0.1), rescale=FALSE)
  #landscapes_midpoint_replacement[i] = nlm_mpd(ncol=ncol, nrow=nrow, roughness = 0.61)
  write.table(as.array(landscape), 
              file = paste("data/landscapes/10000_random/ls_", i, ".csv", sep = ""),
              row.names=F, col.names=F)
}

# 2. loop over all landscapes and calculate metric:
# obsolete, use vectorization instead! (unless being read in individually?!?)
# write as function
for (i in 1:6759) {
  print(i)
  #create landscape and metric file names
  ls_name = paste("ls_",i,".csv",sep="")
  metr_name = paste("metr_",i,".csv",sep="")
  # read in landscape and convert to raster layer
  # write as function? "csv_to_ls()"
  ls_df = read.table(paste("data/landscapes/10000_random/",ls_name, sep=""), sep = " ")
  ls_matr = data.matrix(ls_df)
  landscape = raster(nrow = 128, ncol = 128, ext=extent(0, 128, 0, 128))
  values(landscape) = ls_matr
  
  create_list = F
  if (create_list == T) {
    # caculate metric and write to csv
    #metric = lsm_c_ca(landscape) # class lvl: total class area)
    metric = lsm_l_ai(landscape) 
    write.csv(metric, file = paste("data/metrics/metric_tables/metric_tables_ls_aggregation/", metr_name, sep=""))
  }
  
  create_map = F
  if (create_map == T) {
    # create metric label map for the calculated class metric for semantic segmentation
    # by replacing all class values of landscape with the corresponding metric
    metric_map = landscape
    for (class in metric$class) {
      metr_value = metric[metric$class == class,]$value
      metric_map@data@values[landscape@data@values == class] = metr_value
      
    }
    
    # write metric map to file
    write.table(as.array(metric_map), 
                file = paste("data/metrics/metric_maps/", metr_name, sep = ""),
                row.names=F, col.names=F)
  }
  
  moving_window = T
  if (moving_window == T) {
    window = matrix(1, nrow = 5, ncol = 5)
    window_lsm(landscape, window=window, what=c("lsm_l_pr","lsm_l_joient"))
    window_lsm(landscape_stack, window = window, what = c("lsm_l_pr", "lsm_l_joinent"))
    
  }
  
  
  # create indexing list
  ls_metr_line = paste(ls_name, metr_name, sep = ",")
  write.table(ls_metr_line, file = "data/metrics/metric_tables/table_metr_ls_aggregation_index.csv",   
              row.names=F, col.names=F, append = T, quote = F)
  #write.table(ls_metr_line, file = "data/metrics/metric_maps/metr_index.csv",   
  #            row.names=F, col.names=F, append = T, quote = F)
  
}


# 3. moving window calculation.


