# generate diversity
# transform to .md? -> yess

library(NLMR)
#library(igraph)
library(landscapemetrics)
library(landscapetools)
library(raster)
library(ggplot2)
library(recolorize) # for reading in images/landscapes again
#library(imager)
#library(png)
#library(rgdal)
#library(sets)


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
    landscape = nlm_randomcluster(nrow = nrow,ncol = ncol,p = 0.5,
                                  ai = c(0.3,0.6,0.1), rescale=FALSE)
    #landscapes_midpoint_replacement[i] = nlm_mpd(ncol=ncol, nrow=nrow, roughness = 0.61)
    # plot landscapes without axes, title, legend
    rasterVis::gplot(landscape) +
      ggplot2::geom_tile(ggplot2::aes(fill = factor(value))) +
      theme_void() +  theme(legend.position="none") + coord_fixed()
    
    # save landscapes 128x128 pixels
    i_len = nchar(as.character(i))
    i_just = paste(rep("0", n_len - i_len),  collapse = "") # rjust function fo some nice zeroes
    ls_dir = paste("data/landscapes/", dir, sep = "")
    dir.create(file.path(ls_dir), showWarnings = FALSE) # creates image folder if doesnt exist
    ls_path = paste(ls_dir, "ls_", i_just, i, img_format, sep = "")
    ggsave(file = ls_path, height = nrow, width = ncol, units = "px")#, plot = landscape_test)
    
    #ggsave(filename = "land_example_lossy_07.jpeg", width = 128, height = 128, units = "px")#, plot = landscape_test)
    
    if (calc_metr == T) {
      generate_metrics(n = n, i = i, dir = dir, landscape = landscape)
    }

  }
}


# generate landscapes
generate_landscapes(s = 1, n = 1000, dir = "window_sidi/", calc_metr = T)

# 2. loop over all landscapes and calculate metric:

generate_metrics = function(n = 9999, i = 1, dir = "10000_random/", img_format = ".png", landscape = NA, read_img = F, lsm_metric = "lsm_l_sidi") {
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
    ls_png = readImage("test_images/land_example_loseless_07.png")
    # needs work here: images are 4D, what to do about this?

    plotImageArray(ls_png)
    
    plotImageArray(ls_png[,,2])
    
    png_1_matr = data.matrix(ls_png[,,1])
    png_2_matr = data.matrix(ls_png[,,2])
    png_3_matr = data.matrix(ls_png[,,3])
    full_matr = png_1_matr+png_2_matr+png_3_matr
    lsm_l_sidi(png_1_matr)
    lsm_l_sidi(png_2_matr)
    lsm_l_sidi(png_3_matr)
    
    unique(png_1_matr)
    
    number_of_classes = 3
    png_values = unique(png_1_matr)
    png_values = c(0.9921569, 0.9725490, 0.0000000) 
    number_of_classes = length(png_values)
    metric_map = raster(nrow = 128, ncol = 128, ext=extent(0, 128, 0, 128))
    
    for (class in number_of_classes) {
      metric_map@data@values[png_1_matr == png_values[class]] = class
      
    }
    
    metric_map = raster(nrow = 128, ncol = 128, ext=extent(0, 128, 0, 128))
    for (class in metric$class) {
      metr_value = metric[metric$class == class,]$value
      metric_map@data@values[landscape@data@values == class] = metr_value
    }
    lsm_l_pr(ls)
    ls_raster = as.raster(ls)
    lsm_c_ca(ls_png)
    ls_matr = data.matrix(ls_png)
    landscape = raster(nrow = 128, ncol = 128, ext=extent(0, 128, 0, 128))
    values(landscape) = ls_matr
    
    # read in .csv landscape and convert to raster layer
    # write as function? "csv_to_ls()"
    ls_df = read.table(paste(dir, ls_name, sep=""), sep = " ")
    ls_matr = data.matrix(ls_df)
    landscape = raster(nrow = 128, ncol = 128, ext=extent(0, 128, 0, 128))
    values(landscape) = ls_matr
    values(landscape) = data.matrix(ls)
    as.matrix(ls[0,1])
  }
  
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
  
  # plot metric without axes, title, legend
  rasterVis::gplot(metric$layer_1$lsm_l_sidi) +
    ggplot2::geom_tile(ggplot2::aes(fill = factor(value))) +
    theme_void() +  theme(legend.position="none") + coord_fixed()
  
  # save moving window metrics 128x128 pixels
  ggsave(file = metr_path, height = nrow, width = ncol, units = "px")#, plot = landscape_test)
  
  # create indexing list
  ls_metr_line = paste(ls_name, metr_name, sep = ",")
  metric_list_dir = paste("data/metric_list/", dir, sep = "")
  dir.create(file.path(metric_list_dir), showWarnings = FALSE) #create metric directory if doesnt exist
  write.table(ls_metr_line, file = paste(metric_list_dir, "metric_list.csv", sep = ""),
                row.names=F, col.names=F, append = T, quote = F)
}


# generate metrics for n things:
n = 5000
for (i in 3:n) {
  generate_metrics(n = n, i, dir = "testing_png/") 
}


# 3. moving window calculation.

window = matrix(1, nrow = 5, ncol = 5)
window_lsm_01 = window_lsm(landscape_test_01, window=window, what="lsm_l_sidi") #c("lsm_l_pr","lsm_l_joient"))
rasterVis::gplot(landscape_test_01) +
  ggplot2::geom_tile(ggplot2::aes(fill = factor(value))) +
  theme_void() +  theme(legend.position="none") + coord_fixed()
rasterVis::gplot(window_lsm_01$layer_1$lsm_l_sidi) +
  ggplot2::geom_tile(ggplot2::aes(fill = factor(value))) +
  theme_void() +  theme(legend.position="none") + coord_fixed()
ggsave(file = "metr_path_simpson_test.png", height = 128, width = 128, units = "px")#, plot = landscape_test)
