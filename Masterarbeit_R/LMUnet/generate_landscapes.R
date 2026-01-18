# generate diversity
# transform to .md? -> yess

library(NLMR)
#library(igraph)
library(landscapemetrics)
library(landscapetools)
library(raster)
#library(rgdal)
#library(sets)


# 1. generate n + 1 landscapes, visualize them, save them

# number of landscapes
n = 10000

# image size 128
nrow = 128
ncol = 128

# landscape list
#landscape_list = list()
#landscapes_midpoint_replacement = list()

# remove starting from 0 and count from 1, as it's R? hmmm
# completeley random, simposon diversity von 1
#for (i in 1:n) {
for (i in 6760:n) {
  print(i)
  #landscape_list[i] = paste("random_landscape_", as.character(i), sep = "")
  #landscape_list[i] = nlm_randomcluster(
  landscape = nlm_randomcluster(nrow=nrow,ncol=ncol,p=0.5,
                                ai = c(0.3,0.6,0.1), rescale=FALSE)
  #landscapes_midpoint_replacement[i] = nlm_mpd(ncol=ncol, nrow=nrow, roughness = 0.61)
  write.table(as.array(landscape), 
              file = paste("data/landscapes/10000_random/ls_", i, ".csv", sep = ""),
              row.names=F, col.names=F)
}

# save those landscapes for future use
saveRDS(landscape_list, "data/10000_random_landscapes_01.RData")
#saveRDS(landscapes_midpoint_replacement, "data/landscapes_midponit_replacement_01.RData")
#landscapes_midpoint = readRDS("data/landscapes_midponit_replacement_01.RData")

#write.table(as.array(nlm_randomcluster(nrow=nrow,ncol=ncol,p=0.5,
#                                       ai = c(0.3,0.6,0.1), rescale=FALSE)), 
#            file = paste("data/landscapes/10000_random/ls_", 1, ".csv", sep = ""),
#            row.names=F, col.names=F)

# visualize these landscapes:
show_landscape(nlm_randomcluster(nrow=nrow,ncol=ncol,p=0.5,ai = c(0.3,0.6,0.1), rescale=FALSE))
test = nlm_randomcluster(nrow=nrow,ncol=ncol,p=0.5,ai = c(0.3,0.6,0.1), rescale=FALSE)
#spplot(landscape_list[[6]])
# show_landscape(landscapes_midpoint_replacement[[6]])

# 2. calculate metrics for these landscapes:
# change to  vectorization also, no need for for loops.
#ie:

metrics = lsm_c_ca(landscapes)

# metrics list
metrics_list = list()
metrics_list_l_msidi = list()

# loop over all landscapes and calculate metric:
# obsolete, use vectorization instead!
for (i in 1:n) {
  print(i)
  #landscape = landscape_list[[i+1]]
  landcscape = 
  metric = lsm_c_ca(landscape)
  #metrics_list[[i+1]] = lsm_l_msidi(landscape)
  #metrics_list[[i+1]] = lsm_l_core_cv(landscape)
  #landscape = landscapes_midpoint_replacement[[i+1]]
  #metrics_list_l_msidi[[i+1]] = lsm_l_msidi(landscape)
}

# save metrics for reuse
saveRDS(metrics_list, "data/random_l_msidi_01.RData")
#saveRDS(metrics_list_l_msidi, "data/landscapes_midpoint_replacement_01_l_msidi.RData")

#lsm_l_msidi(landscapes_midpoint_replacement[[5]])

# 2.5. read in landscapes from csv again and convert to raster layer

testing = read.table("data/landscapes/10000_random/ls_1.csv", sep = " ")
#testrast = raster("data/landscapes/10000_random/ls_1.csv", sep = " ")
#testcsv = read.csv("data/landscapes/10000_random/ls_1.csv", sep = " ")
#show_landscape(raster(testing))
#rasterize(testing)

#show_landscape(ololol)


test_matr = data.matrix(testing)
#example_landscape = landscape_list[[1]]
#show_landscape(example_landscape)

# vconvert dataframe to raster
r = raster(nrow = 128, ncol = 128, ext=extent(0, 128, 0, 128))
test_raster = r
values(r) = test_matr

show_landscape(r)

xyz = lsm_c_ca(r)



# 4. Write metrics to csv

asdf = c()            
asdf_02 = c()
# write to csv
for (i in 0:n) {
  print(metrics_list[[i+1]]$value)
  #print(metrics_list_l_msidi[[i+1]]$value)
  #asdf = append(asdf, metrics_list[[i+1]]$value)
  asdf_02 = append(asdf_02, metrics_list_l_msidi[[i+1]]$value)
}
write.csv(asdf, file = "data/metrics/random_metrics_01.csv")

metrics_list[[]]$value

# 3. write landscapes to csv
# (which one was now the final solution? write.table directly to csv? yeah, seems good!)

# load in randomly generated landscape list again
landscape_list = readRDS("data/random_landscapes_01.RData")
show_landscape(landscape_list[[6]])

# write raster files in .tif
# single one
writeRaster(landscapes_midpoint_replacement[[6]], file="data/images/test_image_raster.tif")

landscape_list[[1]]@data@values = as.integer(landscape_list[[1]]@data@values)
writeRaster(landscape_list[[1]],  file="data/images/randomly_generated/test_image_raster_02.jpeg")#, datatype="INT")
# all of them
for (i in 0:n) {
  #print(metrics_list[[i+1]]$value)
  #print(metrics_list_l_msidi[[i+1]]$value)
  writeRaster(landscapes_midpoint_replacement[[i+1]], file= paste("data/images/midpoint_replacement/img_", i, ".tif", sep = ""))
  #writeRaster(landscape_list[[i+1]], file= paste("data/images/img_", i, ".tif", sep = ""))
}

# use unique instead!
list_6_set = unique(landscape_list[[1]]@data@values)
#list_6_set = as.set(landscape_list[[1]]@data@values)
class(landscape_list)
typeof(landscape_list[[1]])      


testint = raster("data/images/randomly_generated/test_image_raster_02.grd")
spplot(testint)

png(filename = "data/images/randomly_generated/test_02.png", width=100, height=100)
spplot(testint)
dev.off()

tiff(filename = "data/images/randomly_generated/test.tif", )
spplot(testint)
dev.off()

plot(testint, legend=FALSE, )

library(terra)
terra::writeRaster(raster(landscape_list[[1]]), file="data/images/randomly_generated/asdfasdf.tif", overwrite = TRUE, datatype = "INT1U")
terra::writeRaster(raster(landscape_list[[1]]), file="data/images/randomly_generated/asdfasdf.tif", overwrite = TRUE, datatype = "INT1U")

raster::plot(testint, mar=c(0,0,0,0))

raster(landscape_list[[2]])

asdf = raster("data/images/randomly_generated/asdfasdf.tif")
plot(asdf)

plot(landscape_list[[1]]$clumps)

n = 100
for (i in 1:n) {
  write.table(x = landscape_list[[i]]@data@values, 
             file = paste("data/images/randomly_generated/image_values_", i, ".csv", sep = ""),
             row.names=F,col.names=F, eol = ",")
}


# raster to array, save array as .csv
dfuck = as.array(landscape_list[[1]])

n = 100
for (i in 1:n) {
  write.table(as.array(landscape_list[[i]]), 
  file = paste("data/images/randomly_generated/image_values_", i, ".csv", sep = ""),
  row.names=F, col.names=F)
}

# save landscape level metrics as csv in certain format
random_metrics = read.csv("data/metrics/random_metrics_01.csv", header = T)
xxxx = random_metrics$x
write.table(xxxx, file = "data/metrics/random_metrics_03.csv",   
            row.names=F, col.names=F)

# table sollte am Ende zwei spalten enthalten, komma separiert (", "): 
#links die Namen der Landschaften, 
#rechts die Metrik(en)

write.table()

# to 2.: calculate metrics (here: total class area on class level)

total_class_area_landscape_list = lsm_c_ca(landscape_list)
length(landscape_list)
length(total_class_area_landscape_list)

# 5. create metric label map for the calculated metric for class level for semantic segmenation

landscape_number = 1
class_number = 1
test_metrics = total_class_area_landscape_list

rm(landscape_number)
rm(class_number)
rm(test_value)

landscape_list[landscape_list[[landscape_number]]@data@values == class_number] = test_metrics[test_metrics$layer == landscape_number][test_metrics$class == class_number]

test_metrics$layer == 1
test_metrics[test_metrics$layer == landscape_number,][test_metrics$class == class_number]

test_metrics
test_metrics[test_metrics$layer == landscape_number,]$class == class_number

test_metrics$layer == landscape_number

# gets the single metric "value" for class "class_number" of landscape "landscape_number" 
#test_value = test_metrics[test_metrics$layer == landscape_number,][test_metrics[test_metrics$layer == landscape_number,]$class == class_number,]$value
test_value = test_metrics[test_metrics$layer == landscape_number & test_metrics$class == class_number,]$value

#testing vectorization on metrics
landscape_numbers = test_metrics$layer
landscape_numbers_unique = unique(test_metrics$layer)
class_numbers = test_metrics$class
class_numbers_unique = unique(test_metrics$class)
# is this possible in vectorization? don't I need a nested loop here?
# let's do nested loop for now, vectorize later
# do this right as the metrics are calculated? yeah, seems useful!
for (landscape_number in landscape_numbers_unique) {
  for (class_number in class_numbers_unique) {
    test_value = test_metrics[test_metrics$layer == landscape_number & test_metrics$class == class_number,]$value
    test_landscapes_02[[landscape_number]]@data@values[test_landscapes_02[[landscape_number]]@data@values == class_number] = test_value
  }
}


test_values_01 = test_metrics[test_metrics$layer == landscape_numbers & test_metrics$class == class_numbers,]$value

# now replace that in the original file
test_landscapes = landscape_list
test_landscapes[[landscape_number]]@data@values[test_landscapes[[landscape_number]]@data@values == class_number] = test_value
# it works!

#test_landscapes[[1]]@data@values == class_number = test_value

test_landscapes[[1]]@data@values

# compare landscapes 02 and 03
test_landscapes_02 = landscape_list
test_landscapes_03 = landscape_list


# write class metrics to files
n = 100
for (i in 1:n) {
  write.table(as.array(test_landscapes_02[[i]]), 
              file = paste("data/metrics/randomly_generated_patch_area_class/metric_values_", i, ".csv", sep = ""),
              row.names=F, col.names=F)
}

# 6. write metrics_list including landscape naems and metric names (patch, class)/ metrics (landscape)


#random_metrics = read.csv("data/metrics/random_metrics_03.csv", header = F) # if landscape levevl
# either create file names from saved data, or retrieve them eralier, eg when saving to file

# append = T for writing new lines to same file
# quote = FF to remove parentheses ""
for (i in 1:n) {
    random_metric = paste("metric_values_", i, ".csv", sep = "") # metr_values?
    random_landscape = paste("image_values_", i, ".csv", sep = "") # should be ls_values?
    ls_metr_line = paste(random_landscape, random_metric, sep = ",")
    write.table(ls_metr_line, file = "data/metrics_list_02.csv",   
                row.names=F, col.names=F, append=T, quote=F)
}


# 7. Combine reading in landscapes again, transforming to rasterlayer, calculation of metric, putting metric into raster layer and writing that file to file
# combining 2, 2.5, 5 and 6


# loop over all landscapes and calculate metric:
# obsolete, use vectorization instead! (unless being read in individually?!?)
# write as function
for (i in 1:n) {
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
  
  # caculate metric and write to csv
  metric = lsm_c_ca(landscape)
  write.csv(metric, file = paste("data/metrics/metric_tables/", metr_name, sep=""))

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
  
  # create indexing list
  ls_metr_line = paste(ls_name, metr_name, sep = ",")
  write.table(ls_metr_line, file = "data/metrics/metric_tables/table_metr_index.csv",   
              row.names=F, col.names=F, append = T, quote = F)
  write.table(ls_metr_line, file = "data/metrics/metric_maps/metr_index.csv",   
              row.names=F, col.names=F, append = T, quote = F)

}