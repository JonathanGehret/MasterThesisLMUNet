library(raster)
library(reticulate)
np <- import("numpy")

#test_ls <- np$load("data/landscapes/npy_10000_random_ls/ls_0011.npy")
#max(test_ls)

#max(4,5)

n = 0
dir = "data/landscapes/npy_10000_random_ls/"
#dir = "data/landscapes/npy_10000_random_metr/"

for (i in 1:1000) {

  #adding zeroes for the name to automatically call it nicely
  i_len = nchar(as.character(i))
  i_just = paste(rep("0", 4 - i_len),  collapse = "") # rjust function fo some nice zeroes
  ls_dir = paste(dir, sep = "")
  file_name = paste(ls_dir, "ls_", i_just, i, ".npy", sep = "")
  
  #load file.npy i from dir
  test_ls <- np$load(file_name)
  
  # return maximum value of each landscape in folder
  n = max(n,max(test_ls))
  print(n)
}

#test_ls <- np$load("data/landscapes/npy_10000_random_ls/ls_0011.npy")


test_ls <- np$load("data/landscapes/npy_var_class_sidi_3_5_mov_win/ls_23509449.npy")
plot(raster(test_ls))

test_metr <- np$load("data/metrics/npy_var_class_sidi_3_5_mov_win/metr_23509449.npy")
plot(raster(test_metr))
