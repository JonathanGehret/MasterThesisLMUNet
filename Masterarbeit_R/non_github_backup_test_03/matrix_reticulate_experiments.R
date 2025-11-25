

# X. matrix experiments

# prepare landscape for metric calculation
landscape = np$load("data/landscapes/testing_reticulate/ls_1.npy")
ls_as_matrix = as.matrix(landscape)
ls_raster = raster(ls_as_matrix)
ls_raster_2 = raster(landscape)

# calculate moving window metric
window = matrix(1, nrow = 5, ncol = 5)
lsm_metric = "lsm_l_sidi"
metric = window_lsm(ls_raster, window = window, what = lsm_metric) # simpsons diversity index, #1 for ls lvl

# convert to matrix as np cant work with raster

metric_array = as.matrix(metric$layer_1$lsm_l_sidi)

# Error in py_call_impl(callable, dots$args, dots$keywords) : 
# Matrix type cannot be converted to python (only integer, numeric, complex, logical, and character matrixes can be converted


# save metric matrix as numpy array
np$save("data/metrics/testing_reticulate/met_1.npy", metric_array)


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
