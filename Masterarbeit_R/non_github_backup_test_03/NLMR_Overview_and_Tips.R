# following:
# https://ropensci.github.io/NLMR/articles/articles/overview_tips.html

#library(landscapemetrics)
library(landscapetools)
#library(igraph)
library(NLMR)
library(raster)
library(dplyr)


# selection of possible merges
# 1.
edge_nlm <- nlm_edgegradient(100, 100)
distance_nlm <- nlm_distancegradient(100, 100, origin = c(20, 20,10, 10))
random_nlm <- nlm_random(100, 100)

# 2.
gauss_nlm <- nlm_gaussianfield(100, 100)
rectan_nlm <- nlm_randomrectangularcluster(100, 100, maxl = 30, minl = 10)

# 3.
mosaic_nlm <- nlm_mosaicfield(100, 100)

# 4.
planar_nlm <- nlm_planargradient(100, 100)
tess_nlm <- nlm_mosaictess(100, 100, germs = 200)

# plot it
landscapetools::show_landscape(list("a" = landscapetools::util_merge(edge_nlm, list(distance_nlm, random_nlm)),
                                    "b" = landscapetools::util_merge(gauss_nlm, rectan_nlm),
                                    "c" = landscapetools::util_merge(mosaic_nlm, list(random_nlm)),
                                    "d" = landscapetools::util_merge(planar_nlm, list(distance_nlm, tess_nlm))))

# counting cells
counting_cells_landscape = nlm_curds(curds = c(0.5, 0.3, 0.6),
                                     recursion_steps = c(32, 6, 2),
                                     wheyes = c(0.1, 0.05, 0.2))
#show_landscape(counting_cells_landscape) discret value error
spplot(counting_cells_landscape) # this works for plotting. weird!

counting_cells_landscape %>%
  freq()

freq(counting_cells_landscape)
