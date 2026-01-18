library(NLMR)
library(igraph)
library(landscapetools)

# set seed for reproducability
set.seed(5420)

# set random image size
nrow = 128
ncol = 128

# example from https://github.com/ropensci/NLMR/


# calculate example NLMs
random_cluster = nlm_randomcluster(
  nrow=100,ncol=100,p=0.5,ai = c(0.3,0.6,0.1), rescale=FALSE)
random_cluster_2 = nlm_randomcluster(
  nrow=100,ncol=100,p=0.5,ai = c(0.3,0.6,0.1), rescale=FALSE)
random_cluster_3 = nlm_randomcluster(
  nrow=100,ncol=100,p=0.5,ai = c(0.3,0.6,0.1), rescale=FALSE)
random_cluster_4 = nlm_randomcluster(
  nrow=100,ncol=100,p=0.5,ai = c(0.3,0.6,0.1), rescale=FALSE)

random_curdling = nlm_curds(curds=c(0.5,0.3,0.6), recursion_steps = c(32,6,2))

midpoint_displacement = nlm_mpd(ncol=100, nrow = 100, roughness = 0.61)

gaussian = nlm_gaussianfield(nrow = nrow, ncol = ncol)

# visualize
par(mfrow = c(2, 2))
show_landscape(random_cluster)
show_landscape(random_cluster_2)
show_landscape(random_cluster_3)
show_landscape(random_cluster_4)
show_landscape(random_curdling)
show_landscape(midpoint_displacement)
