library(furrr)
library(tidyverse)
future::plan(multisession) # parallel computing

n <- 10   # number of loops
ai1 <- runif(n, min = 0.1, max = 0.9)   #vector of decimal numbers length n
ai2 <- runif(1, min = ai1[1], max = 0.99) # one value between lowest value of ai1 and 1
for (i in 2:n) {
  ai2 <- c(ai2, runif(1, min = ai1[i], max = 0.99)) #update ai2 with more numbers like previous limitations
}
p <- runif(n, min = 0.1, max = 0.9) # nochmal n numbers, genau wie ai1 generiert

generate_landscape <- function(p, ai1, ai2) {  # funktion um Landschaften zu generieren.
  nlm_randomcluster(ncol = 128, nrow = 128, # based on Saura & Martínez-Millán (2000)?
                    resolution = 1,  
                    p = p,    # proportion of elements randomly selected from clusters
                    ai = c(ai1, ai2), # Vector with the cluster type distribution (percentages of occupancy). This directly controls the number of types via the given length.
                    neighbourhood = 4, # Clusters are defined using a set of neighbourhood structures, 4 (Rook's or von Neumann neighbourhood) (default), 8 (Queen's or Moore neighbourhood).
                    rescale = FALSE) # reascale between 0 and 1, irrelevant as valueas already there
}

generate_gradient <- function() {
  nlm_edgegradient(ncol = 128, nrow = 128,
                   resolution = 1,
                   rescale = FALSE)
}

param_df = tibble(p = p, ai1 = ai1, ai2 = ai2)

nlm_randomcluster_landscapes <- param_df |> 
  furrr::future_pmap(generate_landscape, .options = furrr_options(seed = TRUE))

landscapetools::show_landscape(nlm_randomcluster_landscapes |> raster::stack())

nlm_randomcluster_gradient_landscapes <- sample(nlm_randomcluster_landscapes, size = 10) |> future_map(.f = function(ls) {
  ls + generate_gradient()
}, .options = furrr_options(seed = TRUE))

landscapetools::show_landscape(nlm_randomcluster_gradient_landscapes |> raster::stack())

