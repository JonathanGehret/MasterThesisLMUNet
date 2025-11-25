library(furrr)
library(tidyverse)
future::plan(multisession) # parallel computing

n <- 10
ai1 <- runif(n, min = 0.1, max = 0.9)
ai2 <- runif(1, min = ai1[1], max = 0.99)
for (i in 2:n) {
  ai2 <- c(ai2, runif(1, min = ai1[i], max = 0.99))
}
p <- runif(n, min = 0.1, max = 0.9)

generate_landscape <- function(p, ai1, ai2) {
  nlm_randomcluster(ncol = 128, nrow = 128,
                    resolution = 1, p = p,
                    ai = c(ai1, ai2),
                    neighbourhood = 4,
                    rescale = FALSE)
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

