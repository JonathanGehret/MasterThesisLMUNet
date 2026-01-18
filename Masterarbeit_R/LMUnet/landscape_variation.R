
nrow = 128
ncol = 128

landscape = nlm_randomcluster(nrow = nrow,ncol = ncol,p = 0.5, ai = c(0.3,0.6,0.1), rescale=FALSE)
landscape_0 = nlm_randomcluster(nrow = nrow,ncol = ncol,p = 0, ai = c(0.3,0.6,0.1), rescale=FALSE)
landscape_1 = nlm_randomcluster(nrow = nrow,ncol = ncol,p = 1, ai = c(0.3,0.6,0.1), rescale=FALSE)
landscape_01 = nlm_randomcluster(nrow = nrow,ncol = ncol,p = 0.1, ai = c(0.3,0.6,0.1), rescale=FALSE)
landscape_09 = nlm_randomcluster(nrow = nrow,ncol = ncol,p = 0.9, ai = c(0.3,0.6,0.1), rescale=FALSE)
landscape_08 = nlm_randomcluster(nrow = nrow,ncol = ncol,p = 0.8, ai = c(0.3,0.6,0.1), rescale=FALSE)
landscape_06 = nlm_randomcluster(nrow = nrow,ncol = ncol,p = 0.6, ai = c(0.3,0.6,0.1), rescale=FALSE)
landscape_07 = nlm_randomcluster(nrow = nrow,ncol = ncol,p = 0.7, ai = c(0.3,0.6,0.1), rescale=FALSE)
landscape_p = nlm_randomcluster(nrow = nrow,ncol = ncol,p = p_value, ai = c(0.3,0.6,0.1), rescale=FALSE)


plot(landscape)
plot(landscape_1)
plot(landscape_01)
plot(landscape_09)
plot(landscape_08)
plot(landscape_06)
plot(landscape_07)
plot(landscape_p)

# p from 0

x = runif(3)

x[1]/sum(x) + x[2]/sum(x) + x[3]/sum(x) 

x[1]/sum(x)
x[2]/sum(x)
x[3]/sum(x) 

# generate number of classes 3 <= # <= 6
n_c = floor(runif(1, min=3, max=7))

# generate n_c ai values between 0 and 1, adding up to 1
ai_values = runif(n_c)
ai_normalized = ai_values / sum(ai_values)

# generate p values between 0.1 and 0.6
p_value = runif(1, min=0.1, max=0.6)
