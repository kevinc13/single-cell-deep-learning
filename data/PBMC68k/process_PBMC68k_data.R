library(Matrix)

setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/data/PBMC68k")
pbmc_data <- readRDS("PBMC68k.rds")
pbmc_mat <- pbmc_data$all_data[[1]]$hg19$mat