library(scater)
library(data.table)

# ---------- Load Helper Files ---------- #
setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/scripts/benchmark-vae")
source("consensus_clustering.R")
source("tsne.R")

setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning")

# ---------- Run PCA ---------- #
usokin <- readRDS("data/Usokin/original/mouse_neuronal_cells.rds")
usokin.pca <- plotPCA(usokin, ncomponents=10,
                      colour_by="pca_major_types",
                      return_SCESet = TRUE,
                      scale_features=TRUE,
                      draw_plot=FALSE)

latent_reps <- reducedDimension(usokin.pca)
cell_types <- usokin$pca_major_types
cell_subtypes <- usokin$pca_all_neuronal_subtypes

params <- c("km_euclidean", "pam_euclidean", "pam_pearson")

clust_results_df <- data.frame()
for (param in params) {
    alg <- strsplit(param, "_")[[1]][1]
    dist <- strsplit(param, "_")[[1]][2]
    
    clust <- cluster_latent_reps(latent_reps, cell_types,
                                 "results/usokin_pca", 4, force_opt_k=4,
                                 clusterAlg=alg, dist=dist,
                                 plot=TRUE)
    
    row <- data.frame("opt_k"=clust$opt_k,
                      "pac_opt_k"=clust$pac[
                          paste("k", clust$opt_k, sep="")],
                      "pac_k4"=clust$pac["k4"],
                      "ari_k4"=clust$ari,
                      "acc_k4"=clust$acc)
    rownames(row) <- param
    clust_results_df <- rbind(clust_results_df, row)
}