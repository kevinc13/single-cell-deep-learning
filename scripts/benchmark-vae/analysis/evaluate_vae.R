library(data.table)

# ---------- Configuration ---------- #

exp_name <- "train_usokin-100g-standardized-1layer-vae"
model_name <- "50_UsokinVAE_FINAL"

# ---------- Load Helper Files ---------- #
setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/scripts/benchmark-vae")
source("consensus_clustering.R")
source("tsne.R")

# ---------- Load Latent Representations ---------- #

project_dir <- "~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/results"
exp_dir <- paste(project_dir, "/", exp_name, sep="")
model_dir <- paste(exp_dir, "/", model_name, sep="")
setwd(model_dir)

df <- fread("latent_representations.txt",
                     header=TRUE,
                     data.table=FALSE)

latent_reps <- df[,2:(ncol(df)-2)]
cell_types <- df$cell_type
cell_subtypes <- df$cell_subtype

# ---------- Consensus Clustering ---------- #
params <- c("km_euclidean", "pam_euclidean", "pam_pearson")

clust_results_df <- data.frame()
for (param in params) {
    alg <- strsplit(param, "_")[[1]][1]
    dist <- strsplit(param, "_")[[1]][2]
    
    clust <- cluster_latent_reps(latent_reps, cell_types,
                                 model_dir, 4, force_opt_k=4,
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


# ---------- t-SNE (Barnes-Hut) ---------- #
# tsne <- tsne_latent_reps(latent_reps,
#                          cell_types,
#                          model_dir)


