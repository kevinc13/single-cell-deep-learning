library(data.table)

# ---------- Configuration ---------- #

exp_name <- "train_usokin-100g-standardized-1layer-vae"
model_name <- "50_UsokinVAE_FINAL"

# ---------- Load Helper Files ---------- #
setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/scripts/R")
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
clust <- cluster_latent_reps(latent_reps,
                             cell_types,
                             model_dir,
                             4,
                             clusterAlg="km",
                             force_opt_k=4)

# ---------- t-SNE (Barnes-Hut) ---------- #
tsne <- tsne_latent_reps(latent_reps,
                         cell_types,
                         model_dir)


