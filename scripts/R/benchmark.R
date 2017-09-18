# ------------------------------------------------------------------------------
# This script benchmarks the clustering performance of various 
# deep learning models and dimensionality reduction techniques
# Author: Kevin Chen 
# ------------------------------------------------------------------------------
base.dir <- "~/Documents/Research/XinghuaLuLab/single-cell-deep-learning"
setwd(paste(base.dir, "/scripts/R", sep=""))

library(data.table)
library(rjson)
library(scater)

source("evaluation.R")

# Configuration ----------------------------------------------------------------
# dataset.dir <- "Usokin/processed/usokin.100g.standardized.txt"
# pca.ntop <- 500
# pca.ncomponents <- 10
# results.dir <- "usokin/pca_100g"

experiment.dir <- "pbmc/vae"
model.name <- "PBMCAE_FINAL"
model.type <- "ae"
output.file <- "clustering_results_forcek4.txt"

latent.reps.col.start <- 2
latent.reps.col.end <- -1
labels.col <- "cell_type"

clust.alg <- "km"
clust.dist <- "euclidean"
max.k <- 7
force.k <- 7

plotVAELosses <- function(model.dir) {
  setwd(model.dir)
  
  # Load training log
  train.log <- fread("training.log", header=TRUE, data.table=FALSE)
  
  # Create loss v. epoch plot (if one doesn't exist already)
  if (!file.exists("losses_v_epoch.png")) {
    plot.data <- data.frame(cbind(train.log$reconstruction_loss,
                                  train.log$kl_divergence_loss))
    colnames(plot.data) <- c("recon_loss", "kl_loss")
    plot.data <- stack(plot.data)
    plot.data <- cbind(rep(train.log$epoch, 2), plot.data)
    colnames(plot.data) <- c("epoch", "loss", "type")
    
    loss.v.epoch.plot <- ggplot(
      plot.data, aes(x=epoch, y=loss, color=type)) +
      geom_line() +
      ggtitle("VAE Training Losses v. Epoch")
    ggsave("losses_v_epoch.png",
           plot=loss.v.epoch.plot, device="png", path = ".",
           width=10, height=6, units="in", dpi=300)
  }
}

benchmarkModelSelectionExperiment <- function(experiment.dir,
                                              model.name, model.type,
                                              latent.reps.col.start,
                                              latent.reps.col.end, labels.col,
                                              max.k, force.k,
                                              clust.alg="km",
                                              clust.dist="euclidean",
                                              output.file="clustering_results.txt") {
  experiment.dir <- paste(base.dir, "/results/", experiment.dir, sep="")
  
  clust.results.df <- data.frame()
  for (exp.name in list.dirs(experiment.dir,
                             recursive=FALSE, full.names=FALSE)) {
    model.dir <- paste(experiment.dir, "/",
                       exp.name, "/",
                       model.name, sep="")
    
    # Load latent representations
    df <- fread(paste(model.dir, "/latent_representations.txt", sep=""),
                header=TRUE,
                data.table=FALSE)
    latent.reps <- df[,latent.reps.col.start:(ncol(df) + latent.reps.col.end)]
    labels <- df[,labels.col]
    
    # Load model configuration
    model.config <- fromJSON(file=paste(model.dir, "/config.json", sep=""))
    
    evaluation <- evaluateLatentRepresentations(model.dir,
                                                latent.reps, labels,
                                                clust.alg, clust.dist,
                                                max.k, force.k=force.k)
    catln("Finished evaluating: ", exp.name)
    
    plotVAELosses(model.dir)
    
    rownames(evaluation$clustering.results)[1] <- exp.name
    row <- evaluation$clustering.results
    
    if (model.type == "vae") {
      model.info <- data.frame(model_name=model.config$name,
                               n_genes=model.config$input_size,
                               n_layers=length(model.config$encoder_layers) / 2,
                               encoder_layers=paste(
                                 model.config$encoder_layers, collapse="|"),
                               n_latent_dim=model.config$latent_size,
                               optimizer=model.config$optimizer,
                               batch_size=model.config$batch_size) 
    } else if (model.type == "ae") {
      model.info <- data.frame(model_name=model.config$name,
                               n_genes=model.config$input_size,
                               n_layers=(length(
                                 model.config$encoder_layers) - 1) / 2,
                               encoder_layers=paste(
                                 model.config$encoder_layers[
                                   1:(length(model.config$encoder_layers) - 1)],
                                 collapse="|"),
                               n_latent_dim=strsplit(
                                 tail(model.config$encoder_layers, n=1),
                                 ":")[[1]][2],
                               optimizer=model.config$optimizer,
                               batch_size=model.config$batch_size)
    } else {
      stop("Invalid model type; must be either 'vae' or 'ae'")
    }
    
    row <- cbind(model.info, row)
    clust.results.df <- rbind(clust.results.df, row)
  }
  
  if (!file.exists(paste(experiment.dir, "/", output.file, sep=""))) {
    write.table(clust.results.df,
                file=paste(experiment.dir, "/", output.file, sep=""),
                row.names=FALSE, col.names=TRUE,
                sep="\t", quote=FALSE)
  }
}

# benchmarkPCA <- function(dataset.file, labels.col, output.dir,
#                          max.k, force.k=NA, clust.alg="km", clust.dist="euclidean",
#                          ncomponents=10, ntop=500) {
#   if (!dir.exists(output.dir)) {
#     dir.create(output.dir, recursive=TRUE)
#   }
#   
#   dataset <- readRDS(dataset.file)
#   labels <- pData(dataset)[,labels.col]
#   
#   pca <- plotPCA(dataset, ncomponents=ncomponents,
#                  return_SCESet=TRUE, scale_features=TRUE,
#                  draw_plot=FALSE, ntop=ntop)
#   latent.reps <- reducedDimension(pca)
#   
#   evaluation <- evaluateLatentRepresentations(output.dir,
#                                               latent.reps, labels,
#                                               clust.alg, clust.dist,
#                                               max.k, force.k=force.k)
#   catln("Finished evaluating PCA")
#   
#   if (is.na(force.k)) {
#     output.file <- paste("clustering_results_maxk", max.k, ".txt", sep="")
#   } else {
#     output.file <- paste("clustering_results_maxk", max.k,
#                          "_forcek", force.k, ".txt", sep="")
#   }
#   
#   write.table(evaluation$clustering.results,
#               file=paste(output.dir, "/", output.file, sep=""),
#               sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)
# }

benchmarkModelSelectionExperiment(experiment.dir,
                                  model.name, model.type,
                                  latent.reps.col.start,
                                  latent.reps.col.end, labels.col,
                                  max.k, force.k,
                                  clust.alg=clust.alg, clust.dist=clust.dist,
                                  output.file=output.file)