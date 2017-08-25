# ------------------------------------------------------------------------------
# This script benchmarks the clustering performance of various 
# deep learning models and dimensionality reduction techniques
# Author: Kevin Chen 
# ------------------------------------------------------------------------------
base.dir <- "~/Documents/Research/XinghuaLuLab/single-cell-deep-learning"
setwd(paste(base_dir, "/scripts/R", sep="")

library(data.table)

source("evaluation.R")

# Configuration ----------------------------------------------------------------
experiment.dir <- "usokin/experiment_1c"
model.name <- "UsokinVAE_BestTotalLoss"
model.type <- "vae"

latent.reps.col.start <- 2
latent.reps.col.end <- -2
labels.col <- -2

clust.alg <- "km"
clust.dist <- "euclidean"
max.k <- 11
force.k <- 4

# Helpers
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
      plot.data, aes(x=epoch, y=loss, fill=type)) +
      geom_area(position="stack") +
      ggtitle("VAE Training Losses v. Epoch")
    ggsave("losses_v_epoch.png",
           plot=loss.v.epoch.plot, device="png", path = ".",
           width=10, height=6, units="in", dpi=300)
  }
}

summarizeModelSelectionExperiment <- function(experiment.dir,
                                              model.name, model.type,
                                              latent.reps.col.start,
                                              latent.reps.col.end,
                                              labels.col, max.k, force.k
                                              clust.alg="km",
                                              clust.dist="euclidean") {
  experiment.dir <- paste(base.dir, "/results/", experiment.dir, sep="")

  clust.results.df <- data.frame()
  for (exp.name in list.dirs(experiment.dir,
                             recursive=FALSE, full.names=TRUE)) {
    model.dir <- paste(experiment.dir, "/",
                       substring(exp.name, 2), "/",
                       model.name, sep="")
    
    if (model.type == "vae") {
      # Load latent representations
      df <- fread(paste(model.dir, "/latent_representations.txt", sep=""),
                  header=TRUE,
                  data.table=FALSE)

      latent.reps <- df[,latent.reps.col.start:(ncol(df) + latent.reps.col.end)]
      labels <- df[,labels.col]
      
      evaluation <- evaluateVAE(model.dir, latent.reps, labels,
                                clust.alg=clust.alg, clust.dist=clust.dist,
                                max.k=max.k, force.k=force.k)
      catln("Finished evaluating: ", exp)
    } else if (model.type == "ae") {
      stop("Haven't implemented this yet")
    } else {
      stop("Invalid model type, must be one of the following: 'vae', 'ae'")
    }

    clust.results.df <- rbind(clust.results.df, evaluation$clustering.results)
  }
}

