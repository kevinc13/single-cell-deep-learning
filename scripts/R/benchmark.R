# ------------------------------------------------------------------------------
# This script benchmarks the clustering performance of various 
# deep learning models and dimensionality reduction techniques
# Author: Kevin Chen 
# ------------------------------------------------------------------------------
base.dir <- "~/Documents/Research/XinghuaLuLab/single-cell-deep-learning"
setwd(paste(base.dir, "/scripts/R", sep=""))

library(data.table)
library(rjson)

source("evaluation.R")

# Configuration ----------------------------------------------------------------
experiment.dir <- "usokin/experiment_1c"
model.name <- "UsokinVAE_BestTotalLoss"
model.type <- "vae"
output.file <- "clustering_results_best_total_loss.txt"

latent.reps.col.start <- 2
latent.reps.col.end <- -2
labels.col <- "cell_type"

clust.alg <- "km"
clust.dist <- "euclidean"
max.k <- 11
force.k <- 4

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


summarizeModelSelectionExperiment(experiment.dir,
                                  model.name, model.type,
                                  latent.reps.col.start,
                                  latent.reps.col.end, labels.col,
                                  max.k, force.k,
                                  clust.alg=clust.alg, clust.dist=clust.dist,
                                  output.file=output.file)