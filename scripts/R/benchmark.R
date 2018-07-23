# ------------------------------------------------------------------------------
# This script performs consensus clustering on 
# latent representations of cell expression profiles
# Author: Kevin Chen 
# ------------------------------------------------------------------------------
base.dir <- "~/Documents/Research/XinghuaLuLab/single-cell-deep-learning"
setwd(base.dir)

library(data.table)
library(rjson)
library(scater)

source("scripts/R/lib/clustering.R")
source("scripts/R/lib/utilities.R")

# Configuration ----------------------------------------------------------------
experiment.dir <- "results/pollen/vae_bn"
clust.alg <- "wardHC"
clust.dist <- "euclidean"
max.k <- 11
force.k <- 11

output.dir <- paste(experiment.dir, "/benchmark_results", sep="")
output.file <- paste(output.dir, "/consensus_wardHC_k11.txt", sep="")

latent.space.file <- "latent_representations.txt"
latent.reps.col.start <- 2
latent.reps.col.end <- -2
labels.col <- "cell_subtype"

exclude.dir <- NA
verbose <- FALSE

dir.create(output.dir, showWarnings=FALSE)

# Clustering -------------------------------------------------------------------
clust.results.df <- data.frame()
for (filepath in list.files(experiment.dir, 
                            pattern=latent.space.file, recursive=TRUE,
                            full.names=TRUE)) {
  if (!any(is.na(exclude.dir)) && length(grep(exclude.dir, filepath))) {
    next;
  }
  
  model.dir <- dirname(filepath)
  exp.dir <- dirname(model.dir)
  
  model.name <- basename(model.dir)
  exp.name <- basename(exp.dir)
  
  # Load latent representations
  df <- fread(filepath, header=TRUE, data.table=FALSE)
  latent.reps <- df[,latent.reps.col.start:(ncol(df) + latent.reps.col.end)]
  if (!any(is.na(labels.col))) {
    labels <- df[,labels.col] 
  } else {
    labels <- NA
  }
  
  # Load model configuration
  if (file.exists(paste(model.dir, "/config.json", sep=""))) {
    model.config <- fromJSON(file=paste(model.dir, "/config.json", sep=""))  
  }
  
  clust <- consensusClustering(latent.reps, labels=labels,
                               model.dir, max.k=max.k, force.k=force.k,
                               alg=clust.alg, dist=clust.dist,
                               plot=TRUE, colors=NA, verbose=verbose)
  catln("Finished clustering: ", exp.name, "/", model.name)
  
  if (!any(is.na(force.k))) {
    row <- data.frame("opt_k"=clust$opt.k,
                                    "pac_opt_k"=clust$pac[
                                      paste("k", clust$opt.k, sep="")],
                                    "force_k"=force.k,
                                    "pac_force_k"=clust$pac[
                                      paste("k", force.k, sep="")],
                                    "ari"=clust$ari)
  } else {
    row <- data.frame("opt_k"=clust$opt.k,
                                    "pac_opt_k"=clust$pac[
                                      paste("k", clust$opt.k, sep="")],
                                    "force_k"=force.k,
                                    "pac_force_k"=NA,
                                    "ari"=clust$ari)
  }
  
  rownames(row) <- exp.name
  
  if (length(grep("VAE", model.name))) {
    if (length(grep("BatchNormalization", model.config$encoder_layers))) {
      n_layers <- length(model.config$encoder_layers) / 2
    } else {
      n_layers <- length(model.config$encoder_layers)
    }
    model.info <- data.frame(model_name=model.config$name,
                             n_genes=model.config$input_shape[1],
                             n_layers=n_layers,
                             encoder_layers=paste(
                               model.config$encoder_layers, collapse="|"),
                             n_latent_dim=ncol(latent.reps),
                             optimizer=model.config$optimizer,
                             batch_size=model.config$batch_size) 
  } else if (length(grep("AE", model.name))) {
    model.info <- data.frame(model_name=model.config$name,
                             n_genes=model.config$input_shape[1],
                             n_layers=(length(
                               model.config$encoder_layers) - 1) / 2,
                             encoder_layers=paste(
                               model.config$encoder_layers[
                                 1:(length(model.config$encoder_layers) - 1)],
                               collapse="|"),
                             n_latent_dim=ncol(latent.reps),
                             optimizer=model.config$optimizer,
                             batch_size=model.config$batch_size)
  # } else if (grep("AAE", model.name)) {
  #   model.info <- data.frame(model_name=model.config$name,
  #                            n_genes=model.config$input_shape[1],
  #                            enc_layers=paste(
  #                              model.config$encoder_layers, collapse="|"),
  #                            disc_layers=paste(
  #                              model.config$discriminator_layers,
  #                              collapse="|"),
  #                            n_latent_dim=model.config$latent_size,
  #                            ae_optimizer=model.config$autoencoder_optimizer,
  #                            disc_optimizer=
  #                              model.config$discriminator_optimizer,
  #                            batch_size=model.config$batch_size)
  } else {
    if (length(grep("([0-9]+)g", model.name))) {
      n_genes <- as.numeric(
        gsub("([a-zA-Z_]+)([0-9]+)g(.*)", "\\2", model.name))
    } else if (length(grep("([0-9]+)g", exp.name))) {
      n_genes <- as.numeric(
        gsub("([a-zA-Z_]+)([0-9]+)g(.*)", "\\2", exp.name))
    } else {
      n_genes <- NA
    }
    
    model.info <- data.frame(model_name=model.name,
                             n_genes=n_genes,
                             n_layers=NA,
                             encoder_layers=NA,
                             n_latent_dim=ncol(latent.reps),
                             optimizer=NA,
                             batch_size=NA) 
  }
  
  row <- cbind(model.info, row)
  clust.results.df <- rbind(clust.results.df, row)
}

write.table(clust.results.df,
            file=output.file,
            row.names=FALSE, col.names=TRUE,
            sep="\t", quote=FALSE)

