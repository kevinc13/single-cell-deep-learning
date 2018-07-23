library(data.table)

source("scripts/R/lib/clustering.R")

loadLatentSpace <- function(model.dir, features.end.col, label.col) {
  latent.space <- fread(paste(model.dir, "/latent_representations.txt", sep=""),
                        header=TRUE, data.table=FALSE, sep="\t")
  rownames(latent.space) <- latent.space[,1]
  latent.space[,1] <- NULL
  features <- latent.space[,1:(ncol(latent.space) + features.end.col)]
  labels <- latent.space[,label.col]
  
  return(list(latent.space=latent.space,
              features=features,
              labels=labels))
}

clusterLatentSpace <- function(features, model.dir, max.k, force.k=NA,
                               alg="wardHC", dist="euclidean", subcluster=TRUE, 
                               clusters.output.file="consensus_clusters.txt",
                               verbose=FALSE) {
  clust.results <- consensusClustering(features, labels=NA, 
                                       base.dir=model.dir, 
                                       max.k, force.k=force.k, alg=alg, 
                                       dist=dist, plot=TRUE, verbose=verbose)
  clusters <- as.factor(clust.results$clusters)
  
  output.df <- as.data.frame(clusters)
  output.df <- cbind(names(clusters), output.df)
  colnames(output.df) <- c("cell_id", "consensus_cluster")
  write.table(output.df,
              file=paste(model.dir, "/", clusters.output.file, sep=""),
              quote=FALSE, sep="\t",
              row.names=FALSE, col.names=TRUE)
  
  return(clust.results)
}





