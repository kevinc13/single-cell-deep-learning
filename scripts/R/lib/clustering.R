# ------------------------------------------------------------------------------
# Provides a set of functions for evaluating latent representations
# Author: Kevin Chen
# ------------------------------------------------------------------------------

library(data.table)
library(caret)
library(ggplot2)
library(ConsensusClusterPlus)
library(pheatmap)
library(RColorBrewer)
library(Rtsne)
library(mclust)

source(paste("/Users/kevin/Documents/Research/XinghuaLuLab/",
             "single-cell-deep-learning/",
             "scripts/R/lib/utilities.R", sep=""))

wardHC <- function(d, k) {
  results <- hclust(d, method="ward.D2")
  return(cutree(results, k))
}

consensusClustering <- function(x, labels=NA, base.dir,
                                max.k, force.k=NA,
                                alg="km", dist="euclidean",
                                plot=TRUE, colors=NA, verbose=TRUE) {
    # Setup results directory
    results.dir <- paste(base.dir, "/consensus_clustering_",
                         paste(alg, dist, sep="_"),
                         "_maxk", max.k, sep="")
    dir.create(results.dir, showWarnings=FALSE)
    results.file <- paste(results.dir,
                          "/consensus_clustering_results.Rdata", sep="")
    
    # print_divider("=")
    catln("Consensus Clustering")
    catln(dim(x)[1], " samples | ", dim(x)[2], " features")
    catln("Parameters:")
    catln("Max K = ", max.k, " | Cluster Alg = ", alg)
    catln("Distance = ", dist)
    print_divider("-")
    
    # Run clustering, if no previous clustering results exist
    if (!file.exists(results.file)) {
      consensus.clustering.results <- ConsensusClusterPlus(
        t(as.matrix(x)),
        maxK=max.k, reps=100,
        clusterAlg=alg, distance=dist,
        plot="pdf", innerLinkage="ward.D2",
        finalLinkage="ward.D2",
        title=results.dir, verbose=verbose)
      save(consensus.clustering.results, file=results.file)
      catln("Finished clustering")
    } else {
      catln("Loading previous consensus clustering results...")
      load(results.file)
    }
    print_divider("-")

    # Calculate PACs and optimal K #
    catln("Calculating PACs and optimal K...")
    k.vec <- 2:max.k
    # Threshold defining the intermediate sub-interval
    x1 <- 0.1
    x2 <- 0.9 
    PAC <- rep(NA, length(k.vec)) 
    names(PAC) <- paste("k", k.vec, sep="") # from 2 to maxK
    
    for(i in k.vec){
      m <- consensus.clustering.results[[i]]$consensusMatrix
      fn <- ecdf(m[lower.tri(m)])
      PAC[i - 1] <- fn(x2) - fn(x1)
    } 
    
    # The optimal K
    opt.k = k.vec[which.min(PAC)]
    catln("PAC:")
    print(PAC)
    cat("\n")
    catln("Optimal K = ", opt.k)
    if (!any(is.na(force.k))) {
      opt.results <- consensus.clustering.results[[force.k]]
      catln("Forcing optimal K = ", force.k)
    } else {
      opt.results <- consensus.clustering.results[[opt.k]]
    }
    print_divider("-")

    hc.order <- opt.results$consensusTree$order
    clust.assignments <- opt.results$consensusClass

    # Calculate clustering performance metrics
    if (!any(is.na(labels))) {
      rownames(opt.results$consensusMatrix) <- names(labels)
      colnames(opt.results$consensusMatrix) <- names(labels)
      
      confusion.mat <- table(clust.assignments, labels)
  
      ari <- adjustedRandIndex(clust.assignments, labels)
      catln("Adjusted Rand Index: ", ari)
      
      catln("Confusion Matrix:")
      print(confusion.mat)
      print_divider("-")
    }
    
    # Plot clustering heatmap
    if (plot && !file.exists(paste(results.dir, "/heatmap.png", sep=""))) {
        cat("Plotting heatmap...")
        
        # If the consensus matrix has no column names, set them to index
        if (is.null(colnames(opt.results$consensusMatrix))) {
          colnames(opt.results$consensusMatrix) <- 
            1:ncol(opt.results$consensusMatrix)
        }
        
        plot.mat <- opt.results$consensusMatrix[rev(hc.order),]

        if (!any(is.na(labels))) {
          names(labels) <- colnames(plot.mat)

          annotations <- data.frame(Cluster = labels)
          if (any(is.na(colors))) {
            cluster.colors <- brewer.pal(
                length(unique(labels)), "Set3")
            names(cluster.colors) <- unique(labels)
          } else {
            cluster.colors <- colors
          }
          annotation.colors <- list(Cluster = cluster.colors)
        }

        png(filename=paste(results.dir, "/heatmap.png", sep=""),
            width=11, height=10, units="in",
            pointsize=4, res=300)
        
        pheatmap.args <- list(plot.mat,
                              cluster_cols=opt.results$consensusTree,
                              cluster_rows=FALSE,
                              clustering_distance_rows=NA,
                              clustering_distance_cols=NA,
                              clustering_method=NA,
                              show_rownames=FALSE,
                              show_colnames=FALSE,
                              color=colorRampPalette(
                                c("#ffffff", "#2185c5"))(10),
                              border_color=NA,
                              fontsize=18)

        if (!any(is.na(labels))) {
          pheatmap.args$annotation_col <- annotations
          pheatmap.args$annotation_colors <- annotation.colors
          pheatmap.args$annotation_names_col <- FALSE
        }

        do.call(pheatmap, pheatmap.args)

        dev.off()
        catln("done")
    }
    
    print_divider("=")
    
    return.list <- list(results=consensus.clustering.results,
                        opt.results=opt.results,
                        clusters=clust.assignments,
                        pac=PAC,
                        opt.k=opt.k,
                        force.k=force.k)

    if (!any(is.na(labels))) {
      return.list$ari <- ari
    }
    
    return(return.list)
}

tsne <- function(x, labels=NA, base.dir, 
                 plot=TRUE, pca=TRUE, 
                 file="tSNE_plot.png", perplexity=30) {
    results.dir <- paste(base.dir, "/tSNE", sep="")
    results.file <- paste(results.dir, "/tSNE_results.Rdata", sep="")
    
    dir.create(results.dir, showWarnings=FALSE)
    
    # print_divider("=")
    cat("t-SNE: ")
    catln(dim(x)[1], " samples | ", dim(x)[2], " features")
    print_divider("-")
    
    # ---------- Run t-SNE ---------- #
    if (!file.exists(results.file)) {
      tsne.results <- Rtsne(x, dims = 2, perplexity=perplexity,
                            verbose=TRUE, pca=pca)
      save(tsne.results, file=results.file)
      catln("Finished running t-SNE (Barnes Hut)")
    } else {
      catln("Loading previous t-SNE results...")
      load(results.file)
    }
    
    projections <- data.frame(tsne.results$Y)
    colnames(projections) <- c("dim1", "dim2")
    
    # ---------- Plot ---------- #
    if (plot && !file.exists(paste(results.dir, "/", file, sep=""))) {
      print_divider("-")
      cat("Plotting t-SNE projections...")
      if (!any(is.na(labels))) {
        tsne.plot <- ggplot(projections, aes(x=dim1, y=dim2, color=labels))  
      } else {
        tsne.plot <- ggplot(projections, aes(x=dim1, y=dim2))
      }
      tsne.plot <- tsne.plot + geom_point() + ggtitle("t-SNE Plot")
      ggsave(file, plot=tsne.plot,
             device="png", path=results.dir,
             width=9, height=8, units="in", dpi=300)
      catln("done")
    }
    
    print_divider("=")
    
    return(list(results=tsne.results,
                projections=projections))
}





