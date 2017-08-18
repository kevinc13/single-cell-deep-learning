library(data.table)
library(ConsensusClusterPlus)
library(pheatmap)
library(RColorBrewer)
library(caret)

source("utils.R")

cluster_latent_reps <- function(x, labels, base_dir,
                                maxK, force_opt_k=NA,
                                clusterAlg="pam",
                                dist="euclidean",
                                plot=TRUE, colors=NA) {
    results_dir <- paste(base_dir,
                         "/ConsensusClustering_",
                         clusterAlg, "_", dist, sep="")
    dir.create(results_dir, showWarnings=FALSE)
    results_filepath <- paste(results_dir, "/cluster_results.Rdata", sep="")
    
    print_divider("=")
    catln("Consensus Clustering")
    catln(dim(x)[1], " samples | ", dim(x)[2], " features")
    catln("Parameters:")
    catln("Max K = ", maxK, " | Cluster Alg = ", clusterAlg)
    catln("Distance = ", dist)
    print_divider("-")
    
    # ---------- Consensus Clustering ---------- #
    if (!file.exists(results_filepath)) {
        cluster_results <- ConsensusClusterPlus(t(as.matrix(x)),
                                                maxK=maxK, reps=100,
                                                clusterAlg=clusterAlg,
                                                distance=dist,
                                                plot="pdf",
                                                title=paste(base_dir, 
                                                            "/ConsensusClustering_",
                                                            clusterAlg, "_",
                                                            dist,
                                                            sep=""))
        save(cluster_results, file=results_filepath)
        catln("Finished clustering")
    } else {
        catln("Loading previous consensus clustering results...")
        load(results_filepath)
    }
    print_divider("-")

    # ---------- Calculate PACs and optimal K ---------- #
    catln("Calculating PACs and optimal K...")
    Kvec = 2:maxK
    x1 = 0.1; x2 = 0.9 # threshold defining the intermediate sub-interval
    PAC = rep(NA,length(Kvec)) 
    names(PAC) = paste("k",Kvec,sep="") # from 2 to maxK
    
    for(i in Kvec){
        M = cluster_results[[i]]$consensusMatrix
        Fn = ecdf(M[lower.tri(M)])
        PAC[i-1] = Fn(x2) - Fn(x1)
    } 
    
    # The optimal K
    optK = Kvec[which.min(PAC)]
    catln("PAC:")
    print(PAC)
    cat("\n")
    catln("Optimal K = ", optK)
    if (!is.na(force_opt_k)) {
        best_results <- cluster_results[[force_opt_k]]
        catln("Forcing optimal K = ", force_opt_k)
    } else {
        best_results <- cluster_results[[optK]]
    }
    print_divider("-")
    
    # ---------- Evaluate Clustering Performance ---------- #
    rownames(best_results$consensusMatrix) <- names(labels)
    colnames(best_results$consensusMatrix) <- names(labels)
    hc_order <- best_results$consensusTree$order
    
    cluster_assignments <- best_results$consensusClass
    confusion_mat <- table(cluster_assignments, labels)
    
    library(mclust)
    ari <- adjustedRandIndex(cluster_assignments, labels)
    catln("Adjusted Rand Index: ", ari)
    
    
    acc <- sum(apply(confusion_mat, 2, max)) / sum(confusion_mat)
    catln("Accuracy: ", acc)
    
    catln("Confusion Matrix:")
    print(confusion_mat)
    print_divider("-")
    
    # ---------- Plot ---------- #
    if (plot) {
        cat("Plotting heatmap...")
        
        # If the consensus matrix has no column names, set them to index
        if (is.null(colnames(best_results$consensusMatrix))) {
            colnames(best_results$consensusMatrix) <-
                1:ncol(best_results$consensusMatrix)    
        }
        
        plot_mat <- best_results$consensusMatrix[rev(hc_order),]
        names(labels) <- colnames(plot_mat)
        
        annotations <- data.frame(Cluster = labels)
        if (any(is.na(colors))) {
            cluster_colors <- brewer.pal(
                length(unique(labels)), "Set3")
            names(cluster_colors) <- unique(labels)
        } else {
            cluster_colors <- colors
        }
        annotation_colors <- list(Cluster = cluster_colors)
        
        png(filename=paste(results_dir, "/heatmap.png", sep=""),
            width=11, height=10, units="in",
            pointsize=4, res=300)
        
        pheatmap(plot_mat,
    
                 cluster_cols=best_results$consensusTree,
                 cluster_rows=FALSE,
                 clustering_distance_rows=NA,
                 clustering_distance_cols=NA,
                 clustering_method=NA,
    
                 annotation_col=annotations,
                 annotation_colors=annotation_colors,
                 annotation_names_col=FALSE,
    
                 show_rownames=FALSE,
                 show_colnames=FALSE,
    
                 color=colorRampPalette(c("#ffffff", "#2185c5"))(10),
                 border_color=NA,
                 fontsize=18)
        dev.off()
        catln("done")
    }
    
    print_divider("=")
    
    return(list(consensus_results=cluster_results,
                best_results=best_results,
                cluster_assignments=cluster_assignments,
                ari=ari,
                acc=acc,
                pac=PAC,
                opt_k=optK,
                force_opt_k=force_opt_k))
}


