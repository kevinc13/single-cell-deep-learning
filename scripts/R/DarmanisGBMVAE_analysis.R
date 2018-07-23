base.dir <- "~/Documents/Research/XinghuaLuLab/single-cell-deep-learning"
setwd(base.dir)

library(data.table)
library(sigclust)

library(limma)
library(WGCNA)
library(dendextend)

library(org.Hs.eg.db)
library(gage)
library(gageData)

library(survminer)
library(survival)
library(pheatmap)
library(RColorBrewer)

source("scripts/R/lib/clustering.R")
source("scripts/R/latent_space_consensus_clustering.R")
source("scripts/R/get_marker_genes.R")

#### Configuration -------------------------------------------------------------
dataset.file <- "data/GSE84465_GBM/processed/gbm.1000g.sce.rds"

model.dir <- "results/darmanis/Train1000gDarmanisGBMVAE/DarmanisGBMVAE_Final"
max.k <- 12
force.k <- NA
features.end.col <- -2
label.col <- "tumor_id"

ntop.marker.genes <- 10

# Control what analyses are run
run.clustering <- TRUE
run.tsne <- TRUE
run.verhaak.subtyping <- FALSE
run.marker.genes <- TRUE
run.pathway.analysis <- TRUE
run.tcga.projection <- TRUE

#### Load Data -----------------------------------------------------------------
tcga.gbm <- readRDS("data/TCGA/original/gbm.logtpm.rds")

# Load single-cell data (original and processed)
sc.gbm <- readRDS("data/GSE84465_GBM/original/gbm.rds")

dataset <- readRDS(dataset.file)
colnames(colData(dataset))[4] <- "cell_origin"

latent.space.data <- loadLatentSpace(model.dir, features.end.col, label.col)
latent.space.df <- latent.space.data$latent.space
features <- latent.space.data$features
tumor.ids <- latent.space.data$labels
rm(latent.space.data)

dataset <- dataset[,rownames(features)]
sc.gbm <- sc.gbm[,rownames(features)]

#### Verhaak Subtype Analysis of TCGA GBM Samples ------------------------------
if (run.verhaak.subtyping) {
  verhaak <- fread("data/GSE57872_GBM/verhaak_subtype_signatures.txt",
                   header=TRUE, data.table=FALSE, sep="\t")
  rownames(verhaak) <- verhaak[,1]
  verhaak[,1] <- NULL
  
  verhaak.genes <- alias2SymbolTable(rownames(verhaak))
  verhaak <- verhaak[!is.na(verhaak.genes),]
  rownames(verhaak) <- verhaak.genes[!is.na(verhaak.genes)]
  
  verhaak <- verhaak[
    rownames(verhaak) %in% fData(tcga.gbm)$official_gene_symbol,]
  gbm.verhaak <- tcga.gbm[
    fData(tcga.gbm)$official_gene_symbol %in% rownames(verhaak),]
  gbm.verhaak <- gbm.verhaak[
    !duplicated(fData(gbm.verhaak)$official_gene_symbol),]
  verhaak <- verhaak[fData(gbm.verhaak)$official_gene_symbol,]
  
  gbm.verhaak.projection <- cor(gbm.verhaak, verhaak)
  subtypes <- as.factor(max.col(gbm.verhaak.projection))
  levels(subtypes) <- colnames(gbm.verhaak.projection)
  tcga.gbm$verhaak_subtype <- subtypes
  
  rm(subtypes)
  rm(gbm.verhaak.projection)
  rm(gbm.verhaak)
  rm(verhaak.genes)
  
  subtype.survfit <- survfit(
    Surv(tcga.gbm$TIME_TO_EVENT, tcga.gbm$EVENT) ~ tcga.gbm$verhaak_subtype)
  ggsurvplot(
    subtype.survfit, 
    data = tcga.gbm, 
    size = 1,                 # change line size
    conf.int = FALSE,          # Add confidence interval
    pval = TRUE,              # Add p-value
    risk.table = TRUE,        # Add risk table
    risk.table.col = "strata",# Risk table color by groups
    risk.table.height = 0.25, # Useful to change when you have multiple groups
    ggtheme = theme_bw()      # Change ggplot2 theme
  )
}

#### Clustering ----------------------------------------------------------------
if (run.clustering) {
  # Perform hierarchical consensus clustering on the latent space 
  if (any(is.na(force.k))) {
    file <- "consensus_clusters_optK.txt"
  } else {
    file <- paste("consensus_clusters_k", force.k, ".txt", sep="")
  }
  
  clust.results <- clusterLatentSpace(features, model.dir, 
                                      max.k, force.k=force.k, 
                                      alg="wardHC", dist="euclidean", 
                                      clusters.output.file=file)
  remove(file)
  if (any(is.na(force.k))) {
    clusters <- as.factor(clust.results$clusters)  
  } else {
    clusters <- as.factor(clust.results$results[[force.k]]$consensusClass)
  }
  
  colData(dataset)$clusters <- clusters
  colData(sc.gbm)$clusters <- clusters
  
  # Calculate significance of clusters
  # sigclust.table <- matrix(nrow=length(levels(clusters)),
  #                          ncol=length(levels(clusters)))
  # pairs <- combn(levels(clusters), 2)
  # for (p in 1:ncol(pairs)) {
  #   cluster.1 <- as.numeric(pairs[1,p])
  #   cluster.2 <- as.numeric(pairs[2,p])
  #   group.1 <- dataset[,dataset$clusters == cluster.1]
  #   group.2 <- dataset[,dataset$clusters == cluster.2]
  #   sigclust.x <- rbind(t(exprs(group.1)), t(exprs(group.2)))
  #   sigclust.labels <- c(rep(1, ncol(group.1)), rep(2, ncol(group.2)))
  #   sigclust.results <- sigclust(sigclust.x, nsim=100,
  #                                labflag=1, label=sigclust.labels,
  #                                icovest=3)
  #   sigclust.table[cluster.1, cluster.2] <- sigclust.results@pval
  #   print(paste("Finished significance testing of pair #", p, sep=""))
  # }
  
  # Silhouette analysis of the clusters  
}

#### Visualization -------------------------------------------------------------
if (run.tsne) {
  tsne(features, labels=as.factor(tumor.ids), base.dir=model.dir,
       file="tSNE_plot_tumor_ids.png")
  tsne(features, labels=clusters, base.dir=model.dir,
       file=paste("tSNE_plot_clusters_k", 
                  length(unique(clusters)), ".png", sep="")) 
}

#### Get Marker Genes ----------------------------------------------------------
if (run.marker.genes) {
  marker.genes <- getMarkerGenes(dataset, clusters,
                                 base.dir=model.dir,
                                 output.dir=paste(
                                   "marker_genes_k", 
                                   length(unique(clusters)), sep=""))
  
  # Get top n marker genes for each cluster
  all.marker.genes <- c()
  for (c in 1:length(unique(clusters))) {
    # order by p value
    all.marker.genes <- union(
      all.marker.genes, rownames(
        head(marker.genes[[c]][order(marker.genes[[c]]$adj.P.Val),],
             n=ntop.marker.genes)
      ))
  }
  
  annotations <- data.frame(Cluster = colData(dataset)$clusters,
                            TumorID = colData(dataset)$cell_origin)
  rownames(annotations) <- colnames(dataset)
  cluster.colors <- brewer.pal(
    length(unique(colData(dataset)$clusters)), "Set3")
  names(cluster.colors) <- unique(colData(dataset)$clusters)
  
  tumorID.colors <- brewer.pal(
    length(unique(colData(dataset)$cell_origin)), "Spectral")
  names(tumorID.colors) <- unique(colData(dataset)$cell_origin)
  annotation.colors <- list(Cluster = cluster.colors,
                            TumorID = tumorID.colors)
  if (any(is.na(force.k))) {
    cluster.cols <- clust.results$opt.results$consensusTree
  } else {
    cluster.cols <- clust.results$results[[force.k]]$consensusTree
  }
  
  png(filename=paste(model.dir, "/marker_genes_k", length(unique(clusters)),
                     "/heatmap.png", sep=""),
      width=12, height=8, units="in",
      pointsize=4, res=300)
  pheatmap(exprs(dataset[all.marker.genes,]),
           
           color=colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
           
           cluster_cols=cluster.cols,
           clustering_method="ward.D2",
           cutree_cols = length(unique(clusters)),
           cutree_rows = 4,
           
           annotation_col=annotations,
           annotation_colors=annotation.colors,
           show_colnames=FALSE,
           fontsize=14,
           fontsize_row=8)
  dev.off()
  
  remove(annotations)
  remove(cluster.colors)
  remove(annotation.colors)
  remove(tumorID.colors)
}

#### Run Pathway  ----------------------------------------------------------
if (run.pathway.analysis && run.marker.genes) {
  data("go.sets.hs")
  data("go.subs.hs")
  go.bp.sets = go.sets.hs[go.subs.hs$BP]
  
  data("kegg.sets.hs")
  data("sigmet.idx.hs")
  kegg.sets.hs <- kegg.sets.hs[sigmet.idx.hs]
  
  fold.changes <- getFoldChanges(sc.gbm, clusters)
  
  go.results <- list()
  kegg.results <- list()
  for (c in 1:length(unique(clusters))) {
    clust.exprs <- fold.changes[,c]
    symbols <- alias2SymbolTable(rownames(fold.changes))
    clust.exprs <- clust.exprs[!is.na(symbols)]
    names(clust.exprs) <- symbols[!is.na(symbols)]
    rm(symbols)
    
    entrez <- mget(x=names(clust.exprs), envir=org.Hs.egALIAS2EG)
    for (i in 1:length(entrez)) {
      names(clust.exprs)[i] <- entrez[[i]][1]
    }
    rm(entrez)
    rm(i)
    
    go.result <- gage(clust.exprs, gsets=go.bp.sets, same.dir=TRUE)
    go.results[[c]] <- go.result
    
    kegg.result <- gage(clust.exprs, gsets=kegg.sets.hs, same.dir=TRUE)
    kegg.results[[c]] <- kegg.result
  }
}

#### Project TCGA GBM Samples onto Single-Cell Signatures ----------------------
if (run.tcga.projection) {
  # Load reference profiles
  ref.profiles.df <- fread(paste(
    model.dir, "/reference_profiles/reference_profiles_k", 
    length(unique(clusters)), ".txt", sep=""),
    header=TRUE, data.table=FALSE)
  rownames(ref.profiles.df) <- ref.profiles.df[,1]
  ref.profiles.df[,1] <- NULL
  
  # Remove last cluster (normal cluster)
  ref.profiles.df <- ref.profiles.df[1:(nrow(ref.profiles.df) - 1),]
  ref.profiles.df <- t(ref.profiles.df)
  
  # Get official symbols
  sc.genes <- alias2SymbolTable(rownames(ref.profiles.df))
  # remove NA genes
  ref.profiles.df <- ref.profiles.df[!is.na(sc.genes),]
  rownames(ref.profiles.df) <- sc.genes[!is.na(sc.genes)]
  rm(sc.genes)
  
  # Intersect gene lists
  tcga.gbm <- tcga.gbm[
    fData(tcga.gbm)$official_gene_symbol %in% rownames(ref.profiles.df),]
  tcga.gbm <- tcga.gbm[!duplicated(fData(tcga.gbm)$official_gene_symbol),]
  ref.profiles.df <- ref.profiles.df[
    rownames(ref.profiles.df) %in% fData(tcga.gbm)$official_gene_symbol,]
  ref.profiles.df <- ref.profiles.df[fData(tcga.gbm)$official_gene_symbol,]
  stopifnot(rownames(ref.profiles.df) == fData(tcga.gbm)$official_gene_symbol)
  
  # Project TCGA samples onto single cell reference profiles
  projection <- cor(exprs(tcga.gbm), ref.profiles.df)
  projection.scaled <- as.data.frame(scale(projection))
  
  # Cluster projections
  projection.clust <- hclust(dist(projection.scaled), method="ward.D2")
  projection.dend <- as.dendrogram(projection.clust)
  projection.dend <- color_branches(projection.dend, k=4)
  plot(projection.dend)
  
  tcga.gbm.clusters <- cutree(projection.clust, k=4)
  
  # tcga.gbm.clusters <- cutreeDynamic(projection.clust,
  #                                    minClusterSize=30, method="tree",
  #                                    distM=as.matrix(dist(projection.scaled)),
  #                                    deepSplit=1)
  
  # projection.results.dir <- paste(model.dir, "/TCGA_GBM_projection", sep="")
  # projection.clust <- consensusClustering(projection.scaled, labels=NA,
  #                                         projection.results.dir,
  #                                         max.k=12, force.k=NA,
  #                                         alg="wardHC", dist="euclidean",
  #                                         plot=TRUE, colors=NA, verbose=TRUE)
  # tcga.gbm.clusters <- projection.clust$results[[2]]$consensusClass
  
  # Survival Analysis
  gbm.survfit <- survfit(
    Surv(tcga.gbm$TIME_TO_EVENT, tcga.gbm$EVENT) ~ tcga.gbm.clusters)
  gbm.coxph <- coxph(
    Surv(tcga.gbm$TIME_TO_EVENT, tcga.gbm$EVENT) ~ tcga.gbm.clusters)
  
  
  ggsurvplot(
    gbm.survfit, 
    data = tcga.gbm, 
    size = 1,                 # change line size
    conf.int = FALSE,          # Add confidence interval
    pval = TRUE,              # Add p-value
    risk.table = TRUE,        # Add risk table
    risk.table.col = "strata",# Risk table color by groups
    risk.table.height = 0.25, # Useful to change when you have multiple groups
    ggtheme = theme_bw()      # Change ggplot2 theme
  )
}



