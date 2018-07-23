library(limma)

getMarkerGenes <- function(dataset, clusters, base.dir, 
                           output.dir="marker_genes") {
  stopifnot(all(colnames(dataset) == names(clusters)))
  
  # Get marker genes
  design <- model.matrix(~0+clusters) 
  
  fit <- lmFit(exprs(dataset), design)
  fit.ebayes <- eBayes(fit)
  
  marker.genes = list()
  for (c in levels(clusters)) {
    marker.genes[[c]] <- topTable(fit.ebayes, coef=as.numeric(c), 
                                  p.value=0.05, sort.by="logFC", number=Inf)
    cat(paste("Number of marker genes for cluster ", c, ": ", 
              nrow(marker.genes[[c]]), "\n", sep=""))
  }
  
  dir.create(paste(base.dir, "/", output.dir, sep=""), showWarnings=FALSE)
  
  for (c in levels(clusters)) {
    marker.genes.df <- cbind(rownames(marker.genes[[c]]), marker.genes[[c]])
    colnames(marker.genes.df)[1] <- "gene"
    write.table(marker.genes.df, file=paste(base.dir, "/", output.dir,
                                            "/cluster_", c, ".txt", sep=""),
                sep="\t", col.names=TRUE, row.names=FALSE, quote=FALSE)
  }
  
  return(marker.genes)
}

getFoldChanges <- function(dataset, clusters) {
  stopifnot(all(colnames(dataset) == names(clusters)))
  
  # Get marker genes
  design <- model.matrix(~0+clusters) 
  
  fit <- lmFit(exprs(dataset), design)
  fit.ebayes <- eBayes(fit)
  
  fold.changes <- topTable(fit.ebayes, n=Inf, sort="none")
  return(fold.changes)
}





