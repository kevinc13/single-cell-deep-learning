library(Seurat)
library(dplyr)
library(Matrix)

setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/data/10X_PBMC")

n_genes <- 100

pbmc_data <- Read10X(data.dir = "original/PBMC8k_cell_matrix_filtered/GRCh38/")
pbmc <- CreateSeuratObject(raw.data = pbmc_data, min.cells = 3, min.genes = 200, 
                           project = "10X_PBMC8k")

mito.genes <- grep(pattern = "^MT-", x = rownames(x = pbmc@data), value = TRUE)
percent.mito <- colSums(pbmc@raw.data[mito.genes, ])/colSums(pbmc@raw.data)

# AddMetaData adds columns to object@data.info, and is a great place to
# stash QC stats
pbmc <- AddMetaData(object = pbmc, metadata = percent.mito, col.name = "percent.mito")
VlnPlot(object = pbmc, features.plot = c("nGene", "nUMI", "percent.mito"), nCol = 3)

# We filter out cells that have unique gene counts over 2,500 or less than
# 200 Note that low.thresholds and high.thresholds are used to define a
# 'gate' -Inf and Inf should be used if you don't want a lower or upper
# threshold.
pbmc <- FilterCells(object = pbmc, subset.names = c("nGene", "percent.mito"), 
                    low.thresholds = c(200, -Inf), high.thresholds = c(2500, 0.05))
pbmc <- NormalizeData(object = pbmc, normalization.method = "LogNormalize", 
                      scale.factor = 10000)

pbmc <- FindVariableGenes(pbmc, do.plot=FALSE, x.low.cutoff = 0, x.high.cutoff = 0.8)
pbmc@var.genes <- names(head(
    sort(pbmc@hvg.info$gene.dispersion.scaled, decreasing=TRUE),
    n=n_genes))

pbmc <- ScaleData(object = pbmc, vars.to.regress = c("nUMI", "percent.mito"))

pbmc <- RunPCA(object = pbmc, pc.genes = pbmc@var.genes, do.print = TRUE, pcs.print = 1:5, 
               genes.print = 5)
pbmc <- FindClusters(object = pbmc, reduction.type = "pca", dims.use = 1:10, 
                     resolution = 0.6, print.output = 0, save.SNN = TRUE)

pbmc <- RunTSNE(object = pbmc, dims.use = 1:10, do.fast = TRUE, check_duplicates=FALSE)
# pbmc <- RunTSNE(pbmc, genes.use=pbmc@var.genes, do.fast=TRUE, check_duplicates=FALSE)

save(pbmc, file="processed/PBMC8k.Seurat.RData")