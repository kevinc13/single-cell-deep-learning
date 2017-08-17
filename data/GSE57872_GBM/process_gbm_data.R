library(Biobase)
library(data.table)
library(scater)

setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/data/GSE57872_GBM")

#### GSE57872: Glioblastoma dataset ####
gbm_cells.TPM <- fread("original/GSE57872_GBM_all_cells_TPM.txt",
                       header=TRUE,
                       data.table=FALSE)
rownames(gbm_cells.TPM) <- gbm_cells.TPM[,1]
gbm_cells.TPM[,1] <- NULL

sample_ids <- colnames(gbm_cells.TPM)
cell_origin <- c(rep("primary_GBM", 430),
                 rep("GBM_cell_line", 102),
                 rep("tumor_cell_line", 2),
                 "population_control",
                 rep("tumor_cell_line", 2),
                 rep("population_control", 3),
                 rep("tumor_cell_line", 2),
                 "population_control")
pdata <- as.data.frame(cell_origin)
rownames(pdata) <- sample_ids
pdata <- AnnotatedDataFrame(pdata)

edata <- as.matrix(gbm_cells.TPM)

gbm.sceset <- newSCESet(exprsData=edata,
                        phenoData=pdata)
gbm.sceset <- calculateQCMetrics(gbm.sceset)

remove(sample_ids)
remove(cell_origin)
remove(pdata)
remove(edata)
remove(gbm_cells.TPM)

saveRDS(gbm.sceset, "original/gbm.rds")
#### Data Processing ####
gbm <- readRDS("original/gbm.rds")

gbm <- gbm[,gbm$cell_origin == "primary_GBM"]

### ---------- Parameters ---------- ###
n_genes <- 500
standardize <- TRUE
scale <- FALSE

### ---------- Select Most Variable Genes ---------- ###
gbm.processed <- gbm

genes_var <- apply(exprs(gbm.processed), 1, var, na.rm=TRUE)
gbm.processed <- gbm.processed[names(sort(genes_var, decreasing=TRUE)[1:n_genes]),]

rm(genes_var)

# plotTSNE(gbm.processed, colour_by="non_malignant_cell_type")

# ---------- Standardize Expression Values to N(0, 1) ---------- #
if (standardize) {
    standardize_gene <- function(x) { (x - mean(x))/sd(x) }
    exprs.scaled <- t(apply(exprs(gbm.processed), 1, standardize_gene))
    exprs(gbm.processed) <- exprs.scaled
    
    rm(standardize_gene)
    rm(exprs.scaled)   
}

# ---------- Scale Expression Values to [0, 1] ---------- #
if (scale) {
    scale_gene <- function(x) { (x - min(x))/(max(x) - min(x)) }
    exprs.scaled <- t(apply(exprs(gbm.processed), 1, scale_gene))
    exprs(gbm.processed) <- exprs.scaled
    
    rm(scale_gene)
    rm(exprs.scaled)    
}

# ---------- Create VAE-ready Benchmark Dataset ---------- #
df <- as.data.frame(t(exprs(gbm.processed)))

df <- cbind(rownames(df), df)
colnames(df)[1] <- "cell_id"

norm_technique <- ""
if (scale) {
    norm_technique <- "scaled"
} else if (standardize) {
    norm_technique <- "standardized"
}

write.table(df,
            file=paste(
                "processed/gbm.", n_genes,
                "g.", norm_technique, ".txt", sep=""),
            quote=FALSE, row.names=FALSE,
            col.names=TRUE,
            sep="\t")

saveRDS(gbm.processed,
        paste("processed/gbm.", n_genes,
              "g.", norm_technique, ".SCESet.rds", sep=""))