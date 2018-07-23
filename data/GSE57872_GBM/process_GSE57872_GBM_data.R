library(Biobase)
library(data.table)
library(scater)
library(scran)

setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/data/GSE57872_GBM")

#### GSE57872: Glioblastoma dataset ####
if (!file.exists("original/gbm.rds")) {
  gbm_cells.TPM <- fread("original/GSE57872_GBM_all_cells_TPM.txt",
                         header=TRUE,
                         data.table=FALSE)
  rownames(gbm_cells.TPM) <- gbm_cells.TPM[,1]
  gbm_cells.TPM[,1] <- NULL
  
  sample_ids <- colnames(gbm_cells.TPM)
  sample_type <- c(rep("primary_GBM", 430),
                   rep("GBM_cell_line", 102),
                   rep("tumor_cell_line", 2),
                   "population_control",
                   rep("tumor_cell_line", 2),
                   rep("population_control", 3),
                   rep("tumor_cell_line", 2),
                   "population_control")
  cell_origin <- as.factor(gsub("_(.*)", "", sample_ids))
  cell_origin[cell_origin == "MGH264"] = "MGH26"
  pdata <- as.data.frame(sample_type)
  pdata$cell_origin <- factor(cell_origin)
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
}
#### Data Processing ####
gbm <- readRDS("original/gbm.rds")

gbm <- gbm[,gbm$sample_type == "primary_GBM"]

### ---------- Parameters ---------- ###
n_genes <- 1000

### ---------- Select Most Variable Genes ---------- ###
gbm.processed <- gbm

# design <- model.matrix(~gbm.processed$cell_origin)
# var.fit <- trendVar(gbm.processed, use.spikes=FALSE, design=design)

var.fit <- trendVar(gbm.processed, use.spikes=FALSE)
var.table <- decomposeVar(gbm.processed, var.fit, get.spikes=FALSE)
var.table <- var.table[var.table$FDR <= 0.05,]
var.filter <- rownames(var.table)[order(var.table$bio, decreasing=TRUE)]
var.filter <- var.filter[1:n_genes]
gbm.processed <- gbm.processed[var.filter,]

remove(var.table)
remove(var.filter)
remove(var.fit)

# plotTSNE(gbm.processed, colour_by="cell_origin")

# ---------- Create VAE-ready Benchmark Dataset ---------- #
df <- as.data.frame(t(exprs(gbm.processed)))

df <- cbind(rownames(df), gbm.processed$cell_origin, df)
colnames(df)[1:2] <- c("cell_id", "tumor_id")

write.table(df,
            file=paste(
              "processed/gbm.", n_genes,
              "g.centered.txt", sep=""),
            quote=FALSE, row.names=FALSE,
            col.names=TRUE,
            sep="\t")

saveRDS(gbm.processed,
        paste("processed/gbm.", n_genes,
              "g.centered.SCESet.rds", sep=""))

