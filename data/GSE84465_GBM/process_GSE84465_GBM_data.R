library(Biobase)
library(data.table)
library(scater)
library(scran)

setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/data/GSE84465_GBM")

if (!file.exists("original/gbm.rds")) {
  counts <- fread("original/GBM_raw_gene_counts.csv",
                  header=TRUE, sep=" ", data.table=FALSE, verbose=TRUE)
  rownames(counts) <- counts[,1]
  counts[,1] <- NULL
  edata <- as.matrix(counts)
  
  pdata <- fread("original/GBM_metadata.csv",
                 header=TRUE, sep=" ", data.table=FALSE)
  rownames(pdata) <- pdata$sample_id
  pdata[,1] <- NULL
  
  gbm.sce <- SingleCellExperiment(
    assays=list(counts=edata),
    colData=pdata
  )

  cpm(gbm.sce) <- calculateCPM(gbm.sce, use.size.factors=FALSE)
  gbm.sce <- normalize(gbm.sce, exprs_values="cpm",
                       return_log=TRUE, log_exprs_offset=1)
  
  gbm.sce <- calculateQCMetrics(gbm.sce)
  
  remove(pdata)
  remove(edata)
  remove(counts)
  
  saveRDS(gbm.sce, "original/gbm.rds")
}
#### Data Processing ####
gbm <- readRDS("original/gbm.rds")

gbm <- gbm[,colData(gbm)$Selection == "Unpanned"]

### ---------- Parameters ---------- ###
n_genes <- 1000

### ---------- Remove dropout genes ---------- ###
gbm.processed <- gbm
low_exp.filter <- rowSums(
  cpm(gbm.processed) == 0) > (ncol(gbm.processed) * 0.9)
gbm.processed <- gbm.processed[!low_exp.filter,]

### ---------- Select Most Variable Genes ---------- ###
var.fit <- trendVar(exprs(gbm.processed))
var.table <- decomposeVar(exprs(gbm.processed), var.fit)
var.table <- var.table[var.table$FDR <= 0.1,]
var.filter <- rownames(var.table)[order(var.table$bio, decreasing=TRUE)]
var.filter <- var.filter[1:n_genes]
gbm.processed <- gbm.processed[var.filter,]

remove(var.table)
remove(var.filter)
remove(var.fit)

# plotTSNE(gbm.processed, colour_by="cell_origin")

# ---------- Create VAE-ready Benchmark Dataset ---------- #
df <- as.data.frame(t(exprs(gbm.processed)))

df <- cbind(rownames(df), colData(gbm.processed)$Location,
            colData(gbm.processed)$Sample.name, df)
colnames(df)[1:3] <- c("cell_id", "location", "cell_origin")

write.table(df,
            file=paste(
              "processed/gbm.", n_genes,
              "g.txt", sep=""),
            quote=FALSE, row.names=FALSE,
            col.names=TRUE,
            sep="\t")

saveRDS(gbm.processed,
        paste("processed/gbm.", n_genes,
              "g.sce.rds", sep=""))

