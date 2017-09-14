library(Biobase)
library(data.table)
library(scater)

setwd(paste("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/",
            "data/GSE81861_CRC", sep=""))

#### Create SCESet ####
if (!file.exists("original/crc.rds")) {
  
  crc.count <- fread("original/GSE81861_CRC_tumor_all_cells_COUNT.csv",
                     header=TRUE,
                     data.table=FALSE)
  rownames(crc.count) <- crc.count[,1]
  crc.count[,1] <- NULL
  
  # Extract feature information
  extractFeatureData <- function(data) {
    chr_locations <- gsub("_(.*)_ENSG(.*)", "",
                          rownames(data))
    gene_info <- gsub("chr([A-Za-z0-9]+):([0-9]+)-([0-9]+)_", "",
                      rownames(data))
    symbols <- gsub("_ENSG(.*)", "", gene_info)
    ensembl_ids <- gsub("(.*)_", "", gene_info)
    
    fdata <- cbind(symbols, chr_locations)
    rownames(fdata) <- ensembl_ids
    colnames(fdata) <- c("symbol", "location")
    fdata <- AnnotatedDataFrame(as.data.frame(fdata))

    return(fdata)
  }
  
  # Extract sample information
  extractSampleInformation <- function(data) {
    sample_ids <- gsub("__(.*)", "", colnames(data))
    cell_types <- gsub("__(.*)", "",
                       gsub("RH[CL]([0-9]+)__", "",
                            colnames(data)))
    pdata <- data.frame(cell_types)
    rownames(pdata) <- sample_ids
    colnames(pdata) <- "cell_type"
    pdata <- AnnotatedDataFrame(as.data.frame(pdata))
    
    # remove(cell_types)
    return(pdata)
  }
  
  fdata <- extractFeatureData(crc.count)
  pdata <- extractSampleInformation(crc.count)
  
  # Format expression data
  edata <- crc.count
  colnames(edata) <- rownames(pdata)
  rownames(edata) <- rownames(fdata)
  edata <- as.matrix(edata)
  
  crc.sceset <- newSCESet(countData=edata,
                          phenoData=pdata,
                          featureData=fdata,
                          logExprsOffset = 1)
  crc.sceset <- crc.sceset[!duplicated(fData(crc.sceset)$symbol)]
  crc.sceset <- calculateQCMetrics(crc.sceset)
  
  saveRDS(crc.sceset, "original/crc.rds")
  
  remove(edata)
  remove(fdata)
  remove(pdata)
  remove(extractFeatureData)
  remove(extractSampleInformation)
  remove(crc.count)
  remove(crc.sceset)
}
#### Data Processing ####
crc <- readRDS("original/crc.rds")

### ---------- Parameters ---------- ###
n_genes <- 100
standardize <- FALSE
scale <- FALSE

### ---------- Gene Filtering ---------- ###
crc.processed <- crc

# Filter out dropout genes
low_exp.filter <- rowSums(
  counts(crc.processed) == 0) > (ncol(crc.processed) * 0.9)
crc.processed <- crc.processed[!low_exp.filter,]

remove(low_exp.filter)

### ---------- Select Most Variable Genes ---------- ###
library(e1071)
exprs <- exprs(crc.processed)

genes_sd <- apply(exprs, 1, sd, na.rm=TRUE)
genes_mean <- rowMeans(exprs, na.rm=TRUE)
genes_cv <- genes_sd/genes_mean

mean_cv_model <- svm(log2(genes_cv) ~ log2(genes_mean))
predicted_log2cv <- predict(mean_cv_model, genes_mean)
gene_scores <- log2(genes_cv) - predicted_log2cv

var.filter <- names(sort(gene_scores, decreasing=TRUE)[1:n_genes])

crc.processed <- crc.processed[var.filter,]

rm(var.filter)
rm(exprs)
rm(genes_sd)
rm(genes_mean)
rm(genes_cv)
rm(mean_cv_model)
rm(predicted_log2cv)

# plotTSNE(crc.processed, colour_by="cell_type")

# ---------- Standardize Expression Values to N(0, 1) ---------- #
if (standardize) {
  standardize_gene <- function(x) { (x - mean(x))/sd(x) }
  exprs.scaled <- t(apply(exprs(crc.processed), 1, standardize_gene))
  exprs(crc.processed) <- exprs.scaled
  
  rm(standardize_gene)
  rm(exprs.scaled)   
}

# ---------- Scale Expression Values to [0, 1] ---------- #
if (scale) {
  scale_gene <- function(x) { (x - min(x))/(max(x) - min(x)) }
  exprs.scaled <- t(apply(exprs(crc.processed), 1, scale_gene))
  exprs(crc.processed) <- exprs.scaled
  
  rm(scale_gene)
  rm(exprs.scaled)    
}

# ---------- Create VAE-ready Benchmark Dataset ---------- #
df <- as.data.frame(t(exprs(crc.processed)))
colnames(df) <- fData(crc.processed)$symbol

df <- cbind(rownames(df), df)
colnames(df)[1] <- "cell_id"

norm_technique <- NA
if (scale) {
  norm_technique <- "scaled"
} else if (standardize) {
  norm_technique <- "standardized"
}

dir.create("processed", showWarnings = FALSE)

if (!is.na(norm_technique)) {
  write.table(df,
              file=paste(
                "processed/crc.", n_genes,
                "g.", norm_technique, ".txt", sep=""),
              quote=FALSE, row.names=FALSE,
              col.names=TRUE,
              sep="\t")
  
  saveRDS(crc.processed,
          paste("processed/crc.", n_genes,
                "g.", norm_technique, ".SCESet.rds", sep=""))
} else {
  write.table(df,
              file=paste(
                "processed/crc.", n_genes, "g.txt", sep=""),
              quote=FALSE, row.names=FALSE,
              col.names=TRUE,
              sep="\t")
  
  saveRDS(crc.processed,
          paste("processed/crc.", n_genes, "g.SCESet.rds", sep=""))
}
