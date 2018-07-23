library(Biobase)
library(data.table)
library(scater)
library(scran)

setwd(paste("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/",
            "data/GSE75688_Breast_Cancer", sep=""))

#### Create SCESet ####
if (!file.exists("original/breast.rds")) {
  
  breast.TPM <- fread("original/GSE75688_breast_cancer_cells.TPM.txt",
                      header=TRUE,
                      data.table=FALSE)
  
  rownames(breast.TPM) <- breast.TPM$gene_id
  breast.TPM$gene_id <- NULL
  
  fdata <- breast.TPM[,1:2,drop=FALSE]
  colnames(fdata) <- c("symbol", "gene_type")
  fdata <- AnnotatedDataFrame(fdata)
  
  edata <- breast.TPM[,3:ncol(breast.TPM)]
  sample_ids <- colnames(edata)
  tumor_ids <- gsub("_(.*)", "", sample_ids)
  pdata <- as.data.frame(rep("single_cell", length(sample_ids)))
  pdata <- cbind(pdata, tumor_ids)
  pdata <- cbind(pdata,
                 as.data.frame(rep("type", length(tumor_ids))))
  rownames(pdata) <- sample_ids
  colnames(pdata) <- c("sample_type", "tumor_id", "tumor_type")
  
  pdata$sample_type <- as.character(pdata$sample_type)
  pdata$sample_type[grep("Pooled", sample_ids)] <- "pooled"
  pdata$sample_type[grep("Tumor", sample_ids)] <- "bulk_tumor"
  
  pdata$tumor_type <- as.character(pdata$tumor_type)
  pdata$tumor_type[grep("BC0[1-2]", tumor_ids)] <- "er_positive"
  pdata$tumor_type[which(tumor_ids == "BC03")] <- "er_positive|her2_postiive"
  pdata$tumor_type[which(tumor_ids == "BC03LN")] <- "er_positive|her2_positive|lymph_node_metastasis"
  pdata$tumor_type[grep("BC0[4-6]", tumor_ids)] <- "her2_positive"
  pdata$tumor_type[grep("BC0[7-9]|BC1[0-1]", tumor_ids)] <- "tnbc"
  pdata$tumor_type[which(tumor_ids == "BC07LN")] <- "tnbc|lymph_node_metastasis"
  stopifnot(sum(pdata$tumor_type == "type") == 0)
  
  pdata <- AnnotatedDataFrame(pdata)
  
  breast.sceset <- newSCESet(tpmData=edata,
                             phenoData=pdata,
                             featureData=fdata,
                             logExprsOffset = 1)
  breast.sceset <- calculateQCMetrics(breast.sceset)
  
  saveRDS(breast.sceset, "original/breast.rds")
  
  remove(breast.sceset)
  remove(edata)
  remove(pdata)
  remove(fdata)
  remove(breast.TPM)
  remove(sample_ids)
  remove(tumor_ids)
}
#### Data Processing ####
breast <- readRDS("original/breast.rds")

### ---------- Parameters ---------- ###
n_genes <- 1000
standardize <- FALSE
scale <- FALSE

# Only include single cell samples
breast.processed <- breast[,breast$sample_type == "single_cell"]

### ---------- Gene Filtering ---------- ###
# Only include protein coding genes
protein_coding.filter <- fData(breast.processed)$gene_type == "protein_coding"
breast.processed <- breast.processed[protein_coding.filter,]

# Filter out dropout genes
low_exp.filter <- rowSums(tpm(breast.processed) == 0) > (ncol(breast.processed) * 0.9)
breast.processed <- breast.processed[!low_exp.filter,]

# Filter out genes not included in TCGA dataset
tcga_gene_list <- readRDS("../TCGA/original/tcga_gene_symbol_list.rds")
tcga.filter <- fData(breast.processed)$symbol %in% tcga_gene_list
breast.processed <- breast.processed[tcga.filter,]

remove(low_exp.filter)
remove(protein_coding.filter)
remove(tcga_gene_list)
remove(tcga.filter)

### ---------- Correct for Intertumoral Variability ---------- ###
# design <- model.matrix(~breast.processed$tumor_id)
# breast.processed <- normalizeExprs(breast.processed, design=design)

### ---------- Select Most Variable Genes ---------- ###

# Using trendVar from package 'scran'
# var.fit <- trendVar(breast.processed, use.spikes=FALSE, design=design)
var.fit <- trendVar(breast.processed, use.spikes=FALSE)
var.table <- decomposeVar(breast.processed, var.fit, get.spikes=FALSE)
var.table <- var.table[var.table$FDR <= 0.05,]
var.filter <- rownames(var.table)[order(var.table$bio, decreasing=TRUE)]
var.filter <- var.filter[1:n_genes]
breast.processed <- breast.processed[var.filter,]

remove(var.table)
remove(var.filter)
remove(var.fit)

# Using SVM model
# library(e1071)
# exprs <- exprs(breast.processed)
# 
# genes_sd <- apply(exprs, 1, sd, na.rm=TRUE)
# genes_mean <- rowMeans(exprs, na.rm=TRUE)
# genes_cv <- genes_sd/genes_mean
# 
# mean_cv_model <- svm(log2(genes_cv) ~ log2(genes_mean))
# predicted_log2cv <- predict(mean_cv_model, genes_mean)
# gene_scores <- log2(genes_cv) - predicted_log2cv
# 
# var.filter <- names(sort(gene_scores, decreasing=TRUE)[1:n_genes])
# breast.processed <- breast.processed[var.filter,]

# rm(var.filter)
# rm(exprs)
# rm(genes_sd)
# rm(genes_mean)
# rm(genes_cv)
# rm(mean_cv_model)
# rm(predicted_log2cv)

# plotTSNE(breast.processed, colour_by="tumor_id")

# ---------- Standardize Expression Values to N(0, 1) ---------- #
if (standardize) {
  standardize_gene <- function(x) { (x - mean(x))/sd(x) }
  exprs.standardized <- t(apply(exprs(breast.processed), 1, standardize_gene))
  exprs(breast.processed) <- exprs.standardized
  
  rm(standardize_gene)
  rm(exprs.scaled)   
}

# ---------- Scale Expression Values to [0, 1] ---------- #
if (scale) {
  scale_gene <- function(x) { (x - min(x))/(max(x) - min(x)) }
  exprs.scaled <- t(apply(exprs(breast.processed), 1, scale_gene))
  exprs(breast.processed) <- exprs.scaled
  
  rm(scale_gene)
  rm(exprs.scaled)    
}

# ---------- Create VAE-ready Benchmark Dataset ---------- #
df <- as.data.frame(t(exprs(breast.processed)))
colnames(df) <- fData(breast.processed)$symbol

df <- cbind(rownames(df), breast.processed$tumor_id, 
            breast.processed$tumor_type, df)
colnames(df)[1:3] <- c("cell_id", "tumor_id", "tumor_type")

norm_technique <- NA
if (scale) {
  norm_technique <- "scaled"
} else if (standardize) {
  norm_technique <- "standardized"
}

if (!is.na(norm_technique)) {
  write.table(df,
              file=paste(
                "processed/breast.", n_genes,
                "g.", norm_technique, ".txt", sep=""),
              quote=FALSE, row.names=FALSE,
              col.names=TRUE,
              sep="\t")

  saveRDS(breast.processed,
          paste("processed/breast.", n_genes,
                "g.", norm_technique, ".SCESet.rds", sep=""))
} else {
  write.table(df,
              file=paste(
                "processed/breast.", n_genes, "g.txt", sep=""),
              quote=FALSE, row.names=FALSE,
              col.names=TRUE,
              sep="\t")

  saveRDS(breast.processed,
          paste("processed/breast.", n_genes, "g.SCESet.rds", sep=""))
}

