library(Biobase)
library(data.table)
library(scater)
library(scran)

setwd(paste("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/",
            "data/GSE72056_Melanoma", sep=""))

#### Create SCESet ####
if (!file.exists("original/melanoma.TPM.rds")) {
  melanoma.TPM <- fread("original/melanoma_cells_TPM.txt",
                        header=TRUE,
                        data.table=FALSE)
  rownames(melanoma.TPM) <- make.unique(melanoma.TPM[,1])
  melanoma.TPM[,1] <- NULL
  
  # Malignant (0=unresolved, 1=no, 2=yes)
  # Non-malignant cell type (1=T, 2=B, 3=Macro, 4=Endothelial, 5=CAF, 6=NK)
  pdata <- data.frame(t(melanoma.TPM[1:3,]))
  colnames(pdata) <- c("tumor_id", "malignant", "non_malignant_cell_type")
  pdata <- AnnotatedDataFrame(pdata)
  
  edata <- as.matrix(melanoma.TPM[4:nrow(melanoma.TPM),])
  # Un-log transform original data
  edata <- 2^edata - 1
  
  melanoma.TPM.sceset <- newSCESet(tpmData=edata,
                                   phenoData=pdata,
                                   logExprsOffset = 1)
  melanoma.TPM.sceset <- calculateQCMetrics(melanoma.TPM.sceset)
  
  remove(pdata)
  remove(edata)
  remove(melanoma.TPM)
  
  melanoma.TPM.sceset$malignant <- factor(melanoma.TPM.sceset$malignant)
  levels(melanoma.TPM.sceset$malignant) <- c("unresolved", "no", "yes")
  melanoma.TPM.sceset$malignant <- as.character(melanoma.TPM.sceset$malignant)
  
  melanoma.TPM.sceset$non_malignant_cell_type <- factor(
    melanoma.TPM.sceset$non_malignant_cell_type)
  levels(melanoma.TPM.sceset$non_malignant_cell_type) <- c("unresolved", "T", "B", 
                                                           "Macro", "Endothelial",
                                                           "CAF", "NK")
  melanoma.TPM.sceset$non_malignant_cell_type <- as.character(
    melanoma.TPM.sceset$non_malignant_cell_type)
  
  melanoma.TPM.sceset$cell_type[
    melanoma.TPM.sceset$malignant == "yes"] <- "malignant"
  melanoma.TPM.sceset$cell_type[
    melanoma.TPM.sceset$non_malignant_cell_type != "unresolved"] <- 
    as.character(
      melanoma.TPM.sceset$non_malignant_cell_type[
        melanoma.TPM.sceset$non_malignant_cell_type != "unresolved"])
  melanoma.TPM.sceset$cell_type[is.na(melanoma.TPM.sceset$cell_type)] <- "unresolved"
  
  saveRDS(melanoma.TPM.sceset, "original/melanoma.TPM.rds")
  remove(melanoma.TPM.sceset)
}

#### Data Processing ####
melanoma <- readRDS("original/melanoma.TPM.rds")

### ---------- Parameters ---------- ###
only_malignant <- TRUE
n_genes <- 1000
standardize <- FALSE
scale <- FALSE

if (only_malignant) {
  melanoma <- melanoma[,melanoma$cell_type == "malignant"]
} else {
  melanoma <- melanoma[,melanoma$cell_type != "unresolved"]
}

### ---------- Gene Filtering ---------- ###
melanoma.processed <- melanoma
if (!only_malignant) {
  melanoma.processed <- melanoma.processed[
    ,melanoma$non_malignant_cell_type != "unresolved"]
}

# Filter out dropout genes
low_exp.filter <- rowSums(tpm(melanoma.processed) == 0) > (ncol(melanoma.processed) * 0.9)
melanoma.processed <- melanoma.processed[!low_exp.filter,]

# Filter out genes not included in TCGA dataset
tcga_gene_list <- readRDS("../TCGA/original/tcga_gene_symbol_list.rds")
tcga.filter <- rownames(melanoma.processed) %in% tcga_gene_list
melanoma.processed <- melanoma.processed[tcga.filter,]

### ---------- Normalize Dataset (Remove Batch Effects) ---------- ###
# design <- model.matrix(~as.factor(melanoma.processed$tumor_id))
# melanoma.processed <- normalizeExprs(melanoma.processed, 
#                                      design=design)

### ---------- Select Most Variable Genes ---------- ###
var.fit <- trendVar(melanoma.processed, use.spikes=FALSE)
var.table <- decomposeVar(melanoma.processed, var.fit, get.spikes=FALSE)
var.table <- var.table[var.table$FDR <= 0.05,]

var.filter <- rownames(var.table)[order(var.table$bio, decreasing=TRUE)]
var.filter <- var.filter[1:n_genes]
melanoma.processed <- melanoma.processed[var.filter,]

remove(var.table)
remove(var.filter)
remove(var.fit)

# library(e1071)
# if (!only_malignant) {
#   exprs <- exprs(melanoma.processed[
#     ,melanoma$non_malignant_cell_type != "unresolved"])
# } else {
#   exprs <- exprs(melanoma.processed)
# }
# genes_sd <- apply(exprs, 1, sd, na.rm=TRUE)
# genes_mean <- rowMeans(exprs, na.rm=TRUE)
# genes_cv <- genes_sd/genes_mean
# 
# mean_cv_model <- svm(log2(genes_cv) ~ log2(genes_mean))
# predicted_log2cv <- predict(mean_cv_model, genes_mean)
# gene_scores <- log2(genes_cv) - predicted_log2cv
# 
# var.filter <- names(sort(gene_scores, decreasing=TRUE)[1:n_genes])
# 
# melanoma.processed <- melanoma.processed[var.filter,]
# 
# rm(var.filter)
# rm(exprs)
# rm(genes_sd)
# rm(genes_mean)
# rm(genes_cv)
# rm(mean_cv_model)
# rm(predicted_log2cv)

# plotTSNE(melanoma.processed, colour_by="non_malignant_cell_type")

# ---------- Standardize Expression Values to N(0, 1) ---------- #
# if (standardize) {
    # standardize_gene <- function(x) { (x - mean(x))/sd(x) }
    # exprs.scaled <- t(apply(exprs(melanoma.processed), 1, standardize_gene))
    # exprs(melanoma.processed) <- exprs.scaled
#     
#     rm(standardize_gene)
#     rm(exprs.scaled)   
# }
# 
# # ---------- Scale Expression Values to [0, 1] ---------- #
# if (scale) {
#     scale_gene <- function(x) { (x - min(x))/(max(x) - min(x)) }
#     exprs.scaled <- t(apply(exprs(melanoma.processed), 1, scale_gene))
#     exprs(melanoma.processed) <- exprs.scaled
#     
#     rm(scale_gene)
#     rm(exprs.scaled)    
# }

# ---------- Create Dataset ---------- #
df <- as.data.frame(t(exprs(melanoma.processed)))
if (!only_malignant) {
  df$cell_type <- melanoma.processed$cell_type 
}
df <- cbind(rownames(df), melanoma.processed$tumor_id, df)
colnames(df)[1:2] <- c("cell_id", "tumor_id")

norm_technique <- NA
if (scale) {
    norm_technique <- "scaled"
} else if (standardize) {
    norm_technique <- "standardized"
}

if (only_malignant) {
  name <- "malignant_cells"
} else {
  name <- "all_cells"
}

if (!is.na(norm_technique)) {
  write.table(df,
              file=paste(
                "processed/", name, ".", n_genes,
                "g.", norm_technique, ".txt", sep=""),
              quote=FALSE, row.names=FALSE,
              col.names=TRUE,
              sep="\t")

  saveRDS(melanoma.processed,
          paste("processed/", name, ".", n_genes,
                "g.", norm_technique, ".SCESet.rds", sep=""))
} else {
  write.table(df,
              file=paste(
                "processed/", name, ".", n_genes, "g.txt", sep=""),
              quote=FALSE, row.names=FALSE,
              col.names=TRUE,
              sep="\t")

  saveRDS(melanoma.processed,
          paste("processed/", name, ".", n_genes, "g.SCESet.rds", sep=""))
}


