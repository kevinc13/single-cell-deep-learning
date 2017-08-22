library(Biobase)
library(data.table)
library(scater)

setwd(paste("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/",
            "data/GSE94820_PBMC", sep=""))

#### Create SCESet ####
if (!file.exists("original/pbmc.rds")) {
    use_subtypes <- F
    
    if (use_subtypes) {
        pbmc.TPM <- fread("original/cell_subtype_exp_matrix.txt",
                          header=TRUE,
                          data.table=FALSE)
    } else {
        pbmc.TPM <- fread("original/cell_type_exp_matrix.txt",
                          header=TRUE,
                          data.table=FALSE)
    }
    
    rownames(pbmc.TPM) <- pbmc.TPM$Gene
    pbmc.TPM$Gene <- NULL
    
    if (use_subtypes) {
        cell_subtypes <- colnames(pbmc.TPM)
        cell_subtypes <- gsub("_S([0-9]+)(.*)", "", cell_subtypes)   
        pdata <- data.frame(cell_subtype=cell_subtypes)
    } else {
        cell_types <- colnames(pbmc.TPM)
        cell_types[grep("pDC", cell_types)] <- "pDC"
        cell_types[grep("CD141", cell_types)] <- "CD141Pos_DC"
        cell_types[grep("CD1C", cell_types)] <- "CD1CPos_DC"
        cell_types[grep("DoubleNeg", cell_types)] <- "CD141NegCD1CNeg_DC"
        cell_types[grep("Mono_classical", cell_types)] <- "Mono_Classical"
        cell_types[grep("Mono_intermediate", cell_types)] <- "Mono_Intermediate"
        cell_types[grep("Mono_nonclassical", cell_types)] <- "Mono_Nonclassical"
        pdata <- data.frame(cell_type=cell_types)
    }
    
    rownames(pdata) <- colnames(pbmc.TPM)
    pdata <- AnnotatedDataFrame(pdata)
    
    edata <- as.matrix(pbmc.TPM)
    
    pbmc.sceset <- newSCESet(tpmData=edata,
                             phenoData=pdata,
                             logExprsOffset = 1)
    pbmc.sceset <- calculateQCMetrics(pbmc.sceset)
    
    saveRDS(pbmc.sceset, "original/pbmc.rds")
    
    remove(pbmc.sceset)
    remove(edata)
    remove(pdata)
    remove(pbmc.TPM)
    if (use_subtypes) {
        remove(cell_subtypes)  
    } else {
        remove(cell_types)
    }
    remove(use_subtypes)
}
#### Data Processing ####
pbmc <- readRDS("original/pbmc.rds")

### ---------- Parameters ---------- ###
n_genes <- 1000
standardize <- TRUE
scale <- FALSE

### ---------- Gene Filtering ---------- ###
pbmc.processed <- pbmc

# Filter out dropout genes
low_exp.filter <- rowSums(tpm(pbmc.processed) == 0) > (ncol(pbmc.processed) * 0.9)
pbmc.processed <- pbmc.processed[!low_exp.filter,]

### ---------- Select Most Variable Genes ---------- ###
library(e1071)
exprs <- exprs(pbmc.processed)

genes_sd <- apply(exprs, 1, sd, na.rm=TRUE)
genes_mean <- rowMeans(exprs, na.rm=TRUE)
genes_cv <- genes_sd/genes_mean

mean_cv_model <- svm(log2(genes_cv) ~ log2(genes_mean))
predicted_log2cv <- predict(mean_cv_model, genes_mean)
gene_scores <- log2(genes_cv) - predicted_log2cv

var.filter <- names(sort(gene_scores, decreasing=TRUE)[1:n_genes])

pbmc.processed <- pbmc.processed[var.filter,]

rm(var.filter)
rm(exprs)
rm(genes_sd)
rm(genes_mean)
rm(genes_cv)
rm(mean_cv_model)
rm(predicted_log2cv)

# plotTSNE(pbmc.processed, colour_by="cell_type")

# ---------- Standardize Expression Values to N(0, 1) ---------- #
if (standardize) {
    standardize_gene <- function(x) { (x - mean(x))/sd(x) }
    exprs.scaled <- t(apply(exprs(pbmc.processed), 1, standardize_gene))
    exprs(pbmc.processed) <- exprs.scaled
    
    rm(standardize_gene)
    rm(exprs.scaled)   
}

# ---------- Scale Expression Values to [0, 1] ---------- #
if (scale) {
    scale_gene <- function(x) { (x - min(x))/(max(x) - min(x)) }
    exprs.scaled <- t(apply(exprs(pbmc.processed), 1, scale_gene))
    exprs(pbmc.processed) <- exprs.scaled
    
    rm(scale_gene)
    rm(exprs.scaled)    
}

# ---------- Create VAE-ready Benchmark Dataset ---------- #
df <- as.data.frame(t(exprs(pbmc.processed)))
df$cell_type <- pbmc.processed$cell_type

df <- cbind(rownames(df), df)
colnames(df)[1] <- "cell_id"

norm_technique <- ""
if (scale) {
    norm_technique <- "scaled"
} else if (standardize) {
    norm_technique <- "standardized"
}

dir.create("processed", showWarnings = FALSE)

write.table(df,
            file=paste(
                "processed/pbmc.", n_genes,
                "g.", norm_technique, ".txt", sep=""),
            quote=FALSE, row.names=FALSE,
            col.names=TRUE,
            sep="\t")

saveRDS(pbmc.processed,
        paste("processed/pbmc.", n_genes,
              "g.", norm_technique, ".SCESet.rds", sep=""))