library(Biobase)
library(data.table)
library(scater)

setwd(paste("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/",
            "data/Pollen", sep=""))

#### Data Processing ####
pollen <- readRDS("original/pollen.rds")

### ---------- Parameters ---------- ###
n_genes <- 1000
standardize <- FALSE
scale <- FALSE

### ---------- Gene Filtering ---------- ###
pollen.processed <- pollen

# Filter out dropout genes
low_exp.filter <- rowSums(tpm(pollen.processed) == 0) >
  (ncol(pollen.processed) * 0.9)
pollen.processed <- pollen.processed[!low_exp.filter,]

### ---------- Select Most Variable Genes ---------- ###
library(e1071)
exprs <- exprs(pollen.processed)

genes_sd <- apply(exprs, 1, sd, na.rm=TRUE)
genes_mean <- rowMeans(exprs, na.rm=TRUE)
genes_cv <- genes_sd/genes_mean

mean_cv_model <- svm(log2(genes_cv) ~ log2(genes_mean))
predicted_log2cv <- predict(mean_cv_model, genes_mean)
gene_scores <- log2(genes_cv) - predicted_log2cv

var.filter <- names(sort(gene_scores, decreasing=TRUE)[1:n_genes])

pollen.processed <- pollen.processed[var.filter,]

rm(var.filter)
rm(exprs)
rm(genes_sd)
rm(genes_mean)
rm(genes_cv)
rm(mean_cv_model)
rm(predicted_log2cv)

# plotTSNE(pollen.processed, colour_by="cell_type")

# ---------- Standardize Expression Values to N(0, 1) ---------- #
if (standardize) {
  standardize_gene <- function(x) { (x - mean(x))/sd(x) }
  exprs.scaled <- t(apply(exprs(pollen.processed), 1, standardize_gene))
  exprs(pollen.processed) <- exprs.scaled
  
  rm(standardize_gene)
  rm(exprs.scaled)   
}

# ---------- Scale Expression Values to [0, 1] ---------- #
if (scale) {
  scale_gene <- function(x) { (x - min(x))/(max(x) - min(x)) }
  exprs.scaled <- t(apply(exprs(pollen.processed), 1, scale_gene))
  exprs(pollen.processed) <- exprs.scaled
  
  rm(scale_gene)
  rm(exprs.scaled)    
}

# ---------- Create VAE-ready Benchmark Dataset ---------- #
df <- as.data.frame(t(exprs(pollen.processed)))
df$cell_type <- pollen.processed$cell_type
df$cell_subtype <- pollen.processed$cell_subtype

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
                "processed/pollen.", n_genes,
                "g.", norm_technique, ".txt", sep=""),
              quote=FALSE, row.names=FALSE,
              col.names=TRUE,
              sep="\t")
  
  saveRDS(pollen.processed,
          paste("processed/pollen.", n_genes,
                "g.", norm_technique, ".SCESet.rds", sep=""))
} else {
  write.table(df,
              file=paste(
                "processed/pollen.", n_genes, "g.txt", sep=""),
              quote=FALSE, row.names=FALSE,
              col.names=TRUE,
              sep="\t")
  
  saveRDS(pollen.processed,
          paste("processed/pollen.", n_genes, "g.SCESet.rds", sep=""))
}