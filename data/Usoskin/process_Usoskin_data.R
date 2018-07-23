library(Biobase)
library(scater)
library(scran)
library(data.table)

setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/data/Usoskin")

#### Create SCESet From Data Table ####
if (!file.exists("original/mouse_neuronal_cells.rds")) {
  data.RPM <- fread("original/mouse_neuronal_cells.RPM.txt",
                    header=TRUE, data.table=FALSE)
  rownames(data.RPM) <- data.RPM[,1]
  data.RPM[,1] <- NULL
  
  pdata <- data.frame(t(data.RPM[1:9,]))
  colnames(pdata) <- c("picking_sessions", "picking_ToC",
                       "library", "reads", "sex",
                       "pca_major_types", "pca_neuronal_subtypes", 
                       "pca_all_neuronal_subtypes", "content")
  pdata <- AnnotatedDataFrame(pdata)
  
  edata <- as.matrix(data.RPM[10:nrow(data.RPM),])
  edata <- apply(edata, 2, as.numeric)
  rownames(edata) <- rownames(data.RPM[10:nrow(data.RPM),])
  
  usoskin <- newSCESet(fpkmData=edata, phenoData=pdata, logExprsOffset=1)
  fData(usoskin)$gene_symbol <- featureNames(usoskin)
  usoskin <- calculateQCMetrics(usoskin)
  remove(pdata)
  remove(edata)
  remove(data.RPM)
  
  # Remove irrelevant cells
  usoskin <- usoskin[,usoskin$content == "cell"]
  usoskin <- usoskin[,grep("outlier", usoskin$pca_major_types, invert=TRUE)]
  usoskin <- usoskin[,grep("unsolved", usoskin$pca_major_types, invert=TRUE)]
  usoskin <- usoskin[,usoskin$pca_major_types != "NoN"]
  
  # Clean up factors
  usoskin$pca_major_types <- droplevels(usoskin$pca_major_types)
  usoskin$pca_neuronal_subtypes <- 
    droplevels(usoskin$pca_neuronal_subtypes)
  usoskin$pca_all_neuronal_subtypes <- 
    droplevels(usoskin$pca_all_neuronal_subtypes)
  
  # Save SCEset
  saveRDS(usoskin, "original/mouse_neuronal_cells.rds")
}

#### Data Processing ####
usoskin <- readRDS("original/mouse_neuronal_cells.rds")

### ---------- Parameters ---------- ###
n_genes <- 1000
standardize <- FALSE
scale <- FALSE

### ---------- Gene Filtering ---------- ###
usoskin.processed <- usoskin

# Remove genes prefixed 'r_'
usoskin.processed <- usoskin.processed[
    grep("r_(.*)", rownames(usoskin.processed), invert=TRUE),]
 
# Filter out dropout genes
low_exp.filter <- rowSums(fpkm(usoskin.processed) == 0) > 
  (ncol(usoskin.processed) * 0.9)
usoskin.processed <- usoskin.processed[!low_exp.filter,]

### ---------- Select Most Variable Genes ---------- ###
var.fit <- trendVar(usoskin.processed, use.spikes=FALSE)
var.table <- decomposeVar(usoskin.processed, var.fit)
# var.table <- var.table[var.table$FDR <= 0.05,]

var.filter <- rownames(var.table)[order(var.table$bio, decreasing=TRUE)]
var.filter <- var.filter[1:n_genes]

usoskin.processed <- usoskin.processed[var.filter,]

remove(design)
remove(var.fit)
remove(var.table)
remove(var.filter)

# library(e1071)
# exprs <- exprs(usoskin.processed)
# 
# # Uncomment if zeroes should be ignored in variability calculations
# exprs[exprs == 0] <- NA
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

# Check marker genes
# marker_genes <-
#     c("Nefh", "Tac1", "Mrgprd", "Th", "Vim", "B2m",
#       "Col6a2", "Ntrk1", "Calca", "P2rx3", "Pvalb")

# marker_genes[which(marker_genes %in% var.filter)]

# usoskin.processed <- usoskin.processed[var.filter,]

# plotTSNE(usoskin.processed, colour_by="pca_major_types")

# rm(rare_exp.filter)
# rm(low_exp.filter)
# rm(var.filter)
# rm(exprs)
# rm(genes_sd)
# rm(genes_mean)
# rm(genes_cv)
# rm(mean_cv_model)
# rm(predicted_log2cv)

# ---------- Standardize Expression Values to N(0, 1) ---------- #
if (standardize) {
    standardize_gene <- function(x) { (x - mean(x))/sd(x) }
    exprs.scaled <- t(apply(exprs(usoskin.processed), 1, standardize_gene))
    exprs(usoskin.processed) <- exprs.scaled
    
    rm(standardize_gene)
    rm(exprs.scaled)   
}

# ---------- Scale Expression Values to [0, 1] ---------- #
if (scale) {
    scale_gene <- function(x) { (x - min(x))/(max(x) - min(x)) }
    exprs.scaled <- t(apply(exprs(usoskin.processed), 1, scale_gene))
    exprs(usoskin.processed) <- exprs.scaled
    
    rm(scale_gene)
    rm(exprs.scaled)    
}

# ---------- Create VAE-ready Benchmark Dataset ---------- #
df <- as.data.frame(t(exprs(usoskin.processed)))
df$cell_type <- usoskin.processed$pca_major_types
df$cell_subtype <- usoskin.processed$pca_all_neuronal_subtypes

df <- cbind(rownames(df), df)
colnames(df)[1] <- "cell_id"

norm_technique <- NA
if (scale) {
  norm_technique <- "scaled"
} else if (standardize) {
  norm_technique <- "standardized"
}

if (!is.na(norm_technique)) {
  write.table(df,
              file=paste(
                "processed/usoskin.", n_genes,
                "g.", norm_technique, ".txt", sep=""),
              quote=FALSE, row.names=FALSE,
              col.names=TRUE,
              sep="\t")
  
  saveRDS(usoskin.processed,
          paste("processed/usoskin.", n_genes,
                "g.", norm_technique, ".SCESet.rds", sep=""))
} else {
  write.table(df,
              file=paste(
                "processed/usoskin.", n_genes, "g.txt", sep=""),
              quote=FALSE, row.names=FALSE,
              col.names=TRUE,
              sep="\t")
  
  saveRDS(usoskin.processed,
          paste("processed/usoskin.", n_genes, "g.SCESet.rds", sep=""))
}






