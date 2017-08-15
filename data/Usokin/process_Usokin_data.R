library(Biobase)
library(scater)
library(data.table)

setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/data/Usokin")

#### Create SCESet From Data Table ####
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

usokin <- newSCESet(fpkmData=edata, phenoData=pdata, logExprsOffset=1)
fData(usokin)$gene_symbol <- featureNames(usokin)
usokin <- calculateQCMetrics(usokin)
remove(pdata)
remove(edata)
remove(data.RPM)

# Remove irrelevant cells
usokin <- usokin[,usokin$content == "cell"]
usokin <- usokin[,grep("outlier", usokin$pca_major_types, invert=TRUE)]
usokin <- usokin[,grep("unsolved", usokin$pca_major_types, invert=TRUE)]
usokin <- usokin[,usokin$pca_major_types != "NoN"]

# Clean up factors
usokin$pca_major_types <- droplevels(usokin$pca_major_types)
usokin$pca_neuronal_subtypes <- 
    droplevels(usokin$pca_neuronal_subtypes)
usokin$pca_all_neuronal_subtypes <- 
    droplevels(usokin$pca_all_neuronal_subtypes)

# Save SCEset
saveRDS(usokin, "original/mouse_neuronal_cells.rds")

#### Data Processing ####
usokin <- readRDS("original/mouse_neuronal_cells.rds")

### ---------- Parameters ---------- ###
n_genes <- 500
standardize <- TRUE
scale <- FALSE

### ---------- QC Plots ---------- ###
# Plot cumulative proportion of library accounted for by top N highest-expressed features
# plot(usokin, block1="pca_major_types", nfeatures=300)

# Plot QC
# plotQC(usokin, type="highest-expression", exprs_values="fpkm")

### ---------- Gene Filtering ---------- ###
usokin.processed <- usokin

# Remove genes prefixed 'r_'
usokin.processed <- usokin.processed[
    grep("r_(.*)", rownames(usokin.processed), invert=TRUE),]

# Filter out rarely expressed genes
rare_exp.filter <- rowSums(fpkm(usokin.processed) > 0) < 5
usokin.processed <- usokin.processed[!rare_exp.filter,]
 
# # Filter out commonly expressed genes
# common_exp.filter <- apply(fpkm(usokin.processed) > 0, 1, sum) >
#     (ncol(usokin.processed) * 0.95)
# usokin.processed <- usokin.processed[!common_exp.filter,]
 
# Filter out dropout genes
low_exp.filter <- rowSums(fpkm(usokin.processed) == 0) > (ncol(usokin.processed) * 0.9)
usokin.processed <- usokin.processed[!low_exp.filter,]

### ---------- Select Most Variable Genes ---------- ###
library(e1071)
exprs <- exprs(usokin.processed)

# Uncomment if zeroes should be ignored in variability calculations
exprs[exprs == 0] <- NA

genes_sd <- apply(exprs, 1, sd, na.rm=TRUE)
genes_mean <- rowMeans(exprs, na.rm=TRUE)
genes_cv <- genes_sd/genes_mean

mean_cv_model <- svm(log2(genes_cv) ~ log2(genes_mean))
predicted_log2cv <- predict(mean_cv_model, genes_mean)
gene_scores <- log2(genes_cv) - predicted_log2cv

var.filter <- names(sort(gene_scores, decreasing=TRUE)[1:n_genes])

# Check marker genes
# marker_genes <-
#     c("Nefh", "Tac1", "Mrgprd", "Th", "Vim", "B2m",
#       "Col6a2", "Ntrk1", "Calca", "P2rx3", "Pvalb")

# marker_genes[which(marker_genes %in% var.filter)]

usokin.processed <- usokin.processed[var.filter,]

# plotTSNE(usokin.processed, colour_by="pca_major_types")

rm(rare_exp.filter)
rm(low_exp.filter)
rm(var.filter)
rm(exprs)
rm(genes_sd)
rm(genes_mean)
rm(genes_cv)
rm(mean_cv_model)
rm(predicted_log2cv)

# ---------- Standardize Expression Values to N(0, 1) ---------- #
if (standardize) {
    standardize_gene <- function(x) { (x - mean(x))/sd(x) }
    exprs.scaled <- t(apply(exprs(usokin.processed), 1, standardize_gene))
    exprs(usokin.processed) <- exprs.scaled
    
    rm(standardize_gene)
    rm(exprs.scaled)   
}

# ---------- Scale Expression Values to [0, 1] ---------- #
if (scale) {
    scale_gene <- function(x) { (x - min(x))/(max(x) - min(x)) }
    exprs.scaled <- t(apply(exprs(usokin.processed), 1, scale_gene))
    exprs(usokin.processed) <- exprs.scaled
    
    rm(scale_gene)
    rm(exprs.scaled)    
}

# ---------- Create VAE-ready Benchmark Dataset ---------- #
df <- as.data.frame(t(exprs(usokin.processed)))
df$neuronal_type <- usokin.processed$pca_major_types
df$neuronal_subtype <- usokin.processed$pca_all_neuronal_subtypes

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
                "processed/usokin.", n_genes,
                "g.", norm_technique, ".txt", sep=""),
            quote=FALSE, row.names=FALSE,
            col.names=TRUE,
            sep="\t")

saveRDS(usokin.processed,
        paste("processed/usokin.", n_genes,
              "g.", norm_technique, ".SCESet.rds", sep=""))




