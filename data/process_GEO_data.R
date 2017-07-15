library(Biobase)
library(data.table)
library(scater)

setwd("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/data")

#### GSE72056: Melanoma dataset ####
melanoma_cells.TPM <- fread("GSE72056_Melanoma/original/melanoma_cells_TPM.txt",
                            header=TRUE,
                            data.table=FALSE)
rownames(melanoma_cells.TPM) <- make.unique(melanoma_cells.TPM[,1])
melanoma_cells.TPM[,1] <- NULL

# Malignant (0=unresolved, 1=no, 2=yes)
# Non-malignant cell type (1=T, 2=B, 3=Macro, 4=Endothelial, 5=CAF, 6=NK)
pdata <- data.frame(t(melanoma_cells.TPM[1:3,]))
colnames(pdata) <- c("tumor_id", "malignant", "non_malignant_cell_type")
pdata <- AnnotatedDataFrame(pdata)

edata <- as.matrix(melanoma_cells.TPM[4:nrow(melanoma_cells.TPM),])
# Un-log transform original data
edata <- 2^edata - 1

melanoma_cells.TPM.sceset <- newSCESet(tpmData=edata,
                                       phenoData=pdata,
                                       logExprsOffset = 1)
melanoma_cells.TPM.sceset <- calculateQCMetrics(melanoma_cells.TPM.sceset)

remove(pdata)
remove(edata)
remove(melanoma_cells.TPM)

melanoma_cells.TPM.sceset$malignant <- factor(melanoma_cells.TPM.sceset$malignant)
levels(melanoma_cells.TPM.sceset$malignant) <- c("unresolved", "no", "yes")

melanoma_cells.TPM.sceset$non_malignant_cell_type <- 
    factor(melanoma$non_malignant_cell_type)
levels(melanoma_cells.TPM.sceset$non_malignant_cell_type) <- c("unresolved", "T", "B", 
                                                               "Macro", "Endothelial",
                                                               "CAF", "NK")

saveRDS(melanoma_cells.TPM.sceset, "GSE72056_Melanoma/original/melanoma_cells.TPM.rds")
remove(melanoma_cells.TPM.sceset)

# melanoma <- readRDS("GSE72056_Melanoma/original/melanoma_cells.TPM.rds")

### ---------- Parameters ---------- ###
n_genes <- 500

### ---------- Gene Filtering ---------- ###
melanoma.processed <- melanoma

# Filter out dropout genes
low_exp.filter <- rowSums(tpm(melanoma.processed) == 0) > (ncol(melanoma.processed) * 0.9)
melanoma.processed <- melanoma.processed[!low_exp.filter,]

### ---------- Select Most Variable Genes ---------- ###
library(e1071)
exprs <- exprs(melanoma.processed[,melanoma$non_malignant_cell_type != "unresolved"])

genes_sd <- apply(exprs, 1, sd, na.rm=TRUE)
genes_mean <- rowMeans(exprs, na.rm=TRUE)
genes_cv <- genes_sd/genes_mean

mean_cv_model <- svm(log2(genes_cv) ~ log2(genes_mean))
predicted_log2cv <- predict(mean_cv_model, genes_mean)
gene_scores <- log2(genes_cv) - predicted_log2cv

var.filter <- names(sort(gene_scores, decreasing=TRUE)[1:n_genes])

melanoma.processed <- melanoma.processed[var.filter,]

rm(var.filter)
rm(exprs)
rm(genes_sd)
rm(genes_mean)
rm(genes_cv)
rm(mean_cv_model)
rm(predicted_log2cv)

plotTSNE(melanoma.processed, colour_by="non_malignant_cell_type")

#### GSE81861: Colorectal cancer dataset ####
crc_tumor_cells.FPKM <- fread("GSE81861_CRC/GSE81861_CRC_tumor_all_cells_FPKM.csv",
                              header=TRUE,
                              data.table=FALSE)
rownames(crc_tumor_cells.FPKM) <- crc_tumor_cells.FPKM[,1]
crc_tumor_cells.FPKM[,1] <- NULL

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
    
    # remove(chr_locations)
    # remove(gene_info)
    # remove(symbols)
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

fdata <- extractFeatureData(crc_tumor_cells.FPKM)
pdata <- extractSampleInformation(crc_tumor_cells.FPKM)

# Format expression data
edata <- crc_tumor_cells.FPKM
colnames(edata) <- rownames(pdata)
rownames(edata) <- rownames(fdata)
edata <- as.matrix(edata)

# Create ExpressionSet
crc_tumor_cells.FPKM.eset <- new("ExpressionSet", exprs=edata,
                                 phenoData=pdata, featureData=fdata)
remove(fdata)
remove(pdata)
remove(edata)
remove(crc_tumor_cells.FPKM)

saveRDS(crc_tumor_cells.FPKM.eset, "GSE81861_CRC/crc_tumor_cells.FPKM.eset.rds")

## Normal cells
crc_normal_cells.FPKM <- fread("GSE81861_CRC/GSE81861_CRC_NM_all_cells_FPKM.csv",
                               header=TRUE, data.table=FALSE)
rownames(crc_normal_cells.FPKM) <- crc_normal_cells.FPKM[,1]
crc_normal_cells.FPKM[,1] <- NULL

fdata <- extractFeatureData(crc_normal_cells.FPKM)
pdata <- extractSampleInformation(crc_normal_cells.FPKM)

edata <- crc_normal_cells.FPKM
colnames(edata) <- rownames(pdata)
rownames(edata) <- rownames(fdata)
edata <- as.matrix(edata)

crc_normal_cells.FPKM.eset <- new("ExpressionSet", exprs=edata,
                                  phenoData=pdata, featureData=fdata)
saveRDS(crc_normal_cells.FPKM.eset,
        "GSE81861_CRC/crc_normal_cells.FPKM.eset.rds")

remove(fdata)
remove(pdata)
remove(edata)
remove(crc_normal_cells.FPKM)

remove(extractSampleInformation)
remove(extractFeatureData)

#### GSE57872: Glioblastoma dataset ####
gbm_cells.TPM <- fread("GSE57872_GBM/GSE57872_GBM_all_cells_TPM.txt",
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

gbm_cells.TPM.eset <- new("ExpressionSet", exprs=edata,
                          phenoData=pdata)
remove(sample_ids)
remove(cell_origin)
remove(pdata)
remove(edata)
remove(gbm_cells.TPM)

saveRDS(gbm_cells.TPM.eset, "GSE57872_GBM/gbm_cells.TPM.eset.rds")


#### GSE75688: Breast cancer dataset ####
breast_cancer.TPM  <- fread("GSE75688_Breast_Cancer/GSE75688_Breast_Cancer.TPM.txt",
                            header=TRUE,
                            data.table=FALSE)
rownames(breast_cancer.TPM) <- breast_cancer.TPM[,1]
breast_cancer.TPM[,1] <- NULL

fdata <- breast_cancer.TPM[,1:2,drop=FALSE]
colnames(fdata) <- c("symbol", "gene_type")
fdata <- AnnotatedDataFrame(fdata)

edata <- breast_cancer.TPM[,3:ncol(breast_cancer.TPM)]
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

breast_cancer.TPM.eset <- new("ExpressionSet", exprs=edata,
                              phenoData=pdata, featureData=fdata)
saveRDS(breast_cancer.TPM.eset, "GSE75688_Breast_Cancer/breast_cancer.TPM.eset.rds")
remove(breast_cancer.TPM)
remove(edata)
remove(pdata)
remove(fdata)
remove(sample_ids)
remove(tumor_ids)

