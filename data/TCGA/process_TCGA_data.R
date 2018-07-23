library(Biobase)
library(data.table)
library(limma)

setwd(paste("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/",
            "data/TCGA", sep=""))

if (!file.exists("original/tcga.rds")) {
  pancan.TPM <- fread("original/PANCAN_rsem_gene.TPM.txt",
                      header=TRUE,
                      data.table=FALSE, verbose=TRUE)
  pancan.clinical <- fread("original/PANCAN_clinical_matrix.txt",
                           header=TRUE,
                           data.table=FALSE, verbose=TRUE)
  pancan.map <- fread("original/PANCAN_gene_probe_map.txt",
                      header=TRUE,
                      data.table=FALSE, verbose=TRUE)
  
  # Process expression data
  # TPM values are transformed by log2(x+0.001)
  edata <- pancan.TPM
  rownames(edata) <- edata[,1]
  edata[,1] <- NULL
  
  # Process phenotypic information
  pdata <- pancan.clinical
  rownames(pdata) <- pdata[,1]
  pdata[,1] <- NULL
  edata <- edata[,colnames(edata) %in% rownames(pdata)]
  pdata <- pdata[colnames(edata),]
  # Remove underscores
  colnames(pdata) <- gsub("^_(.*)", "\\1", colnames(pdata))
  # Remove samples with no specified 'primary disease'
  pdata <- pdata[!is.na(pdata$primary_disease),]
  edata <- edata[,colnames(edata) %in% rownames(pdata)]
  pdata <- AnnotatedDataFrame(pdata)
  
  # Process feature information
  fdata <- pancan.map
  rownames(fdata) <- fdata[,1]
  fdata[,1] <- NULL
  edata <- edata[rownames(fdata),]
  fdata <- AnnotatedDataFrame(fdata)
  
  stopifnot(sum(rownames(edata) != rownames(fdata)) == 0)
  stopifnot(sum(colnames(edata) != rownames(pdata)) == 0)
  
  pancan.eset <- ExpressionSet(as.matrix(edata), phenoData=pdata,
                               featureData=fdata)
  
  tcga.genes <- alias2SymbolTable(fData(pancan.eset)$gene)
  fData(pancan.eset)$official_gene_symbol <- tcga.genes
  pancan.eset <- pancan.eset[!is.na(fData(pancan.eset)$official_gene_symbol),]

  saveRDS(pancan.eset, "original/tcga.rds")
  
  gene_list <- fData(pancan.eset)$gene
  saveRDS(gene_list, "original/tcga_gene_symbol_list.rds")
  
  ensembl_list <- rownames(pancan.eset)
  saveRDS(ensembl_list, "original/tcga_gene_ensembl_list.rds")
  
  remove(edata)
  remove(pdata)
  remove(fdata)
  remove(pancan.TPM)
  remove(pancan.clinical)
  remove(pancan.map)
  remove(pancan.eset)
  remove(gene_list)
  remove(ensembl_list)
  remove(tcga.genes)
}

# Process GBM data
if (!file.exists("original/gbm.logtpm.rds")) {
  tcga <- readRDS("original/tcga.rds")
  
  gbm <- tcga[,tcga$primary_disease == "glioblastoma multiforme"]
  # Re-transform the data to log2(TPM + 1)
  tpm <- 2^exprs(gbm) - 0.001
  exprs(gbm) <- log2(tpm + 1)
  rm(tpm)
  
  saveRDS(gbm, "original/gbm.logtpm.rds") 
  rm(tcga)
} else {
  gbm <- readRDS("original/gbm.logtpm.rds")
}

# Center
# center_gene <- function(x) { x - mean(x) }
# exprs.centered <- t(apply(exprs(gbm), 1, center_gene))
# exprs(gbm) <- exprs.centered

# rm(center_gene)
# rm(exprs.centered)
# 
# # Classify each GBM sample according to Verhaak subtype signatures
# verhaak <- fread("../GSE57872_GBM/verhaak_subtype_signatures.txt",
#                  header=TRUE, data.table=FALSE, sep="\t")
# rownames(verhaak) <- verhaak[,1]
# verhaak[,1] <- NULL
# 
# verhaak <- verhaak[rownames(verhaak) %in% fData(gbm)$gene,]
# gbm.verhaak <- gbm[fData(gbm)$gene %in% rownames(verhaak),]
# 
# # Remove duplicates
# duplicates <- duplicated(fData(gbm.verhaak)$gene)
# gbm.verhaak <- gbm.verhaak[!duplicates,]
# 
# projection <- cor(exprs(gbm.verhaak), verhaak)




