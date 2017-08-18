library(Biobase)
library(data.table)
library(scater)

setwd(paste("~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/",
            "data/GSE94820_PBMC", sep=""))

pbmc.TPM <- fread("original/cell_type_exp_matrix.txt",
                  header=TRUE,
                  data.table=FALSE)
rownames(pbmc.TPM) <- pbmc.TPM$Gene
pbmc.TPM$Gene <- NULL

cell_subtypes <- colnames(fread("original/cell_subtype_exp_matrix.txt",
                          header=TRUE,
                          data.table=FALSE))
# Remove leading "gene" column
cell_subtypes <- cell_subtypes[2:length(cell_subtypes)]
cell_subtypes <- gsub("_S([0-9]+)(.*)", "", cell_subtypes)

cell_types <- colnames(pbmc.TPM)
cell_types[grep("pDC", cell_types)] <- "pDC"
cell_types[grep("CD141", cell_types)] <- "CD141Pos_DC"
cell_types[grep("CD1C", cell_types)] <- "CD1CPos_DC"
cell_types[grep("DoubleNeg", cell_types)] <- "CD141NegCD1CNeg_DC"
cell_types[grep("Mono_classical", cell_types)] <- "Mono_Classical"
cell_types[grep("Mono_neoclassical", cell_types)] <- "Mono_Neoclassical"
cell_types[grep("Mono_intermediate", cell_types)] <- "Mono_Intermediate"
cell_types[grep("Mono_nonclassical", cell_types)] <- "Mono_Nonclassical"

pdata <- data.frame(cell_type=cell_types, cell_subtype=cell_subtypes)
rownames(pdata) <- colnames(pbmc.TPM)
pdata <- AnnotatedDataFrame(pdata)