# ------------------------------------------------------------------------------
# This script runs PCA and saves the reduced dimensional representations
# ------------------------------------------------------------------------------
base.dir <- "~/Documents/Research/XinghuaLuLab/single-cell-deep-learning"
setwd(base.dir)

library(data.table)
library(scater)

# Configuration ----------------------------------------------------------------
dataset.file <- "data/GSE57872_GBM/processed/gbm.1000g.centered.SCESet.rds"
pca.ntop <- 1000
pca.ncomponents <- 10
pca.output.dir <- "results/gbm/pca/pca_1000g_10comp"

dataset.label.cols <- c("cell_origin")
output.label.cols <- c("tumor_id")

# Run PCA ----------------------------------------------------------------------
dataset <- readRDS(dataset.file)
dataset.pca <- plotPCA(dataset, ncomponents=pca.ncomponents,
                       return_SCESet = TRUE, scale_features=TRUE,
                       draw_plot=FALSE, ntop=pca.ntop)

latent.reps <- as.data.frame(reducedDimension(dataset.pca))

# Save PCA Latent Representations ----------------------------------------------
df <- cbind(rownames(latent.reps), latent.reps, 
            pData(dataset)[dataset.label.cols])
colnames(df)[1] <- "cell_id"
colnames(df)[(pca.ncomponents + 2):ncol(df)] <- output.label.cols

dir.create(pca.output.dir, showWarnings=FALSE)

write.table(df, file=paste(pca.output.dir, 
                           "/latent_representations.txt", sep=""),
            row.names=FALSE, col.names=TRUE, sep="\t", quote=FALSE)


