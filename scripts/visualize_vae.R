library(data.table)
library(Rtsne)
library(ggplot2)

# ---------- Configuration ---------- #

exp_name <- "train_usokin-100g-standardized-1layer-vae"
model_name <- "51_UsokinVAE_FINAL"

# ---------- Load Latent Representations ---------- #

project_dir <- "~/Documents/Research/XinghuaLuLab/single-cell-deep-learning/results"
exp_dir <- paste(project_dir, "/", exp_name, sep="")
model_dir <- paste(exp_dir, "/", model_name, sep="")
setwd(model_dir)

latent_reps <- fread("latent_representations.txt",
                     header=TRUE,
                     data.table=FALSE)

x <- latent_reps[,2:(ncol(latent_reps)-2)]
cell_types <- latent_reps$cell_type
cell_subtypes <- latent_reps$cell_subtype

# colors <- brewer.pal(length(unique(cell_types)), "Set1")
# names(colors) <- unique(cell_types)

tsne <- Rtsne(x,
              dims = 2,
              perplexity=30,
              verbose=TRUE,
              pca=FALSE)
# tsne <- Rtsne(x,
#               dims = 2,
#               perplexity=30,
#               verbose=TRUE)

projections <- data.frame(tsne$Y)
colnames(projections) <- c("dim1", "dim2")

ggplot(projections, aes(x=dim1, y=dim2, color=cell_types)) + 
    geom_point() + 
    ggtitle("t-SNE Plot")

## Plotting
# plot(tsne$Y, t='n', main="t-SNE Plot", xlab="dim1", ylab="dim2")
# text(tsne$Y, labels=cell_types, col=colors[cell_types])