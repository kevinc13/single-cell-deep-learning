library(Rtsne)
library(ggplot2)

tsne_latent_reps <- function(x, labels, base_dir, plot=TRUE, pca=TRUE) {
    results_dir <- paste(base_dir, "/tSNE", sep="")
    results_filepath <- paste(results_dir, "/tSNE_results.Rdata", sep="")
    
    dir.create(results_dir, showWarnings=FALSE)
    
    print_divider("=")
    cat("t-SNE: ")
    catln(dim(x)[1], " samples | ", dim(x)[2], " features")
    print_divider("-")
    
    # ---------- Run t-SNE ---------- #
    if (!file.exists(results_filepath)) {
        tsne_results <- Rtsne(x,
                              dims = 2,
                              perplexity=30,
                              verbose=TRUE,
                              pca=pca)
        save(tsne_results, file=results_filepath)
        catln("Finished running t-SNE (Barnes Hut)")
    } else {
        catln("Loading previous t-SNE results...")
        load(results_filepath)
    }
    print_divider("-")

    projections <- data.frame(tsne_results$Y)
    colnames(projections) <- c("dim1", "dim2")
    
    # ---------- Plot ---------- #
    if (plot) {
        cat("Plotting t-SNE projections...")
        plot <- ggplot(projections, aes(x=dim1, y=dim2, color=labels)) +
            geom_point() +
            ggtitle("t-SNE Plot")
        ggsave("tSNE_plot.png",
               plot=plot, device="png",
               path = results_dir,
               width=9,
               height=8,
               units="in",
               dpi=300)
        catln("done")
    }
    
    print_divider("=")
    
    return(list(tsne_results=tsne_results,
                projections=projections))
}


