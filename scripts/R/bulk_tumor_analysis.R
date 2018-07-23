base.dir <- "~/Documents/Research/XinghuaLuLab/single-cell-deep-learning"
setwd(base.dir)

library(data.table)
library(survminer)

# Configuration ----------------------------------------------------------------
# Shortcut for repetitive filepaths
name <- "GBM"
marker.genes.file <- paste("results/", tolower(name),
                           "/vae/Train1000g", name, "VAE/", 
                           name, "VAE_Final/consensus_clusters.txt", sep="")
clusters.col <- "consensus_cluster"

# Load Data --------------------------------------------------------------------
clusters.df <- fread(clusters.file, header=TRUE, sep="\t", data.table=FALSE)
rownames(clusters.df) <- clusters.df[,1]
clusters.df[,1] <- NULL
clusters <- factor(clusters.df[,clusters.col])


#### Survival Analysis ####
# Extract tissue specific survival data
tumor_clusters <- geneexp_tumor_clusters
clin_data <- fread("data/TCGA/pancancer_clinical_data.txt", data.table=FALSE, header=TRUE)
tumor_clin_data <- clin_data[clin_data$bcr_patient_barcode %in% names(tumor_clusters),]
tumor_survival_data <- as.data.frame(cbind(tumor_clin_data$bcr_patient_barcode,
                                           tumor_clin_data$vital_status,
                                           tumor_clin_data$days_to_death,
                                           tumor_clin_data$days_to_last_followup))
colnames(tumor_survival_data) <- c("ID", "Status", "days_to_death", "days_to_last_followup")

tumor_survival_data$days_to_death_or_last_followup <- NA
for (i in 1:nrow(tumor_survival_data)) {
  tumor_survival_data$days_to_death_or_last_followup[i] <- ifelse(
    is.na(as.numeric(as.character(tumor_survival_data$days_to_death))[i]),
    as.numeric(as.character(tumor_survival_data$days_to_last_followup))[i],
    as.numeric(as.character(tumor_survival_data$days_to_death))[i]
  )
}

tumor_survival_data$death_event <- ifelse(tumor_survival_data$Status == "Alive", 1, 0)

# Make sure dimensions are same
tumor_clusters <- tumor_clusters[names(tumor_clusters) %in% clin_data$bcr_patient_barcode]

tumor_survfit <- survfit(Surv(tumor_survival_data$days_to_death_or_last_followup,
                              tumor_survival_data$death_event)~tumor_clusters)

log_rank <- survdiff(Surv(tumor_survival_data$days_to_death_or_last_followup,
                          tumor_survival_data$death_event)~tumor_clusters)
log_rank_p_val <- 1 - pchisq(log_rank$chisq, length(log_rank$n) - 1)

# png(filename="~/Desktop/luad_surv_analysis.png",
#     width=11, height=6, units="in",
#     pointsize=4, res=300)
# ggsurvplot(tumor_survfit,
#            pval=TRUE,
#            pval.coord=c(400, 0.2),
#            risk.table=FALSE,
#            main="Survival Analysis of LUAD Clusters",
#            xlab="Days",
#            break.time.by=1000,
#            legend = c(0.85, 0.85),
#            legend.labs=c("1 (n = 221)", "2 (n = 149)", "3 (n = 48)", "4 (n = 82)"),
#            legend.title="Cluster",
#            
#            font.main=18,
#            font.x=16, font.y=16,
#            font.tickslab=14, font.legend=14,
#            palette=c("#7ecefd", "#f20253", "#677480", "#ff9715"))
# dev.off()

png(filename="~/Desktop/luad_geneexpcluster_surv_analysis.png",
    width=11, height=6, units="in",
    pointsize=4, res=300)
ggsurvplot(tumor_survfit,
           pval=TRUE,
           pval.coord=c(400, 0.2),
           risk.table=FALSE,
           main="Survival Analysis of Gene Expression-Based LUAD Clusters",
           xlab="Days",
           break.time.by=1000,
           legend = c(0.85, 0.7),
           legend.title="Cluster",
           
           font.main=18,
           font.x=16, font.y=16,
           font.tickslab=14, font.legend=14)
dev.off()