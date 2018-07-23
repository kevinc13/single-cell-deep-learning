library(data.table)
library(ggplot2)
library(gridExtra)
library(grid)

plotVAELosses <- function(model.dir) {
  setwd(model.dir)
  
  # Load training log
  train.log <- fread("training.log", header=TRUE, data.table=FALSE)
  
  # Create loss v. epoch plot (if one doesn't exist already)
  if (!file.exists("losses_v_epoch.png")) {
    plot.data <- data.frame(cbind(train.log$reconstruction_loss,
                                  train.log$kl_divergence_loss))
    colnames(plot.data) <- c("recon_loss", "kl_loss")
    plot.data <- stack(plot.data)
    plot.data <- cbind(rep(train.log$epoch, 2), plot.data)
    colnames(plot.data) <- c("epoch", "loss", "type")
    
    loss.v.epoch.plot <- ggplot(
      plot.data, aes(x=epoch, y=loss, color=type)) +
      geom_line() +
      ggtitle("VAE Training Losses v. Epoch")
    ggsave("losses_v_epoch.png",
           plot=loss.v.epoch.plot, device="png", path = ".",
           width=10, height=6, units="in", dpi=300)
  }
}

plotAAELosses <- function(model.dir) {
  setwd(model.dir)
  
  ae.log <- fread("autoencoder_model.training.log", 
                  header=TRUE, data.table=FALSE)
  disc.log <- fread("discriminator_model.training.log",
                    header=TRUE, data.table=FALSE)
  
  if (!file.exists("losses_v_epoch.png")) {
    ae.loss.plot <- ggplot(
      data.frame(epoch=ae.log$epoch,
                 loss=ae.log$loss,
                 type="recon_loss"),
      aes(x=epoch, y=loss, color=type)) +
      geom_line() +
      ggtitle("AAE Recon Loss v. Epoch")
    disc.loss.plot <- ggplot(
      data.frame(epoch=disc.log$epoch,
                 loss=disc.log$loss,
                 type="disc_loss"), 
      aes(x=epoch, y=loss, color=type)) +
      geom_line() +
      ggtitle("AAE Disc Loss v. Epoch")
    g = grid.arrange(ae.loss.plot, disc.loss.plot, ncol=2)
    ggsave("losses_v_epoch.png",
           plot=g, device="png", path = ".",
           width=12, height=6, units="in", dpi=300)
  }
}