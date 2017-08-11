#Below is a very simple example of using Seurat to analyze a beautiful human pancreas dataset generated using inDrop, from the Yanai lab at NYUMC.

library(Seurat)
library(Matrix)

# Sample workflow for leading in files from GEO
######################################################################
all.files=list.files("~/Downloads/YanaiData/data/")
all.data=data.frame()
for(i in all.files[1:4]) {
  dat=read.csv(paste("~/Downloads/YanaiData/data/",i,sep=""))
  all.data=rbind(all.data,data.frame(dat))
  print(i)
}

new.data=t(all.data[,c(-1,-2,-3)])
colnames(new.data)=all.data[,1]
pancreas.data <- new.data
pancreas.md <- all.data[,2:3]; rownames(pancreas.md)=all.data[,1]
######################################################################

pancreas.data <- Matrix(pancreas.data, sparse = T)
pancreas <- new("seurat", raw.data = pancreas.data)
pancreas <- AddMetaData(pancreas, metadata = pancreas.md)
pancreas <- Setup(pancreas, min.genes = 500, do.scale = F, project = "PANCREAS", do.center = F)
pancreas <- MeanVarPlot(pancreas, x.low.cutoff = 0.1)
pancreas <- RegressOut(pancreas, latent.vars = c("orig.ident", "nUMI"), genes.regress = pancreas@var.genes, model.use = "negbinom")
pancreas <- PCAFast(pancreas, pcs.compute = 30)
pancreas <- RunTSNE(pancreas, dims.use = 1:19, do.fast = T)
pancreas <- FindClusters(pancreas, pc.use = 1:19, save.SNN = T, do.sparse = T)

#color by cluster ID, annotated cluster from the manuscript, or batch
#Can switch the identity class using SetAllIdent if desired
TSNEPlot(pancreas,do.label = T)
TSNEPlot(pancreas,group.by="assigned_cluster")
TSNEPlot(pancreas,group.by="orig.ident")

#Find Markers of ductal cell subcluster, using the negative binomial test
#only test genes with a 20% difference in detection rate to speed-up (optional)
ductal.markers=FindMarkers(pancreas,5,12,test.use = "negbinom",min.diff.pct = 0.2)

#Visualize canonical and new markers
FeaturePlot(pancreas, c("GCG", "INS","TFF1","PRSS1","VGF","TRIB3","DDR1","CRYBA2","SLC30A8"),cols.use = c("lightgrey","blue"),nCol = 3)

#Can save the object for future loading
save(pancreas, file = "~/Projects/datasets/pancreas.Robj")


