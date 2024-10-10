# BiGCN
BiGCN is an innovative Graph Convolutional Network (GCN)-based approach designed to effectively integrate B cell transcriptomes with BCR repertoires. BiGCN converts the integration of B cells and BCRs into a graph fusion problem and produces an explicit integrated latent embedding that encapsulates information from both modalities.
## Dependencies 
python(v3.7)
### python packages
torch, numpy, xlrd, sklearn
## Data preparation
BCR embedding: derived from scBCR-seq data by ‘Atchley factors’  and contrastive learning.

PCA of B cells: the PCA representation of gene expression matrix of B cells from scRNA-seq data.

crudegraph: the graph constructed for BCR clonatypes based on the V/J genes.

## BiGCN Usage

