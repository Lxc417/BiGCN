# BiGCN
Regarding the gene expression and BCR-seq as two-view features of B cells and transforming them into one-to-one corresponding graphs, BiGCN is an innovative Graph Convolutional Network (GCN)-based approach designed
to effectively fuse the two graphs into an explicit integraed embedding. This integrated embedding has improved generalization ability, not only shares information of both gene expression and BCRs but also contains
potential information hidden in respective data. BiGCN is capable of capturing accurate B cell development trajectory.
## Dependencies 
python (we use Python v3.7, and you can use other versions of Pyhon but need mathced packages)
### python packages
torch

numpy  

sklearn

## Data preparation

We first preprocess (the detailed preprocessing method is described in Methods section of manuscript) and extract paired scRNA-seq and scBCR-seq data, 
then BiGCN takes the following three data files as input:

1. BCR_embedding.xlsx: As Benisse did for BCR data, we also encoded BCR sequences by ‘Atchley factors’ which represented each amino acid with five numerical values,
   and then transformed this Atchley factor matrix into a 20-dimensional embedding using a contrastive learning model. We denote this 20-dimensional embedding as the original BCR embedding.
   
2. exp_pca.csv: the PCA representation of gene expression matrix of B cells from scRNA-seq data.

3. crudegraph.csv: the graph constructed for BCR clonatypes based on the V/J genes. The BCR crude graph is constructed among BCR clonotypes,
    in which the node denotes a BCR clonotype (V_CDR3H_J) and an edge connected in BCR crude graph means the two BCR clonotypes share the same V/J genes.
   The weight of edge in the BCR crude graph is defined as the cosine similarity calculated based on the original BCR embedding. 

## BiGCN Usage
### construct B-cell and BCR graphs
#### Data conversion

data_process.py

input_data.py

#### construct graphs
graphStructure.py

### integrating B-cell and BCR graphs
main.py

model.py


