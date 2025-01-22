# BiGCN
Regarding the gene expression and BCR-seq as two-view features of B cells and transforming them into one-to-one corresponding graphs, BiGCN is an innovative Graph Convolutional Network (GCN)-based approach designed
to effectively fuse the two graphs into an explicit integraed embedding. This integrated embedding has improved generalization ability, not only shares information of both gene expression and BCRs but also contains
potential information hidden in respective data. BiGCN is capable of capturing accurate B cell development trajectory.
## Dependencies 
python(v3.7)
### python packages
torch

numpy  

sklearn

## Data preparation
BCR embedding: derived from scBCR-seq data by ‘Atchley factors’  and contrastive learning.

PCA of B cells: the PCA representation of gene expression matrix of B cells from scRNA-seq data.

crudegraph: the graph constructed for BCR clonatypes based on the V/J genes.

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


