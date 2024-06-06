# cyclone-tracking

A scalable and end-to-end approach to tropical cyclone tracking using Graph Attention Networks

# Data Loading

- We have successfully loaded the [IBTracs](data/ibtracs.ipynb) dataset, however, it is not in grid form and thus not representable as a graph.
- We found a starting point for working with the [era5](data/era5.ipynb) dataset.

### Grid types

- Full 1440x721 lon/lat grid
- Equiangular conservative with(out) poles at subsampled lon/lat resolutions
- Reduced Gaussian grid (ignored by us)
- Icosahedric mesh made us and GraphCast
