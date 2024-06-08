# Data Directory Guide

## Getting Started

To use the methods from this directory, look only for the [data_loading](data_loading.py) file, which has method _..._ that can get you from a url of the specific dataset you want to work with to a pytorch weather dataset. If you want more control over the dataset, use the other methods in sequence with your arguments to end up with your dataset.

## Files

- [ibtracs](ibtracs.ipynb): Notebook showing how to load ibtracs dataset
- [era5](era5.ipynb): Notebook showing how to download era5 dataset
- [data_viz](data_viz.ipynb): Notebook showing how to visualize grid data. We could use this to visualize the decoded grid (or even mesh) to visualize results
- [data_process](data_process.ipynb): Notebook showcasing how you can get a barebones version of the dataset
- [data_loading](data_loading.ipynb): Notebook where we load the barebones dataset, create lag features, split it, scale it, and arrive to PyTorch's datasets.
- [full_pipeline](full_pipeline.ipynb): How to get from a url to a dataset.

### TODO

- Alternative datasets
  - torch.geometric
