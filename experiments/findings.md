# Findings from running experiments
- Learning rate should not be greater than 1e-4, 1e-5 is preferred. We get too much jitter with bigger learning rates
- Use largest batch size possible
    - Gradient descent seems much more stable
    - Faster processing
- using the geometric dataloader inside of the graph layer breaks gradient descent
- running the model without a GCN of any kind in the mesh gives better results
- MSE loss will be smaller if you have fewer features (seems logical)
- Datasets with intrinsically smaller observation widnows perform better
    - This is because the windows taken from era5 do not overlap, so smaller windows in same amount of time means more windows. That means more samples
    - More samples of small obs window preffered to fewer with large window

## What remains to be found out
- How does adding more features influence accuracy?
    - MSE loss goes up, so how can we measure in a standard way?
    - Proposed solution: divide MSE by number of features to get more standard metric
- How does GAT perform vs GCN?
- How does more attention heads perform
- If large batch size increases gradient descent smoothness, can we use larger learning rates?