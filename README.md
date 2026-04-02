Summary:

Determine congestion on the electrical grids, by using graph-based 
Algorithms.

Dataset.
The dataset was based on the IEEE-32 buses model and optimal DC Power Flow.
the IEEE-32 model contains n buses, x generators and (insert image)

Algorithms:

1. Message Passing Neural Networks (MPNN)
2. Graph Attention Networks (GAT)
3. Edge-conditional Graph Neural Networks (Edge-conditioned GNN)
4. Graph Sample and Aggregate (GraphSAGE)
5. Graph Convolutional Networks (GCN)

Hyperparameters:

    General Params for Graph Neural Networks:
        - Learning Rate: 0.0001 - 0.01
        - Weight Decay: 0.0001 - 0.1
        - Number of Epochs: 100 - 1000
        - Batch Size: 32 - 256
        - Number of Layers: 2 - 5
        - Hidden Size: 64 - 512
        - Dropout: 0.1 - 0.5
        - Number of message passing rounds: 3 - 10
        - Number of attention heads: 4 - 8 (only for GAT)

Results:

Conclusion:

Next Steps:

    
    
    