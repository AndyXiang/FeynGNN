# FeynGNN
Graph neural network for understanding Feynman diagram

# Project Design

## Stage 1.0
Reconstruct the existing code. The reconstruction shall contain three parts:
1. Feynman graph class;
2. data set up package, incluing data creator, saver and loader;
3. model file, including model architecture, batchlizer of data and training program;
4. training process file and validating, should be visualised.
This stage shall be accomplished before 2024.8.31.(**Accomplished**)

## Stage 1.1
1. Careful setup and describtion of datatset.
2. Visualision and module of hyperparameters.
3. Train with full dataset and find the highest performance.
4. Train with masked dataset and test the generalization ability of the model.

## Stage 2
Generalize the model to all 2->2 processes. Try different architecture, incluing different GCN layer, different pooling module. The baseline target is to achieve at least 0.95 RSquare rate on each process. 

*Overall Steps*:
1. Calculate all desire processes, and encoding them. Step due: 2024.10.31.
2. Train the old architecture on the generalized dataset with different hyper-parameters. (*One problem needed to discuss is the time we spend on the old model.*) Step due: 2024.12.31.
3. Use new architecture to imporve the performance. Step due: 2025.3.31.



## Stage 3
This stage is to generalize the model to Feynman diagrams with different topologies. 

### Overall Steps
1. Reconstruct the model file to adapt the new goal.
2. The generating of diagrams may be accomplished by Mathematica. To transfer the data, need to invent a effcient way to encoding the diagram datas in Mathematica to string data and decoding the string data in python. 
3. Apply new architecture to establish the new model.