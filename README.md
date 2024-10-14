# FeynGNN
Graph neural network for understanding and calculation of Feynman diagram.

# Release 1.0
## 1.0
The directory **main** contains the four python files of the FeynGNN.  PhysicsProcess.py contains four functions that create the graph and compute the amplitudes of chosen scattering process, including:
1. Pair annihilation: $\mu^+\mu^-\to e^+e^-$.
2. Columb scattering: $e^+\mu^+\to e^+\mu^+,e^-\mu^-\to e^-\mu^-$.
3. Bhabha scattering: $e^+ e^-\to e^+ e^-,\mu^+\mu^-\to \mu^+\mu^-$.
4. Moller scattering: $e^+e^+\to e^+ e^+,\mu^-\mu^-\to \mu^-\mu^-$.
5. Compton scattering: $e^\pm \gamma\to e^\pm \gamma, \mu^\pm\gamma\to \mu^\pm\gamma$.
6. Photon creation: $e^\pm e^\mp\to 2\gamma, \mu^\pm \mu^\mp\to 2\gamma$

The DataHelper.py contains the graphset class and can create, save, and load graph datasets. The GNNModel.py define the graph neural network model and the input of hyperparameters.

The **data** directory contains the created data. The data is classified into different csv files for different process. The features of nodes and edges are aligned in squences.

The **model** directory contains the parameters of trained model and the corresponding hyperparameters. The validating figures are stored in **validatingFIG** .

## 1.1 (Incoming)
Aim: More explainations written in the coding files.

## 1.2
Aim: Add the Compton scattering and annihilation into photons into the PhysicsProcess.py and build a model with great performance.
