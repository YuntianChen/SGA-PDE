# SGA-PDE
Symbolic genetic algorithm for discovering open-form partial differential equations (SGA-PDE) is a model to discover open-form PDEs directly from data without prior knowledge about the equation structure. 
For more introduction about this model, please refer to the paper [Any equation is a forest: Symbolic genetic algorithm for discovering open-form partial differential equations (SGA-PDE)](https://arxiv.org/abs/2106.11927).

SGA-PDE focuses on the representation and optimization of PDE. Firstly, SGA-PDE uses symbolic mathematics to realize the flexible representation of any given PDE, transforms a PDE into a forest, and converts each function term into a binary tree. Secondly, SGA-PDE adopts a specially designed genetic algorithm to efficiently optimize the binary trees by iteratively updating the tree topology and node attributes. The SGA-PDE is gradient-free, which is a desirable characteristic in PDE discovery since it is difficult to obtain the gradient between the PDE loss and the PDE structure.

Why do we need SGA-PDE? 
1. Partial differential equations (PDEs) are concise and understandable representations of domain knowledge, which are essential for deepening our understanding of physical processes and predicting future responses. However, many systems in practical engineering applications are too complex and irregular, resulting in complicated forms of PDEs (governing equations) describing the mapping between variables, which are difficult to derive directly from theory. Therefore, researchers often collect data through physical experiments and obtain governing equations by analyzing the experimental data. 
2. There are mainly two kinds of methods of automatic mining PDE: sparse regression and genetic algorithm. The sparse regressions (e.g., LASSO and STRidge) require the user to determine the approximate form of the governing equation in advance, and then give all possible differential operators as the function terms in the candidate set. It is impossible to find the function terms that do not exist in the candidate set from the data in these methods. The current evolutionary-strategy-based methods are still unable to mine open-form PDEs from data (e.g., the PDE with compound function or fractional structure).

If you encounter any problems in using the code, please contact Yuntian Chen: cyt_cn@126.com.


# The guide for SGA-PDE:
## Contents
(1) configure.py: Experimental parameter setting and model hyperparameter setting. Contains the process of selecting a dataset (Burgers, KdV, Chafee-infante, PDE_divide, PDE_compound).

(2) setup.py: 1. Load data from Data_generator; 2. Evaluate the fitness between a PDE and observations (calculate the error between the left and right side of the PDE). The gradients involved in the PDE can be calculated by finite difference or autograd; 3. Draw figures of the gradients of different orders, the left and right side of the given PDE; 4. Set the operators and operands in the SGA.

(3) tree.py: Define the node class and tree class in the binary tree, which correspond to the operators/operands and function terms in the PDE, respectively. Define tree2str_merge, which can transform the binary tree into a partial differential function term.

(4) PDE.py: Define the PDE class. Define the evaluate_mse function to evaluate the fitness of a generated PDE.

(5) PDE_find.py: Use the finite difference method to calculate the gradients in the function terms. Use STRidge to find the optimal combination of all function terms (trees) in the current iteration step, and evaluate the fitness between the optimal combination (i.e., the discovered PDE) of the current iteration step and the observations.
 
(6) SGA.py: The main program of SGA-PDE. Define the SGA class. The crossover operation in SGA-PDE is defined in 'corss_over' function. The mutation and replacement operation in SGA-PDE are defined in 'change' function.

(7) Data_generator.py: Generate the datasets. If Metadata is used, compare the metadata with original data.

(8) MetaNN_generator.py: Optional module. Build the neural network (surrogate model) for generating Metadata and evaluate the neural network by RMSE and R2. This module is not used by default. More details about the Metadata can be found in [DL-PDE](https://arxiv.org/ftp/arxiv/papers/1908/1908.04463.pdf).

![image](https://user-images.githubusercontent.com/41933063/122041299-5fabf880-ce0b-11eb-8045-611bf98070ff.png)

Fig. 1. Flow chart of the symbolic genetic algorithm (SGA-PDE). The yellow boxes are the function terms in the form of trees, which constitute the candidate set. The green boxes represent the genetic algorithm specially designed to optimize the binary trees. The red boxes represent the performance assessment of the discovered PDE. 

## Loading dataset

 ```python
u = Data_generator.u
x = Data_generator.x
t = Data_generator.t
x_all = Data_generator.x_all
```

## Hyperparameters
configure.py:

    problem = 'chaffee-infante' # choose the dataset
    
    seed = 0 # set the random seed
    
    fine_ratio = 2 # the ratio of Metadata set to original dataset. A ratio of 2 means that the sampling interval of Metadata is twice that of original data.
    
    use_metadata = False # whether to use Metadata
    
    delete_edges = False # whether to delete the Metadata on the boundaries of the field where the derivatives are not accurate based on finite difference.
    
    aic_ratio = 1  # the hyperparameter in the AIC. lower this ratio, less important is the number of elements to AIC value.

    max_epoch = 100 * 1000 # Hyperparameter to generate the Metadata Neural Networks.
    
    hidden_dim = 50 # Hyperparameter to generate the Metadata Neural Networks. Number of neurons in the hidden layer.

    normal = True # Hyperparameter to generate the Metadata Neural Networks. Whether to normalize the inputs.

    
    
setup.py

    simple_mode = True  # use the simple operators and operands.
    
    use_metadata = False  # whether to use the Metadata.
    
    use_difference = True # whether to use finite difference or autograd to calculate the gradients.
    
 
sga.py

    num = 20 # the population of each generation is 20.
    
    depth = 4 # the maximum tree depth hyperparameter to constrain the number of nesting calculations in each function term.
    
    width = 5 # the maximum forest width hyperparameter to constrain the number of function terms in the PDE.
    
    p_var = 0.5 # The probability of generating a node as an operand (leaf node) instead of an operator is 0.5.
    
    p_mute = 0.3 # The probability of mutation at each node is 0.3.
    
    p_cro = 0.5 # The probability of crossover between different function terms in two PDE is 0.5.
    
    p_rep = 1 # The probability of regenerating a term to replace the original term in the PDE.
    
    sga_run = 100 # the maximum generation is set to 100.
    
## Requirements

The program is written in Python, and uses pytorch, scipy. A GPU is necessary, the ENN algorithm can only be running when CUDA is available.

- Install PyTorch(pytorch.org)
- Install CUDA
- `pip install -r requirements.txt`



## Author contributions
Y. Chen and D. Zhang conceptualized and designed the study. Y. Chen, Y. Luo, and Q. Liu developed the SGA-PDE algorithms. H. Xu developed the simulation codes for the example problems considered. 
