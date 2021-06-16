# SGA-PDE
Symbolic genetic algorithm for discovering open-form partial differential equations (SGA-PDE) is a model to discover open-form PDEs directly from data without prior knowledge about the equation structure. 
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

# Appendix
## Appendix A: Symbolic mathematical representation of open-form PDEs
A large number of governing equations in practice are in the form of PDEs related to time and space. This study explores automatic mining algorithms for open-form PDEs, where the key issue is how to represent the diverse open-form PDEs in a flexible manner. The conventional method to solve the PDE discovery problem is to determine in advance all of the candidates that may appear in the PDE, such as the derivatives of different orders, and then find the solution through sparse regression. This method can only be optimized in a closed candidate set, and cannot generate function terms outside of the fixed candidate set. Therefore, it is difficult to generate function terms with composite functions or fractional structures, as shown on the left side of Figure 2a. 

In order to avoid the restriction of the closed candidate set, SGA-PDE represents open-form PDEs by refining the basic components of the equations, i.e., transforming the representation units of the equations from the function term level to the operator and operand level. Specifically, this study defines operations involving two objects as double operators (e.g., addition, subtraction, multiplication, and division). In addition, the operations involving a single object, such as exponents, logarithms, and trigonometric functions, are defined as single operators. Operands are the independent variables and dependent variables in the equation, such as x, y, t, and u. In order to encode different PDEs, this study uses binary trees to combine operators and operands, where all leaf nodes in the tree are operands, and all internal nodes correspond to the operators. The double operators are the nodes with degree two (i.e., each node has two children), and the single operators are the nodes with degree one. 
In theory, any PDE can be transformed into a binary tree by the above method. For example, the aforementioned complex PDE with compound function and fractional structure can be written into the symbolic mathematical representation on the right side of Figure 4a. Therefore, by matching the nodes in the binary tree with the operands and operators in the PDE, any partial differential governing equation can be represented as a forest of binary trees.

![image](https://user-images.githubusercontent.com/41933063/122180394-f84d8180-ceba-11eb-93b3-bfd477f0901d.png)

Fig. 2. Schematic diagram of binary tree representation and transformation method based on preorder traversal. 

When performing symbolic mathematical operations, a binary tree needs to be converted into a computable string, and finally displayed as a function term that conforms to the mathematical expression habits. This study uses preorder traversal to achieve this goal. The preorder traversal starts from the root node of the binary tree. In the calculation, the left child node is processed first, and the current branch is traversed to the leaf node layer by layer. We then return to the nearest parent node whose degree is not full, enter the other branch (led by the right child node), and then traverse to the leaf node of that branch. The above process is repeated until all internal nodes of the entire binary tree are traversed. Figure 2b shows a binary tree corresponding to the function term ∂/∂x(∂u/∂x+u·u), and the red lines show the path of the preorder traversal. The binary tree can be converted into a computable string according to the traversal path (Figure 2c), where each parent node and its child nodes are combined with parentheses to form a calculation unit. The parent node must be an internal node (operator) that represents a certain operation, and its child nodes (and the following branches) are the objects of the operation in the calculation unit. The different calculation units are marked with different colored underlines in Figure 2c. Finally, in the lower part of Figure 2c, the order of each parent node and child node in the computable string is adjusted according to mathematical expression habits. 

Table 1 shows three examples of explainable binary trees and forests, their corresponding computable strings, and function terms. Since the trees contained in a forest are independent of each other (i.e., there is no weighted summation between trees in the representation step), the trees are connected with “&” rather than “+”.

Table 1. Comparison of explainable binary trees/forests, computable strings, and function terms.

![image](https://user-images.githubusercontent.com/41933063/122180593-24690280-cebb-11eb-811e-da6e666ff2a6.png)



## Appendix B: Random generation of open-form PDEs
In order to find the governing equation that best matches the data, SGA-PDE not only needs to represent any function term as a binary tree, but also needs to be able to randomly generate PDEs that comply with mathematical rules. In other words, the generated forest needs to meet the following five rules: (1) the leaf nodes of each tree are operands, and internal nodes must be operators; (2) the number of child nodes of internal nodes must be the same as the degree (i.e., the degree of all internal nodes is full); (3) the root node of each tree does not contain addition and subtraction; (4) the depth of any branch shall not exceed the given maximum depth (to avoid generating excessively complex nesting forms); and (5) the number of randomly generated trees in the forest is less than the given maximum width (to avoid generating too many redundant terms). The first three rules ensure that the generated tree is mathematically reasonable, while the latter two rules embody Occam's Razor (i.e., the simpler the better) to produce a concise equation structure.

 ![image](https://user-images.githubusercontent.com/41933063/122180708-419dd100-cebb-11eb-9397-14ae00a38be4.png)

Fig. 3. Flow chart of randomly generating a forest (PDE). The mutations in the genetic algorithm occur in the steps of generating operators and operands, and the crossover happens in the process of adding new trees.

Figure 3 shows the process of randomly generating a forest that meets the rules in SGA-PDE. First, a root node is generated for each tree in the forest, and then the child nodes of each branch of each tree are generated in turn, until a leaf node is randomly generated or the maximum depth is satisfied. Whenever the degree of an internal node is full, it enters the other branch of its parent node until the degree of the root node is also satisfied. When the degrees of all nodes of a tree are full, the current tree is complete and the next tree is ready to be generated. When the maximum number of trees (i.e., the width of PDE) is met, the random generation process ends, and a forest (PDE) with multiple trees (function terms) is obtained.

## Appendix C: Genetic algorithm for binary trees
In SGA-PDE, we use symbolic mathematics to transform any PDE into a forest of binary trees. Since all of the binary trees randomly produced by the SGA-PDE meet the mathematical rules of PDE, and any function term in a PDE can be represented as a binary tree, SGA-PDE constructs a bijection between the function term space and the binary tree space (i.e., each element of one set is paired with exactly one element of the other set, and there are no unpaired elements). This means that the open-form PDE representation method based on symbolic mathematics is effective and non-redundant, and can be used as the genetic representation of the solution domain. On this basis, this paper proposes a genetic algorithm specially designed for the tree structure to iteratively optimize candidate equations, so as to automatically mine the PDE that conforms to the observations. 

Specifically, SGA-PDE not only includes the crossover and mutation in the conventional genetic algorithm, but also introduces the replacement operation. SGA-PDE will first measure the fitness of each candidate solution in the current generation through AIC, which is taken as the performance of the current solution, and then select the best performing candidate solutions to perform crossover, mutation, and replacement operations in turn. 
In the crossover, the trees (function terms) in the forest (PDE) will be recombined with each other to stochastically generate new candidates from the existing population, as shown in Figure 4. Each column in the figure represents a forest candidate (PDE), and each rectangle in the column represents a tree (function term). Assuming that there are 2n forest candidates in the previous generation, in the crossover of the current generation, the forest candidates are first evaluated and sorted based on AIC, and the best n candidates among them are recombined to generate n crossover candidates. Then, the new generated candidates are deduplicated (i.e., the candidates that have appeared in previous generations are deleted), and m new candidates are obtained. Finally, according to the fitness of the data, the 2n candidates from the previous generation and the m new candidates generated by the crossover are sorted together, and the 2n candidates with the best performance are taken as the result of the current generation and passed to the next step.

In the mutation, new function terms can be generated through the change of operands and operators while the topology of the existing binary tree remains unchanged. In order to ensure that the new binary tree also complies with mathematical rules of PDE, each node can only be mutated into a node with the same attribute during mutation. In other words, operands, single operators, and double operators can only be mutated into new operands, single operator, and double operator, respectively (i.e., the degree of nodes before and after the mutation is unchanged). Figure 6 shows the mutation process of operands and operators. It can be seen that through node mutation, new function terms can be generated under the condition that the binary tree topology is completely unchanged. This also reflects the flexibility of using a binary tree to represent PDE in SGA-PDE.

 ![image](https://user-images.githubusercontent.com/41933063/122181058-95101f00-cebb-11eb-8615-a26a842e1cfd.png)

Fig. 4. Schematic diagram of the genetic algorithm for the tree structure in SGA-PDE. The crossover uses optimal parents to generate new PDEs. The mutation includes random change of operators and operands. The replacement includes random change of tree topology.

In the replacement, a new function term is obtained by directly regenerating a tree. The mathematical meaning of a binary tree is determined by the node and the tree topological structure. The mutation generates new function terms by modifying nodes, while replacement changes the node and tree topological structure at the same time. Replacement is a more radical exploration of the solution domain and can expand the topological structure of the binary tree in the candidate set.
 
SGA-PDE uses the topological structure of the binary tree to represent the complex function term structure in PDEs, and thus the optimization of the topological structure is an important task of the genetic algorithm. Since the mutation in SGA-PDE is to find a better node combination for a given topological structure, it is an exploration of the topological structure from the perspective of optimization. The replacement introduces new topological structures, which is essentially an exploitation of the topological structure. Therefore, through the combination of mutation and replacement, SGA-PDE can take into account the exploration and exploitation of the binary tree topology, which is conducive to efficiently finding the optimal symbolic mathematical representation of the PDE at the function term (binary tree) level.
 
The genetic algorithm adopted by SGA-PDE has been specially designed according to the tree structure and characteristics of PDE. Crossover is the reorganization of different binary trees in the forest, which can generate new solution candidates at the PDE level without changing the function terms (binary trees); whereas, mutation and replacement are internal changes to a certain function term, introducing new solution candidates at the function term (binary tree) level. The combination of crossover, mutation, and replacement realizes genetic variation at different levels in SGA-PDE and balances exploration and exploitation, which is important for finding the most suitable governing equation from the data.
