## Study case: Burgers’ equation
Burgers’ equation is a fundamental partial differential equation derived by Bateman in 1915(2, 3) and later studied by Burgers in 1948(4). It can simulate the propagation and reflection of shock waves, which is common in a variety of dynamical problems, such as nonlinear acoustics(5) and gas dynamics(6). The general form of Burgers’ equation takes the following form: 

	 
![image](https://user-images.githubusercontent.com/41933063/122323321-824c2780-cf59-11eb-9379-a955073d032a.png)
(S1)

where u(x,t) is a given field with spatial (x) and temporal (t) dimension; and ν represents the diffusion coefficient or kinematic viscosity according to the problem, and it is a constant physical property.
	Burgers’ equation involves nonlinear terms, and it also includes an interaction term of u and ∂u/∂x. Therefore, this equation is used to validate the approach’s effectiveness of discovering nonlinear terms from observations(1, 7). 
The conventional spectral method is utilized to generate the dataset, where the diffusion coefficient ν is set as one(1, 8). In the dataset, there are 201 temporal observation steps at intervals of 0.05, and 256 spatial observation steps at intervals of 0.0625. Therefore, the total number of data points is 51,456. 

![image](https://user-images.githubusercontent.com/41933063/122323338-8d06bc80-cf59-11eb-83f1-51f35567b5d3.png)

Fig. S1. Data distribution of the observations used to mine the Burgers’ equation. (A) Observation u. (B) Derivative u_t of the observation u with respect to time at all positions at the intermediate moment. (C) Derivative u_x of the observation u with respect to space at all positions at the intermediate moment.

Figure S1a shows the distribution of the observations u in Burgers’ equation. The bottom of the figure is the projection of u on the x-t plane. It can be seen that Burgers’ equation is volatile in space due to the influence of derivatives. In addition, the amplitude of observations will decay as time progresses. Figure S1b and Figure S1c, respectively, show the first derivative of the observations with respect to time (u_t) and space (u_x) at the intermediate time step (t=5). 
It should be noted that the three variables in Figure S1 are all of the data used by SGA-PDE when mining Burgers’ equation. The data do not cover all of the function terms in Burgers’ equation. Therefore, this experiment can verify the ability of SGA-PDE to generate new function terms through genetic algorithms designed for the tree structure.

## Study case: Korteweg-de Vries (KdV) equation
The Korteweg-de Vries (KdV) equation is another classical example to verify the performance of the PDE discovery algorithm(1, 7). It models the propagation of waves on shallow water surfaces. It is known as not only a prototypical example of exactly solvable models, but also one of the earliest models that have soliton solutions. The KdV equation was first discovered by Boussinesq in 1877(9), and later developed by Korteweg and de Vries in 1895 when investigating small-amplitude and long-wave motion in shallow water(10). The KdV equation takes the following form:

![image](https://user-images.githubusercontent.com/41933063/122323727-20d88880-cf5a-11eb-97d2-2d730c4eb758.png)
(S2)

where a and b are constants, which are set as a=-1 and b=-0.0025, respectively. 
Since the KdV equation contains a third-order derivative term, its volatility is greater and it is challenging to learn the correct equation structure directly from the data. In order to increase the difficulty of the experiment, we only provide the original observation u (Figure S2a) and the first derivative of u with respect to time and space (Figure S2b and Figure S2c) for the SGA-PDE. The ability of SGA-PDE to find higher-order derivatives that do not exist in the candidate set based on limited data can be verified through this experiment. 
The conventional spectral method is utilized to generate the dataset(1). Regarding the data size, there are 201 temporal observation steps at intervals of 0.005, and 512 spatial observation steps at intervals of 0.0039. Therefore, the total number of data points is 102,912.

![image](https://user-images.githubusercontent.com/41933063/122323752-29c95a00-cf5a-11eb-9fd8-abb6db85ef25.png)

Fig. S2. Data distribution of the observations used to mine the KdV equation. (A) Observation u. (B) u_t at all positions at the intermediate moment. (C) u_x at all positions at the intermediate moment.

## Study case: Chafee-Infante equation
The Chafee-Infante equation is a reaction diffusion equation proposed by Chafee and Infante in 1974(11), which describes the physical processes of material transport and particle diffusion, and it is applied in fluid mechanics, high-energy physical processes, electronics, and environmental science(12, 13). It is a kind of nonlinear evolution equation in the form of Eq. S3:

	 
![image](https://user-images.githubusercontent.com/41933063/122698393-d583ea00-d279-11eb-91d0-4dd71f27d221.png)
(S3)

where a is the diffusion coefficient, which is set as a=1 in this study. 
The Chafee-Infante equation is also called the Newell-Whitehead equation when the diffusion coefficient a is 1(14, 15). The Chafee-Infante equation has a second-order derivative and a third-order exponential term, with strong nonlinearity. The dataset is generated via the forward difference method(1), and there are 200 temporal observation steps at intervals of 0.002, and 301 spatial observation steps at intervals of 0.01. Therefore, the total number of data points is 60,200. All of the data used by SGA-PDE to discover the Chafee-Infante equation is shown in Figure S3.

![image](https://user-images.githubusercontent.com/41933063/122698418-dc126180-d279-11eb-8336-5081e223a490.png)
 
Fig. S3. Data distribution of the observations used to mine the Chafee-Infante equation. (A) Observation u. (B) u_t at all positions at the intermediate moment. (C) u_x at all positions at the intermediate moment.

## Study case: Open-form PDE with fractional structure and compound function
In order to verify the ability of SGA-PDE to mine complex open-form PDEs, we generated two testing datasets based on two PDEs. The first PDE has a fractional structure (PDE_divide), as shown in Eq. S4. The second is PDE_compound, which contains a compound function (Eq. S5). 

	 
![image](https://user-images.githubusercontent.com/41933063/122698467-f1878b80-d279-11eb-9698-0127481e1357.png)
(S4)


![image](https://user-images.githubusercontent.com/41933063/122698486-fb10f380-d279-11eb-9d3b-8ecb8e1221a7.png)
(S5)

Specifically, to generate datasets of PDE_divide and PDE_compound, the problems are solved numerically using the finite difference method, where x∈[1,2]. The initial condition is u(x,0)=-sin⁡(πx), and the boundary condition is u(1,t)=u(2,t)=0,t>0. Both PDEs are solved with 100 spatial observation points for 250,001 timesteps. Then, temporal observation points are taken every 1,000 timesteps. Therefore, we have 100 spatial observation points and 251 temporal observation points, and the total number of the dataset is 25,100.

![image](https://user-images.githubusercontent.com/41933063/122698519-0e23c380-d27a-11eb-82fd-c85cae2f0021.png)
 
Fig. S4. Data distribution of the observations used to mine the PDE with fractional structure. (A) Observation u. (B) u_t at all positions at the intermediate moment. (C) u_x at all positions at the intermediate moment.

![image](https://user-images.githubusercontent.com/41933063/122698544-1aa81c00-d27a-11eb-8565-227298ff4d90.png)
 
Fig. S5. Data distribution of the observations used to mine the PDE with compound function. (A) Observation u. (B) u_t at all positions at the intermediate moment. (C) u_x at all positions at the intermediate moment.

Since it is almost impossible to determine all potential function terms with fractional structure or compound functions in the candidate set in advance, PDE discovery algorithms based on sparse regression (e.g., PDE-FIND and DL-PDE) cannot handle the open-form PDEs, such as Eq. S4 and Eq. S5. Although DLGA based on the genetic algorithm can extend the candidate set, it still lacks a method to generate compound functions or fractional structures through a simple combination of gene fragments. Therefore, the above two open-form PDEs are challenging for conventional methods.
Finally, the generated dataset of PDE_divide is shown in Figure S4, and the data distribution of PDE_compound is shown in Figure S5. Although the changing trends of the observation values of the two PDEs appear similar, their governing equations are completely dissimilar, which also reflects the difficulty of directly mining the control equations through data. These two equations can verify the SGA-PDE’s ability to mine open-form PDEs through its flexible representation, which is essential for the future use of SGA-PDE to extract unknown and undiscovered governing equations from data.

## References
1.	H. Xu, H. Chang, D. Zhang, DLGA-PDE: Discovery of PDEs with incomplete candidate library via combination of deep learning and genetic algorithm. J. Comput. Phys. 418, 109584 (2020).
2.	H. Bateman, Some recent researches on the motion of fluids. Mon. Weather Rev. 43, 163-170 (1915).
3.	G. B. Whitham, Linear and nonlinear waves.  (John Wiley & Sons, 2011), vol. 42.
4.	J. M. Burgers, in Adv. Appl. Mech. (Elsevier, 1948), vol. 1, pp. 171-199.
5.	O. Rudenko, S. Soluian, The theoretical principles of nonlinear acoustics. Moscow Izdatel Nauka,  (1975).
6.	S. Kutluay, A. Bahadir, A. Özdeş, Numerical solution of one-dimensional Burgers equation: explicit and exact-explicit finite difference methods. J. Comput. Appl. Math. 103, 251-261 (1999).
7.	S. H. Rudy, S. L. Brunton, J. L. Proctor, J. N. Kutz, Data-driven discovery of partial differential equations. Sci. Adv. 3, e1602614 (2017).
8.	M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J. Comput. Phys. 378, 686-707 (2019).
9.	J. Boussinesq, Essai sur la théorie des eaux courantes.  (Impr. nationale, 1877).
10.	D. J. Korteweg, G. De Vries, On the change of form of long waves advancing in a rectangular canal, and on a new type of long stationary waves. The London, Edinburgh, Dublin Philosophical Magazine Journal of Science 39, 422-443 (1895).
11.	N. Chafee, E. F. Infante, A bifurcation problem for a nonlinear partial differential equation of parabolic type. Appl. Anal. 4, 17-37 (1974).
12.	M. Tahir et al., Exact traveling wave solutions of Chaffee–Infante equation in (2+ 1)‐dimensions and dimensionless Zakharov equation. Math. Methods Appl. Sci. 44, 1500-1513 (2021).
13.	Y.-H. Sun, S.-H. Yang, J. Wang, F.-S. Liu, New exact solutions of nonlinear Chafee-Infante reaction and diffusion equation. Journal of Sichuan Normal University 3,  (2012).
14.	A. C. Newell, J. A. Whitehead, Finite bandwidth, finite amplitude convection. J. Fluid Mech. 38, 279-303 (1969).
15.	A. Korkmaz, Complex wave solutions to mathematical biology models I: Newell–Whitehead–Segel and Zeldovich equations. J. Comput. Nonlinear Dyn. 13,  (2018).


