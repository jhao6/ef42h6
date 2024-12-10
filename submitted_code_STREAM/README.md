This is code for the submission "Forgetting Measure Estimation Provably Helps: A New Memory-based Continual Learning Algorithm"
## Abstract

Continual Learning (CL) is a learning paradigm that aims to train a model across a sequence of tasks, enabling the model to retain previous knowledge while adapting to new environments. To address the issue of catastrophic forgetting, several memory-based CL methods have been developed. These methods store relevant information from previous tasks and use it to guide the learning of new tasks. Specifically, they update the model using a direction derived from both the current gradient and the memory gradient information at each iteration. In this paper, we demonstrate that some of these approaches, which rely solely on gradient information for the learning process, fail to provide learning and forgetting guarantees, as illustrated by a simple counterexample. Motivated by this insight, we propose a new algorithm called STREAM. STREAM employs \emph{a momentum estimator based on the loss functions to measure forgetting} and uses this measure to balance the learning of current and memory data. STREAM directly addresses the constrained optimization problem of minimizing the loss of current tasks while maintaining minimal forgetting of previous tasks, with a provable guarantee. This local guarantee implies end-to-end learning and forgetting guarantees throughout the entire learning process. The algorithm updates the model by selecting either the current gradient or the memory gradient, based on a moving-average estimator for the forgetting measure. Under standard assumptions of the objective function, we prove that the STREAM algorithm can converge to a nearly $\epsilon$-stationary point of the constrained optimization problem within the polynomial number of stochastic gradient evaluations, providing provable learning and forgetting guarantees. We conducted extensive experiments in various CL settings, including single/multimodal classification and task/class incremental CL, to show that our proposed algorithm significantly outperforms strong baselines in CL.

##Requirements
PyTorch >= v1.6.0. The code is based on Improved Schemes for Episodic Memory-based Lifelong Learning and
Gradient Episodic Memory for Continual Learning 

### Training by STREAM on Split CIFAR-100
    python main.py --model stream --dataset CIFAR100 

### Training by STREAM on Multiple Dataset
    python main.py --model stream --dataset mixture 

### Training by STREAM on Split Tiny-imagenet
    python main.py --model stream --dataset Tiny-imagenet 

