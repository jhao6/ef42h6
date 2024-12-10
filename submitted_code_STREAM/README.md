This is code for the submission "Forgetting Measure Estimation Provably Helps: A New Memory-based Continual Learning Algorithm"
## Abstract

Continual Learning (CL) is a learning paradigm that aims to train a model across a sequence of tasks while retaining previous knowledge and adapting to new environments. To mitigate catastrophic forgetting, several memory-based CL methods have been developed. These methods store relevant information from previous tasks and use it to guide learning new tasks, typically by updating the model based on both current and memory gradient information. In this paper, we demonstrate that some of these approaches, which rely solely on gradient information, may fail to provide learning and forgetting guarantees, as illustrated by a simple counterexample. Motivated by this insight, we propose STREAM, a new algorithm that employs \emph{a momentum estimator based on the loss functions} to measure forgetting and balance the learning of current and memory tasks. STREAM directly addresses the constrained optimization problem of minimizing the loss of current tasks while maintaining minimal forgetting of the previous tasks. In particular, we prove that STREAM converges to nearly $\epsilon$-stationary points with $\widetilde{O}(\epsilon^{-8})$ complexity of querying stochastic gradient, matching the state-of-the-art complexity result of single-loop algorithms under this problem setting, without requiring a large batch size. We conduct extensive experiments in various CL settings, including single/multimodal classification and task/class incremental CL, to show that our proposed algorithm significantly outperforms strong baselines in CL. 

##Requirements
PyTorch >= v1.6.0. The code is based on Improved Schemes for Episodic Memory-based Lifelong Learning and
Gradient Episodic Memory for Continual Learning 

### Training by STREAM on Split CIFAR-100
    python main.py --model stream --dataset CIFAR100 

### Training by STREAM on Multiple Dataset
    python main.py --model stream --dataset mixture 

### Training by STREAM on Split Tiny-imagenet
    python main.py --model stream --dataset Tiny-imagenet 

