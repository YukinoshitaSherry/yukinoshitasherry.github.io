---
title: Rebuttal技巧
date: 2025-07-29
categories: 
    - 学CS/SE
tags: 
    - 科研技巧
desc: 写于2025.7 NIPS 人生首次rebuttal,持续补完中。虽然人生首投稿NIPS估计是寄了，但经验可以复用，持续努力吧。
---

### 目标

### 步骤



#### 格式检查
openreview上会有字数限制，看各个会议的要求。
格式主要是markdown、可以插入latex公式,不能插入链接、图片。
所以流程图等等用箭头象形一下。
记得先点击preview检查一下再提交。
<br>

### 策略
问题一般分为三类。

#### 好问题，但没时间做，没办法解决 

#### 好问题，可以加实验解决 
加实验，把结果给reviewer

#### 坏问题，理解错误 
说服对方祂哪里错，列举reference作为证据

eg: scAgents > Weak baselines for some tasks: Comparisons for ATAC and CITE-seq tasks include only simple models like linear regression and random forest, rather than more recent approaches.

Thank you for your insightful comment. Indeed, our comparison for the ATAC and CITE-seq tasks relies on traditional machine learning methods. There are two main reasons for this choice, as our task focuses on predicting responses to unseen perturbations in scATAC-seq and multiomics CITE-seq settings.
First, most current perturbation prediction models are designed for scRNA-seq gene expression prediction11,12 and cannot be directly applied to scATAC-seq or CITE-seq due to substantial modality differences. For example, GERAS1 is tailored for gene node prediction and cannot handle peak features in ATAC-seq or protein features in CITE-seq. scGen2 relies on variational autoencoders with normal or negative binomial assumptions that are mismatched with the distributions observed in ATAC peaks or CITE proteins. scGPT3 is not compatible either, as it cannot process peak or protein tokens absent from its pretrained vocabulary. Similarly, CellOT4 struggles with the high dimensionality of ATAC-seq and does not support prediction under unseen perturbations.
Second, the limited number of existing multimodal perturbation prediction models are not suitable for our scenario of unseen peak prediction. For example, the recent preprint EpiAgent5 proposes an approach similar to scGPT, using cis-regulatory elements (CREs) as tokens to predict perturbation outcomes for scATAC-seq. However, it focuses on CRE token prediction rather than peak-level prediction as in our model. GET 13 also claims to support in silico perturbation prediction, but it is restricted to bulk ATAC-seq knockout experiments and cannot be extended to drug or cytokine perturbations.
For CITE-seq, OT-based method6 predicts one modality from another (e.g., using scRNA-seq to predict CITE-seq), emphasizing cross-modality translation rather than concurrent prediction, which differs from our objective. Additionally, most multimodal generative methods focus on integration rather than perturbation generalization. For instance, totalVI7, MultiVI8, and GLUE15 are developed for data harmonization under shared conditions and cannot generate data under novel perturbations. Biolord14 supports counterfactual predictions in single-modality ATAC or CITE-seq data, but lacks a covariate encoder necessary for generalizing to unseen perturbations.
Lastly, we would like to emphasize that linear regression and random forest are not necessarily naive baselines. Recent studies demonstrate that these classical models can outperform many newly proposed deep learning approaches in scRNA-seq tasks9,10. As such, our use of these models provides meaningful benchmarks and a solid foundation for evaluating the performance of our agent-based methods.


Lotfollahi, M., Wolf, F.A. & Theis, F.J. scGen predicts single-cell perturbation responses. Nat Methods 16, 715–721 (2019).
Cui, H. et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nat Methods 21, 1470–1480 (2024).
Bunne, C. et al. Learning single-cell perturbation responses using neural optimal transport. Nat Methods 20, 1759–1768 (2023).
Chen, X. et al. EpiAgent: Foundation model for single-cell epigenomic data. bioRxiv (2024).
Ryu, J. et al. Cross-modality matching and prediction of perturbation responses with labeled Gromov-Wasserstein optimal transport. arXiv preprint arXiv:2405.00838 (2024).
Gayoso, A. et al. Joint probabilistic modeling of single-cell multi-omic data with totalVI. Nat Methods 18, 272–282 (2021).
Ashuach, T. et al. MultiVI: deep generative model for the integration of multimodal data. Nat Methods 20, 1222–1231 (2023).
Ahlmann-Eltze, C., Huber, W. & Anders, S. Deep learning-based predictions of gene perturbation effects do not yet outperform simple linear baselines. bioRxiv (2024).
Adduri, A. et al. Predicting cellular responses to perturbation across diverse contexts with STATE. bioRxiv (2025).
Li, L. et al. A systematic comparison of single-cell perturbation response prediction models. bioRxiv (2024).
Li, C. et al. Benchmarking AI models for in silico gene perturbation of cells. bioRxiv (2024).
Fu, X. et al. A foundation model of transcription across human cell types. Nature 637, 965–973 (2025).
Piran, Z. et al. Disentanglement of single-cell data with biolord. Nat Biotechnol 42, 1678–1683 (2024).
Cao, Z.-J. & Gao, G. Multi-omics single-cell data integration and regulatory inference with graph-linked embedding. Nat Biotechnol 40, 1458–1466 (2022).

<br>

### 语料




