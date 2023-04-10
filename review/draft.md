# Intro
Hello, everyone! We are group 15. The paper we are going to present today is CRET: Cross-Modal Retrieval Transformer for Efficient Text-Video Retrieval, written by researchers from ant group company.

# Contents
This is a brief content of our presentation. First, we will briefly explain our motivation for choosing this paper. Then, we will explain the CRET method in detail and present experimental results in the original paper. This presentation is concluded with our critical review.

# Motivation
Information can be stored and delivered through various carriers. In machine learning area, a modality refers to the data collected via the same information channel. During the class, we mainly focus on unimodal information retrieval task. The input and output of classical models we've seen so far are both text. However, we can't directly use these classical models to handle multimodal information retrieval task. For example, text-to-video retrieval. We need a model that can process information from different modality. Researchers from ant group company proposed a novel method called CRET that achieves the state-of-art performance on text-to-video retrival task. Next we will introduce CRET method in detail.

# Experiments
Authors conducted experiments on four benchmark datasets to verify the effectiveness of CRET method. They select recall at rank K and MdR as metrics. Higher recall at rank K means better performance. Note that MdR measures the median rank of correct items in the retrieved ranking list. Therefore, lower MdR indicates a better model.

The table on the slide shows experimental results on MSRVTT dataset.

This slide presents results on LSMDC and MSVD datasets.

And this table is the result on DiDeMo. We can see that CRET method outperforms other baselines on all datasets. Authors of the original paper also conducted abalation studies and validated the Gaussian assumption.

# Critical review
Next, I will talk about our critical review. We think this paper clearly illustrates the principle of CRET method. And its structure is consistent with requirements of the scientific paper. We examine the reproducibility of paper from three aspects: source code, the avaliablity of data sets, and experimental settings. Datasets can be downloaded from links mentioned in references. And authos reported main parameter setting in paper. However, it is unfortunate that authors don't make their code publicly available. The main theorectical contribution of this paper is the design of CRET model. From the perspective of application, we think the CRET method has broad prospect in video-streaming websites and search enigines. Finally, I will summarize strong and weak points of this paper. One advantage is that it fully consider the algorithm effciency both in model design and implementation. Another strong point is that authors use extensive ablation studies and hypothesis tests to support the validity of model. However, the inaccessibility of code has a great impact on reproducibility. Besides, authors don't report the standard variance of experimental results and don't clearly state the number of times for running each model.

That's all of our presentation. Thank you!
