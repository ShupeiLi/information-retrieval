# Intro
Hello, everyone! We are group 15. The paper we are going to present today is CRET: Cross-Modal Retrieval Transformer for Efficient Text-Video Retrieval, written by researchers from ant group company.

# Contents
This is a brief content of our presentation. First, we will briefly explain our motivation for choosing this paper. Then, we will explain the CRET method in detail and present experimental results in the original paper. This presentation is concluded with our critical review.

# Motivation
Information can be stored and delivered through various carriers. In machine learning area, a modality refers to the data collected via the same information channel. During the class, we mainly focus on unimodal information retrieval task. The input and output of classical models we've seen so far are both text. However, we can't directly use these classical models to handle multimodal information retrieval task. For example, text-to-video retrieval. We need a model that can process information from different modality. Researchers from ant group company proposed a novel method called CRET that achieves the state-of-art performance on text-to-video retrival task. Next, my teammate will introduce CRET method in detail.

# Methodology
## Slides 1
There are two popular methods in the text-video eld. One is the embedding-based method. It utilizes the embeddings of features extracted from the text and video separately but suffers from the low-accuracy because of the loss of correspondence details.

The other is the model-based method. It iterates all the text-video pairs to evaluate the distance without extracting explicit embeddings. So it achieves better accuracy but suffers from low efficiency.

The authors of this paper proposed a novel EDB method named CRET, which solves the conflict between efficiency and accuracy.

## Slide 2
The overview of the CRET model is shown in the slide. We can see from the picture that the proposed CRET model encodes the video and text separately. 

(鼠标指)\
The text encoder applies the BERT as the base model to encode features into global and local features. The video encoder consists of spatial and temporal encoders(encode sampled multiple frames and frame-level encoding). The spatial encoder encodes all the features into global and local features. The CLS represents the global features. We can see that we feed the global features into the temporal encoder to get the temporal embeddings. On the other side, the local features are projected to the same dimension as the text embeddings.

On the top left of this picture, we estimate the parameters for the distribution of the features extracted from the video frames using the global temporal features. Then we calculate the GEES loss according to the estimated parameters and the global text features.

As for the right part of this picture, we can see that we align text and video features using the CCM module to deal with the loss of correspondence details.

Actually, the CCM module utilises the multi-head self-attention mechanism. We align these features using the transformer in which the text and video encoders share the same weights. As we can see from the equation, we rst calculate the distance between the token features and the query centre. Then we regard the distance as the weight of this feature. In this way, we put more importance on the features that are close to the query centre. Afterwards, we concatenate and project the aligned features from the multi-head module, and calculate the similarity score of these aligned features from text and video modalities.

## Slide 3
Next, we will discuss more details about the two important parts of this model—CCM module and GEES loss.

Let us move to the GEES loss. The traditional loss function NCE requires a trade-o between the accuracy and computational burden. The author improved this loss function by rst making an assumption about the frame distribution. In detail, we suppose that the frame-level features of each video follow the Multivariate Gaussian Distribution. Then simplify and approximate the NCE function which combines with the assumption. This improvement enhances the optimization efficiency of the SGD algorithm during the training process.

In a word, we improve the efficiency of the EDB method using the optimized GEES loss and enhance the accuracy by aligning the features from the text and video modalities using the CCM module.

# Experiments
Authors conducted experiments on four benchmark datasets to verify the effectiveness of CRET method. They select recall at rank K and MdR as metrics. Higher recall at rank K means better performance. Note that MdR measures the median rank of correct items in the retrieved ranking list. Therefore, lower MdR indicates a better model.

The table on the slide shows experimental results on MSRVTT dataset.

This slide presents results on LSMDC and MSVD datasets.

And this table is the result on DiDeMo. We can see that CRET method outperforms other baselines on all datasets. Authors of the original paper also conducted ablation studies and validated the Gaussian assumption.

# Critical review
Next, I will talk about our critical review. We think this paper clearly illustrates the principle of CRET method. And its structure is consistent with requirements of the scientific paper. We examine the reproducibility of paper from three aspects: source code, the avaliablity of data sets, and experimental settings. Datasets can be downloaded from links mentioned in references of the original paper. And authors also reported main parameter settings in paper. However, it is unfortunate that authors don't make their code publicly available. The main theorectical contribution of this paper is the design of CRET model. From the perspective of application, we think the CRET method has broad prospect in video-streaming websites and search engines. Finally, I will summarize strong and weak points of this paper. One advantage is that it fully consider the algorithm effciency both in model design and implementation. Another strong point is that authors use extensive ablation studies and hypothesis tests to support the validity of model. However, the inaccessibility of code has a great impact on reproducibility. Besides, authors don't report the standard variance of experimental results and don't clearly state the number of times for running each model. Therefore, it is hard to assess the stability of CRET based on tables provided in the paper.

That's all of our presentation. Thank you!
