# Code from "Exploring the Correlation Between Linguistic Features and Cross-Lingual Transfer in a Multilingual Representation Space"
* `CKA.py` contains the function to compute the layer similarity using the Centered Kernel Alignment method<sup>*</sup> 
* `01 - Trainer.py` fine-tunes mBERT on XNLI in different source languages
* `02 - extract_embeddings.py` allows to compute embeddings at different layers from the base and fine-tuned models using the XNLI test set
* `03 - layer_similarity.py` computes the similarity between the layers from the base and the fine-tuned model using CKA and the computed embeddings
* `04 - language_similarity.py` extracts 5 different language distance metrics using the pre-computed lang2vec distances
* `05 - zero_shot_eval.py` evaluates the fine-tuned models in different target languages in a zero-shot setting
* `06 - correlation_study.py` computes the correlations between
    * language distance metrics and impact on representation space
    * impact on representation space and transfer performance
    * transfer performance and language distance metrics


<sup>*</sup> Kornblith, S., Norouzi, M., Lee, H. & Hinton, G. (2019). [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414). Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:3519-3529.