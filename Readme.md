# Code for [*Identifying the Correlation Between Language Distance and Cross-Lingual Transfer in a Multilingual Representation Space*](https://aclanthology.org/2023.sigtyp-1.3/) (Philippy et al., 2023)

* `CKA.py` contains the function to compute the layer similarity using the Centered Kernel Alignment method [Kornblith et al., 2019] 
* `01 - Trainer.py` fine-tunes mBERT on XNLI in different source languages
* `02 - extract_embeddings.py` allows to compute embeddings at different layers from the base and fine-tuned models using the XNLI test set
* `03 - layer_similarity.py` computes the similarity between the layers from the base and the fine-tuned model using CKA and the computed embeddings
* `04 - language_similarity.py` extracts 5 different language distance metrics using the pre-computed lang2vec distances
* `05 - zero_shot_eval.py` evaluates the fine-tuned models in different target languages in a zero-shot setting
* `06 - correlation_study.py` computes the correlations between
    * language distance metrics and impact on representation space
    * impact on representation space and transfer performance
    * transfer performance and language distance metrics


Kornblith, S., Norouzi, M., Lee, H. & Hinton, G. (2019). [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414). Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:3519-3529.

## Citation
```
@inproceedings{philippy-etal-2023-identifying,
    title = "Identifying the Correlation Between Language Distance and Cross-Lingual Transfer in a Multilingual Representation Space",
    author = "Philippy, Fred  and
      Guo, Siwen  and
      Haddadan, Shohreh",
    editor = "Beinborn, Lisa  and
      Goswami, Koustava  and
      Murado{\u{g}}lu, Saliha  and
      Sorokin, Alexey  and
      Kumar, Ritesh  and
      Shcherbakov, Andreas  and
      Ponti, Edoardo M.  and
      Cotterell, Ryan  and
      Vylomova, Ekaterina",
    booktitle = "Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.sigtyp-1.3",
    doi = "10.18653/v1/2023.sigtyp-1.3",
    pages = "22--29",
}
```