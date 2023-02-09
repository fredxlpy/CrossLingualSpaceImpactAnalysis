import pandas as pd
from tqdm import tqdm
from src.CKA import cka, gram_linear
import numpy as np

"""
This script computes the similarity of a target language representation space
in each layer before and after fine-tuning on a source language.

Example:
    Source language: French
    Target Language: Hindi
    The CKA value will quantify the similarity of the Hindi representation space
    before and after fine-tuning on French in a specific layer. In other words,
    we can see how much the Hindi representation space has been impacted by fine-tuning on French.
"""

source_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
target_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

results = []
for i,target_lang in enumerate(target_langs):
    print(f"Starting {target_lang} ({i}/{len(target_langs)})")

    BASE_embeddings = np.load(f'./Output/Embeddings/base_{target_lang}.npy')

    for source_lang in tqdm(source_langs):

        FT_embeddings = np.load(f'./Output/Embeddings/{source_lang}_to_{target_lang}.npy')

        for layer in range(1,13):

            X = BASE_embeddings[:,layer,:]
            Y = FT_embeddings[:,layer,:]
            similarity = cka(gram_linear(X), gram_linear(Y))

            results.append([source_lang, target_lang, layer, similarity])

results = pd.DataFrame(results, columns=['Source', 'Target', 'Layer', 'CKA'])
results.to_excel('./Output/CKA.xlsx', index=False)