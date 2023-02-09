import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Create target Directory if don't exist
dirName = './Output/Embeddings/'
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:
    print("Directory " , dirName ,  " already exists")

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize_function(examples):
    try:
        return tokenizer(examples["premise"], examples["hypothesis"],
                         padding="max_length", max_length=128, truncation='only_first')
    except:
        return tokenizer(examples["premise"], examples["hypothesis"],
                         padding="max_length", max_length=128, truncation=True)

# Choose source languages
source_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

# Choose target languages (i.e. languages which the representation space embeddings should be extracted from)
target_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

for source_lang in source_langs:

    # Load model fine-tuned in source language
    ft_model = AutoModelForSequenceClassification.from_pretrained(f'./Models/model_{source_lang}').to(device)
    ft_model.eval()

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(f'./Models/model_{source_lang}').to(device)
    base_model.eval()


    for target_lang in target_langs:

        # Load and tokenize test set
        dataset = load_dataset("xnli", target_lang, split='test')
        tokenized_set = dataset.map(tokenize_function, batched=True)
        tokenized_set.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

        dataloader = DataLoader(tokenized_set, batch_size=16)

        # Compute embeddings from fine-tuned models
        outputs = []
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = ft_model.bert(**batch, output_hidden_states=True)['hidden_states']
            out = torch.stack(out, dim=1).detach().cpu()[:,:,0,:]  # We only keep the [CLS] embedding
            outputs.append(out)
        output = torch.cat(outputs, dim=0)

        # Saving the embeddings as numpy arrays
        output = output.detach().numpy()
        np.save(f'./Output/Embeddings/{source_lang}_to_{target_lang}.npy', output)

        # Compute base model embeddings
        outputs = []
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = base_model.bert(**batch, output_hidden_states=True)['hidden_states']
            out = torch.stack(out, dim=1).detach().cpu()[:,:,0,:]  # We only keep the [CLS] embedding
            outputs.append(out)
        output = torch.cat(outputs, dim=0)

        # Saving the embeddings as numpy arrays
        output = output.detach().numpy()
        np.save(f'./Output/Embeddings/base_{target_lang}.npy', output)