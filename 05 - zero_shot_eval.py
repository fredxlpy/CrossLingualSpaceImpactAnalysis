from datasets import load_dataset
import evaluate
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch

"""
Loading the models that were fine-tuned in different source languages
and evaluate them in target languages
"""

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize_function(examples):
    try:
        return tokenizer(examples["premise"], examples["hypothesis"],
                     padding="max_length", max_length=128, truncation='only_first')
    except:
        return tokenizer(examples["premise"], examples["hypothesis"],
                         padding="max_length", max_length=128, truncation=True)

# Create evaluation metric (= Accuracy)
metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Choose source and target languages
source_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
target_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']


accuracies = []
for source_lang in source_langs:

    # Load model that has been fine-tuned on a given source language
    model = AutoModelForSequenceClassification.from_pretrained(f'./Models/model_{source_lang}').to(device)
    model.eval()

    for target_lang in target_langs:

        # Load and tokenize test set in target language
        test_set = load_dataset("xnli", target_lang, split='test')
        test_set = test_set.map(tokenize_function, batched=True)
        test_set.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

        # Evaluate
        training_args = TrainingArguments(output_dir=f"Trainer_eval",
                                          evaluation_strategy="epoch",
                                          per_device_eval_batch_size=64,
                                          do_train=False,
                                          do_eval=True,
                                          save_strategy="no"
                                          )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_set,
            compute_metrics=compute_metrics,
        )

        result = trainer.evaluate()

        accuracies.append([source_lang, target_lang, result['eval_accuracy']])

full_results = pd.DataFrame(accuracies, columns=['Source', 'Target', 'Accuracy'])
full_results.to_excel('./Output/ZS_results.xlsx', index=False)