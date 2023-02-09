from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate

"""
Fine-tuning mBERT in source languages on XNLI
"""

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize_function(examples):
    try:
        return tokenizer(examples["premise"], examples["hypothesis"],
                         padding="max_length", max_length=128, truncation='only_first')
    except:
        return tokenizer(examples["premise"], examples["hypothesis"],
                         padding="max_length", max_length=128, truncation=True)

# Load evaluation metric
metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Choose source languages to fine-tune on
source_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

# Choose layers that should be frozen during fine-tuning
encoder_layers_to_freeze = []  # [1,2,6]

for source_lang in source_langs:

    # Load train set in source language
    dataset = load_dataset("xnli", source_lang)

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)

    # Freezing layers
    for i, layer in enumerate(model.bert.encoder.layer):
        if i+1 in encoder_layers_to_freeze:
            for params in layer.parameters():
                params.requires_grad = False

    # Tokenize data
    tokenized_datasets = dataset.map(tokenize_function, batched=True).shuffle(True)

    # Define name of fine-tuned model
    model_name = f'model_{source_lang}' if len(encoder_layers_to_freeze) == 0 else f'model_{source_lang}_' + '_'.join(
        [str(l) for l in encoder_layers_to_freeze])

    # Define hyper-parameters
    training_args = TrainingArguments(output_dir=f"trainer/{model_name}",
                                      evaluation_strategy="epoch",
                                      learning_rate=2e-5,
                                      save_strategy="no",
                                      #save_strategy='epoch',
                                      num_train_epochs=3,
                                      per_device_train_batch_size=32,
                                      per_device_eval_batch_size=32,
                                      warmup_steps=0,
                                      do_train=True,
                                      do_eval=True,
                                      report_to="none",
                                      optim='adamw_torch'
                                      )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save fine-tuned model
    trainer.save_model(f'./Models/{model_name}')