from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os

# Charger les données
dataset = load_dataset("emotion")

# Regrouper les labels : 0-2 négatif, 3 neutre, 4-5 positif
def map_label(example):
    label = example["label"]
    if label in [0, 1, 2]:
        return {"label": 0}  # négatif
    elif label == 3:
        return {"label": 1}  # neutre
    else:
        return {"label": 2}  # positif

dataset = dataset.map(map_label)

# Tokenization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Charger modèle
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Entraînement
training_args = TrainingArguments(
    output_dir="./models",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(1000)),
    eval_dataset=dataset["validation"].select(range(200)),
    compute_metrics=compute_metrics,
)

trainer.train()

# Sauvegarder modèle
trainer.save_model("models/model")
tokenizer.save_pretrained("models/model")
