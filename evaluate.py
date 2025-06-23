from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score, f1_score
import json

# Préparation données
dataset = load_dataset("emotion")

def map_label(example):
    label = example["label"]
    if label in [0, 1, 2]:
        return {"label": 0}
    elif label == 3:
        return {"label": 1}
    else:
        return {"label": 2}

dataset = dataset.map(map_label)

model_path = "models/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(model_path)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args
)

# Évaluation
eval_dataset = dataset["test"].select(range(200))
predictions = trainer.predict(eval_dataset)

preds = torch.argmax(torch.tensor(predictions.predictions), dim=-1).numpy()
labels = predictions.label_ids

acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="macro")

print("Accuracy:", acc)
print("F1 Score:", f1)

# Sauvegarde dans fichier metrics.json
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc, "f1": f1}, f)
