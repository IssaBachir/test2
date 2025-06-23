from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score, f1_score
import json

# Chargement dataset
dataset = load_dataset("emotion")

# Re-catégorisation des labels (exemple : 3 classes)
def map_label(example):
    label = example["label"]
    if label in [0, 1, 2]:
        return {"label": 0}
    elif label == 3:
        return {"label": 1}
    else:
        return {"label": 2}

dataset = dataset.map(map_label)

# Chargement tokenizer du modèle sauvegardé
model_path = "models/model_final.pth"  # adapte le chemin si besoin
# IMPORTANT : ici tu dois avoir le dossier du modèle, pas juste le .pth !
# Si tu as juste le .pth, il faut charger différemment, je te conseille de sauvegarder aussi le config/tokenizer !

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Chargement du modèle pré-entraîné
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
)

# Sélection d’un sous-ensemble de test pour accélérer l’évaluation
eval_dataset = dataset["test"].select(range(200))

# Prédiction
predictions = trainer.predict(eval_dataset)

preds = torch.argmax(torch.tensor(predictions.predictions), dim=-1).numpy()
labels = predictions.label_ids

acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="macro")

print(f"Accuracy: {acc}")
print(f"F1 Score: {f1}")

# Sauvegarde des métriques dans un fichier JSON
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc, "f1": f1}, f)

