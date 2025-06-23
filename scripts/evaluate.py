import torch
import numpy as np
from sklearn.metrics import accuracy_score
import json
from DiamondModel import DiamondModel
from standardisation import standardisation, to_tensor

def evaluate():
    X_test = np.load("data/x_test.npy", allow_pickle=True)
    y_test = np.load("data/y_test.npy", allow_pickle=True)

    X_test_t = to_tensor(standardisation(X_test))
    y_test_t = to_tensor(y_test).long()

    model = DiamondModel(X_test.shape[1])
    model.load_state_dict(torch.load("models/model_final.pth"))
    model.eval()

    with torch.inference_mode():
        logits = model(X_test_t)
        preds = torch.argmax(logits, dim=1)

    acc = accuracy_score(y_test_t.numpy(), preds.numpy())

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

if __name__ == "__main__":
    evaluate()
