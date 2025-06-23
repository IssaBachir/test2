import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from DiamondModel import DiamondModel
from standardisation import standardisation, to_tensor


class Trainner:
    def __init__(self):
        self.X_train = np.load("data/x_train.npy",allow_pickle=True)
        self.y_train = np.load("data/y_train.npy",allow_pickle=True)
        self.X_test = np.load("data/x_test.npy",allow_pickle=True)
        self.y_test = np.load("data/y_test.npy",allow_pickle=True)
        
        self.X_train_t = to_tensor(standardisation(self.X_train))
        self.X_test_t = to_tensor(standardisation(self.X_test))
        self.y_train_t = to_tensor(self.y_train)
        self.y_test_t = to_tensor(self.y_test)

    # Methode d'exactitude modifiée pour la classification multiclasse
    def accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    def train(self, model, epochs=10000, step=1000):

        # Fonction de perte (CrossEntropyLoss pour la classification multiclasse)
        loss_fn = nn.CrossEntropyLoss()

        # Optimiseur
        optimizer = optim.Adam(
            model.parameters(), lr=0.001
        )  # Assurez-vous que votre modèle est défini

        epoch_count = []
        train_acc_list = []
        test_acc_list = []
        train_loss_list = []
        test_loss_list = []

        torch.manual_seed(42)

        for epoch in range(epochs):

            model.train()
            y_logits = model(
                self.X_train_t
            )  # Pas de squeeze ici, sortie de taille (batch_size, 5)
            y_pred = torch.argmax(
                y_logits, dim=1
            )  # Récupère l'indice de la classe avec la probabilité maximale

            # Calcul de la perte/exactitude
            self.y_train_t = self.y_train_t.long()
            self.y_test_t = self.y_test_t.long()

            loss = loss_fn(y_logits, self.y_train_t)
            acc = self.accuracy_fn(self.y_train_t, y_pred)

            # Optimizer zero grad
            optimizer.zero_grad()

            # loss backward
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Test
            model.eval()
            with torch.inference_mode():
                # forward pass
                test_logits = model(self.X_test_t)
                test_pred = torch.argmax(test_logits, dim=1)

                # calculate the test_loss/accurary
                test_loss = loss_fn(test_logits, self.y_test_t)
                test_acc = self.accuracy_fn(self.y_test_t, test_pred)

                # print out what's happening every 10 epoch's
                if epoch % step == 0:
                    epoch_count.append(epoch)
                    train_acc_list.append(acc)
                    test_acc_list.append(test_acc)
                    train_loss_list.append(loss)
                    test_loss_list.append(test_loss)

                    print(
                        f"Epoch:{epoch}, | Loss:{loss:.5f} | Acc={acc:.2f}% | Test Loss:{test_loss:.5f} | Test Acc:{test_acc:.2f}%"
                    )

        return model

    # save the model
    def save_model(model, path):
        torch.save(model.state_dict(), path)

if __name__ == '__main__':
    trainner = Trainner()
    model = DiamondModel(trainner.X_train.shape[1])
    model = trainner.train(model)
    trainner.save_model(model, f"./models/model_final.pth")
    print("Model saved")
