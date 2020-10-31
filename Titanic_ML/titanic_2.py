import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from IPython.display import display, HTML
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

DATA = {
    'train': 'drive/My Drive/KAGGLE/titanic/train.csv',
    'test': 'drive/My Drive/KAGGLE/titanic/test.csv'
}
CUDA_LAUNCH_BLOCKING = 1.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train = pd.read_csv(DATA['train'])
test = pd.read_csv(DATA['test'])
train.head()

train.describe(include='all')

train.isnull().sum().sort_values(ascending=False)

train = train.fillna(train.mean())
test = test.fillna(test.mean())
train.info()
# test.info()

Y = train.iloc[:, 1]
X = train.drop(["Survived", "Name", "Ticket",
                "PassengerId", "Embarked", "Cabin"], axis=1)

X_label = test['PassengerId']
X_test = test.drop(["Name", "Ticket", "PassengerId",
                    "Embarked", "Cabin"], axis=1)

X.replace({'Sex': {'male': 1, 'female': 0}}, inplace=True)
X_test.replace({'Sex': {'male': 1, 'female': 0}}, inplace=True)
# display(HTML(X.to_html()))
X.info()

NUM_PASSENGER = len(Y)
BATCH_SIZE = NUM_PASSENGER
print("Number of passengers: {}".format(NUM_PASSENGER))


def normalize_data(X):
    mean = np.mean(X, axis=0)  # Mean of row
    std = np.std(X, axis=0)  # Std of row
    X[:, 1:] = (X[:, 1:]-mean[1:])/std[1:]
    return X

# Make polynomial Features


def make_feature(X, poly=2):
    poly = PolynomialFeatures(poly)
    X_poly = poly.fit_transform(X)
    # print(X.shape)
    # print(X_poly.shape)
    return X_poly


# Convert into numpy array
X = np.array(X).reshape(X.shape[0], -1).astype(np.float32)
Y = np.array(Y).reshape(-1, 1)
X = make_feature(X, 2)
X = normalize_data(X)

X_test = np.array(X_test).reshape(X_test.shape[0], -1).astype(np.float32)
X_label = np.array(X_label).reshape(-1, 1)
X_test = make_feature(X_test, 2)
X_test = normalize_data(X_test)

NUM_PARAMETER = len(X[0])
print(NUM_PARAMETER)


class LogisticRegression(nn.Module):
    def __init__(self, inp):
        super(LogisticRegression, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(inp, 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def training_step(model, optimizer, error, trainloader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for (target, label) in trainloader:
        model.zero_grad()
        target = Variable(target).view(-1, NUM_PARAMETER).to(device)
        label = Variable(label).view(-1, 1).to(device)

        outputs = model(target)

        prediction = 1 if outputs.data > 0.5 else 0

        loss = error(outputs, label.float())

        loss.backward()
        optimizer.step()
        train_loss += loss.data
        total += len(target)
        correct += (prediction == label).sum()
    acc = correct*1.0/total
    return train_loss, acc


def evaluating_step(model, optimizer, error, valloader):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for (target, label) in valloader:
            target = Variable(target).view(-1, NUM_PARAMETER).to(device)
            label = Variable(label).view(-1, 1).to(device)

            outputs = model(target)
            prediction = 1 if outputs.data > 0.5 else 0
            loss = error(outputs, label.float())
            val_loss += loss.data
            total += len(target)
            correct += (prediction == label).sum()
        acc = correct*1.0/total
    return val_loss, acc


def plot(train_loss_list, train_acc_list, val_loss_list, val_acc_list, EPOCHS):
    fig = plt.figure(figsize=(20, 5))
    fig.add_subplot(1, 2, 1)
    plt.plot(range(EPOCHS), train_loss_list, color='green', label="Train Loss")
    plt.plot(range(EPOCHS), val_loss_list, color="blue", label="Val Loss")
    plt.legend()
    fig.add_subplot(1, 2, 2)
    plt.plot(range(EPOCHS), train_acc_list,
             color='green', label="Train Accuracy")
    plt.plot(range(EPOCHS), val_acc_list, color="blue", label="Val Accuracy")
    plt.legend()
    plt.show()


def train(EPOCHS, model, optimizer, error, trainloader, valloader):
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    for epoch in range(EPOCHS):
        train_loss, train_acc = training_step(
            model, optimizer, error, trainloader)
        val_loss, val_acc = evaluating_step(model, optimizer, error, valloader)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print("Epoch: ({}/{}), Train loss: {:3f}, Train Acc: {:3f}, Val loss: {:3f}, Val Acc: {:3f}".format(epoch+1, EPOCHS, train_loss,
                                                                                                            train_acc, val_loss, val_acc))
    return model, train_loss_list, train_acc_list, val_loss_list, val_acc_list


best_model = LogisticRegression(NUM_PARAMETER).to(device)
TRAIN_DATA = 1
if TRAIN_DATA:
    EPOCHS = 30
    best_val_acc = 0
    best_fold = 0
    kf = KFold()
    for fold, (train_id, val_id) in enumerate(kf.split(X)):
        X_train, X_val = X[train_id], X[val_id]
        y_train, y_val = Y[train_id], Y[val_id]

        trainset = data.TensorDataset(torch.from_numpy(
            X_train), torch.LongTensor(torch.from_numpy(y_train)))
        valset = data.TensorDataset(torch.from_numpy(
            X_val), torch.LongTensor(torch.from_numpy(y_val)))

        trainloader = data.DataLoader(trainset, num_workers=4, batch_size=1)
        valloader = data.DataLoader(valset, num_workers=4, batch_size=1)

        model = LogisticRegression(NUM_PARAMETER).to(device)
        error = nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0002, weight_decay=1e-5)
        print("Start training at fold: {}...".format(fold+1))
        model, train_loss_list, train_acc_list, val_loss_list, val_acc_list = train(
            EPOCHS, model, optimizer, error, trainloader, valloader)
        print("Fold {} completed! Plotting...".format(fold+1))
        plot(train_loss_list, train_acc_list,
             val_loss_list, val_acc_list, EPOCHS)
        if val_acc_list[-1] > best_val_acc:
            best_val_acc = val_acc_list[-1]
            torch.save(model.state_dict(
            ), f"drive/My Drive/KAGGLE/titanic/checkpoint/best_fold_{best_fold}.pt")
            best_model = model
            best_fold = fold+1

    print("Best fold: {}".format(best_fold))


def predict(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        X = Variable(torch.from_numpy(X)).view(-1, NUM_PARAMETER).to(device)
        outputs = model(X)
        preds = [1 if pred > 0.5 else 0 for pred in outputs.data]

    return preds


best_model.load_state_dict(torch.load(
    f"drive/My Drive/KAGGLE/titanic/checkpoint/best_fold_2.pt"))
result = predict(best_model, X_test)
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['PassengerID', 'Survived'])
    for i, j in tqdm(zip(X_label, result)):
        writer.writerow([i, j])
