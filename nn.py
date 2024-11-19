import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary



parsed_data = []
with open("pose_data.txt", "r") as data_file:
    for line in data_file.readlines():
        line = line.strip()
        line = line.replace('(','')
        line = line.replace(')','')
        data_line = []
        line = line.split(",")
        for val in line:
            if val != '':
                data_line.append(float(val))
        data_line.pop(6)
        parsed_data.append(data_line)
    data_file.close()

parsed_data.pop(-1)
data = np.array(parsed_data)
x = data[:,0:6].astype(np.float32)
y = data[:,6:].astype(np.float32)
y[:, 2:] = y[:, 2:] + 90
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=25)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x = torch.tensor(x_train, dtype=torch.float32)
y = torch.tensor(y_train, dtype=torch.float32)

model = torch.nn.Sequential(
        torch.nn.Linear(6,1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024,1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024,1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024,4)
)

loss_func = torch.nn.L1Loss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
model.cuda()
x.cuda()
y.cuda()
summary(model,input_size=(1,1,6))
batch_size = 15000
epochs = 300
losses = []
for i in range(epochs):
    current_loss = np.inf
    for n in range(0, len(x), batch_size):
       x_batch = x[n:n + batch_size].cuda()
       y_pred = model(x_batch)
       y_batch = y[n:n+ batch_size].cuda()
       loss = loss_func(y_pred, y_batch)
       opt.zero_grad()
       loss.backward()
       opt.step()
    print(f"finished epoch {i}, loss: {loss}")
    losses.append(loss.item())

x_test = torch.tensor(x_test, dtype=torch.float32).cuda()
predictions = []
model.eval()
with torch.no_grad():
    for i in range(len(x_test)):
        preds = model(x_test[i])
        predictions.append(preds)

plt.plot([e for e in range(epochs)],losses)
plt.xlabel("Epochs")
plt.ylabel("Training loss (MAE))")
plt.show()

for j in range(len(predictions[:10])):
    print(f"y test values: {y_test[j]}")
    print(f"predicted values: {predictions[j]}")

#torch.save(model, "4_dof_arm")

