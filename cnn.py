import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary



class PoseData(torch.utils.data.Dataset):
    def __init__(self, data_tensor, labels):
        self.data_tensor = data_tensor
        self.labels = labels

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self,idx):
        pose_vector = self.data_tensor[idx]
        target = self.labels[idx]
        pose_vector = torch.tensor(pose_vector)
        target = torch.tensor(target)
        pose_vector = torch.unsqueeze(pose_vector,0)
        return {'pose':pose_vector, 'target_joints':target}


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
dataset = PoseData(x_train, y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=50000, shuffle=True)
test_data = PoseData(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

model = torch.nn.Sequential(
   torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
   torch.nn.ReLU(),
   torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
   torch.nn.ReLU(),
   torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
   torch.nn.ReLU(),
   torch.nn.Flatten(),  # Flatten the output from the conv layers
   torch.nn.Linear(32 * 6, 512),  # Adjust the input size according to the output of the conv layers
   torch.nn.ReLU(),
   torch.nn.Linear(512, 4),
# Output 4 joint angles
)

torch.backends.cudnn.benchmark = True
#loss_func = torch.nn.MSELoss()
loss_func = torch.nn.L1Loss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
model.cuda()
summary(model,input_size=(1,1,6))
epochs = 250
losses = []
lowest_loss = 1000000
early_stop_count = 0
for i in range(epochs):
    current_loss = 0.0
    if early_stop_count >= 10:
        print("Stopping early since no improvement")
        break
    for batch in dataloader:
        pose_batch, target_batch, = batch['pose'].cuda(), batch['target_joints'].cuda()
        y_pred = model(pose_batch)
        loss = loss_func(y_pred, target_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        current_loss += loss.item()
    print(f"finished epoch {i}, loss: {loss}")
    epoch_loss = current_loss / len(dataloader)
    losses.append(epoch_loss)
    if epoch_loss < lowest_loss:
        lowest_loss = epoch_loss
        early_stop_count = 0
    else:
        early_stop_count += 1
    

#x_test = torch.tensor(x_test, dtype=torch.float32).cuda()
predictions = []
model.eval()
#single_sample = x_test[0].unsqueeze(0).unsqueeze(0)
#print(single_sample.shape)
with torch.no_grad():
    for batch in test_loader:
       preds = list(model(batch['pose'].cuda()))
       for pred in preds:
           predictions.append(pred)
   

plt.plot([e for e in range(epochs)],losses)
plt.xlabel("Epochs")
plt.ylabel("Training loss (MAE))")
plt.show()

for j in range(len(predictions[:10])):
    print(f"y test values: {y_test[j]}")
    print(f"predicted values: {predictions[j]}")

#torch.save(model, "4_dof_arm")
