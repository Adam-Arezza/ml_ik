import torch

model = torch.load("4_dof_arm")
point = [0.2, 0.2, 0.2]
x = torch.tensor(point, dtype=torch.float32).cuda()
prediction = model(x)
print(prediction)
