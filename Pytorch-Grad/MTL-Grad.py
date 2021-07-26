import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/Grads')


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)


def convert_to_tensor():
    tensor_x = torch.tensor([2.])
    tensor_y1 = torch.tensor([4.])
    tensor_y2 = torch.tensor([3.])
    processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y1, tensor_y2)
    return processed_dataset


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=1, out_features=1)
        self.fc2 = nn.Linear(in_features=1, out_features=1)

        self.fc3 = nn.Linear(in_features=1, out_features=1)
        self.fc4 = nn.Linear(in_features=1, out_features=1)

        self.fc5 = nn.Linear(in_features=1, out_features=1)
        self.fc6 = nn.Linear(in_features=1, out_features=1)

        torch.nn.init.ones_(self.fc1.weight)
        torch.nn.init.ones_(self.fc1.bias)
        torch.nn.init.ones_(self.fc2.weight)
        torch.nn.init.ones_(self.fc2.bias)
        torch.nn.init.ones_(self.fc3.weight)
        torch.nn.init.ones_(self.fc3.bias)
        torch.nn.init.ones_(self.fc4.weight)
        torch.nn.init.ones_(self.fc4.bias)
        torch.nn.init.ones_(self.fc5.weight)
        torch.nn.init.ones_(self.fc5.bias)
        torch.nn.init.ones_(self.fc6.weight)
        torch.nn.init.ones_(self.fc6.bias)


    def forward(self, t):
        # input layer
        t = self.fc2(self.fc1(t))

        t1 = self.fc4(self.fc3(t))
        t2 = self.fc6(self.fc5(t))

        return t1, t2


trainloader = torch.utils.data.DataLoader(convert_to_tensor(), batch_size=1, shuffle=True)
model = Network()

X, Y1, Y2 = next(iter(trainloader))
print(X.size())
writer.add_graph(model, X)
writer.close()

# optimizer = optim.SGD([{'params': model.fc1.parameters()}], lr=1)
optimizer = optim.SGD(model.parameters(), lr=1)
loss = nn.MSELoss(reduction="sum")

y_hat1, y_hat2 = model(X)
optimizer.zero_grad()
L1 = loss(Y1, y_hat1)
# L1.backward(retain_graph=True)
print("loss: {0}".format(L1))

L2 = loss(Y2, y_hat2)
print("loss: {0}".format(L2))

# L2.backward(retain_graph=True)

loss = L1 + L2
loss.backward()

print("Y_hat1: {0}".format(y_hat1))
print("Y_hat2: {0}".format(y_hat2))
print("--" * 10)
print("Initial parameters")
print("Weight: {0}".format(model.fc1.weight))
print("Bias: {0}".format(model.fc1.bias))
print("--" * 10)
print("Grads")
print("FC1 Weight Grad:{0}".format(model.fc1.weight.grad))
print("FC1 Bias Grad:{0}".format(model.fc1.bias.grad))

print("FC2 Weight Grad:{0}".format(model.fc2.weight.grad))
print("FC2 Bias Grad:{0}".format(model.fc2.bias.grad))

print("FC3 Weight Grad:{0}".format(model.fc3.weight.grad))
print("FC3 Bias Grad:{0}".format(model.fc3.bias.grad))

print("FC4 Weight Grad:{0}".format(model.fc4.weight.grad))
print("FC4 Bias Grad:{0}".format(model.fc4.bias.grad))

print("FC5 Weight Grad:{0}".format(model.fc5.weight.grad))
print("FC5 Bias Grad:{0}".format(model.fc5.bias.grad))

print("FC6 Weight Grad:{0}".format(model.fc6.weight.grad))
print("FC6 Bias Grad:{0}".format(model.fc6.bias.grad))

optimizer.step()

print("--" * 10)
print("New parameters")
print("FC1 Weight: {0}".format(model.fc1.weight))
print("FC1 Bias:{0}".format(model.fc1.bias))

print("FC2 Weight:{0}".format(model.fc2.weight))
print("FC2 Bias:{0}".format(model.fc2.bias))

print("FC3 Weight:{0}".format(model.fc3.weight))
print("FC3 Bias:{0}".format(model.fc3.bias))

print("FC4 Weight:{0}".format(model.fc4.weight))
print("FC4 Bias:{0}".format(model.fc4.bias))

print("FC5 Weight:{0}".format(model.fc5.weight))
print("FC5 Bias:{0}".format(model.fc5.bias))

print("FC6 Weight:{0}".format(model.fc6.weight))
print("FC6 Bias:{0}".format(model.fc6.bias))
