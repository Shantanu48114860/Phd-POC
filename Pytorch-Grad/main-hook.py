import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Function
writer = SummaryWriter('runs/Grads')


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)


def convert_to_tensor():
    tensor_x = torch.tensor([2.])
    tensor_y = torch.tensor([4.])
    processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return processed_dataset


class Grad_Reverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_out):
        out = -ctx.alpha * grad_out
        return out, None


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.ones_(self.fc1.weight)
        torch.nn.init.ones_(self.fc1.bias)
        # self.fc2 = nn.Linear(in_features=120, out_features=60)
        # self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # input layer
        t = self.fc1(t)
        t = Grad_Reverse.apply(t, 1)
        return t


trainloader = torch.utils.data.DataLoader(convert_to_tensor(), batch_size=1, shuffle=True)
model = Network()

X, Y = next(iter(trainloader))
print(X.size())
writer.add_graph(model, X)
writer.close()

optimizer = optim.SGD(model.parameters(), lr=1)
loss = nn.MSELoss(reduction="sum")

y_hat = model(X)
L = loss(Y, y_hat)
print("loss: {0}".format(L))
optimizer.zero_grad()
L.backward(retain_graph=True)

print("Y_hat: {0}".format(y_hat))
print("--" * 10)
print("Initial parameters")
print("Weight: {0}".format(model.fc1.weight))
print("Bias: {0}".format(model.fc1.bias))
print("--" * 10)
print("Grads")
print("Weight Grad:{0}".format(model.fc1.weight.grad))
print("Bias Grad:{0}".format(model.fc1.bias.grad))


optimizer.step()

print("--" * 10)
print("New parameters")
print("Weight: {0}".format(model.fc1.weight))
print("Bias: {0}".format(model.fc1.bias))
