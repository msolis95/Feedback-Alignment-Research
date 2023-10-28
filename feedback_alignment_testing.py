import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# create a tensor on CPU
test_tensor = torch.tensor([1, 2, 3])

if torch.cuda.is_available():
    device = torch.device("cuda")
    test_tensor = test_tensor.to(device)
    print("Tensor moved to GPU")
else:
    device = "cpu"
    print("GPU is not available, using CPU instead")

# MNIST DATA
batch_size = 100

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
testset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
print("Train X dataset shape:", trainset.data.shape)
print("Test X dataset shape:", testset.data.shape)




# Activation Functions
def tanh(x):
  return torch.tanh(x)

def dtanh(x):
  return 1 - (tanh(x)**2)

def logistic(x):
  return (1 + torch.exp(-x))**(-1)

def dlogistic(x):
  exp = torch.exp(-x)
  return exp*(1 + exp)**(-2)

# Loss Functions
def BCE(y_hat, y): # loss function from paper shown below
  return -(y*torch.log(y_hat) + (1 - y)*torch.log(1 - y_hat))

def dBCE(y_hat, y):
  return -((y/y_hat) - ((1 - y)/(1 - y_hat)))

def MSE(y_hat, y):
  return (y_hat - y)**2

def dMSE(y_hat, y):
  return 2*(y_hat - y)



def get_normalized_inner_product(tensor1, tensor2):
  vector1 = tensor1.view(tensor1.shape[0]*tensor1.shape[1])
  vector2 = tensor2.view(tensor2.shape[0]*tensor2.shape[1])
  num = torch.dot(vector1, vector2)
  den = torch.linalg.vector_norm(tensor1)*torch.linalg.vector_norm(tensor2)
  return (num/den).item()



class NeuralNetwork():
    def __init__(self, dims, FA_type, batch_size, Ws, bs, f, df, fy, dfy, loss_fn, dloss_fn, random_features=False):
        super().__init__()
        self.FA_type = FA_type
        self.random_features = random_features
        self.Ws = [W.detach().clone() for W in Ws]
        self.bs = [b.detach().clone() @ torch.ones(1, batch_size).to(device) for b in bs]
        self.Bs = self.create_Bs(dims)
        for x in (self.Ws + self.bs + self.Bs):
          x.requires_grad = True
        self.f = f
        self.df = df
        self.fy = fy
        self.dfy = dfy
        self.loss_fn = loss_fn
        self.dloss_fn = dloss_fn

    def create_Bs(self, dims):
      match self.FA_type:
        case NeuralNetwork.BP:
          return [] # does not use B_i
        case NeuralNetwork.FA:
          return [torch.rand_like(self.Ws[1].T).to(device) - 0.5,
                  torch.rand_like(self.Ws[2].T).to(device) - 0.5]
        case NeuralNetwork.DFA:
          return [torch.rand(dims[1], self.Ws[2].T.shape[1]).to(device) - 0.5,
                  torch.rand_like(self.Ws[2].T).to(device) - 0.5]
        case NeuralNetwork.IFA:
          return [torch.rand(dims[1], self.Ws[2].T.shape[1]).to(device) - 0.5] # does not use B2

    def update_params(self, dWs, dbs):
      if not self.random_features:
        self.Ws[0] = (self.Ws[0] - alpha*dWs[0]).detach().clone()
      self.Ws[1] = (self.Ws[1] - alpha*dWs[1]).detach().clone()
      self.Ws[2] = (self.Ws[2] - alpha*dWs[2]).detach().clone()
      if not self.random_features:
        self.bs[0] = (self.bs[0] - alpha*dbs[0]).detach().clone()
      self.bs[1] = (self.bs[1] - alpha*dbs[1]).detach().clone()
      self.bs[2] = (self.bs[2] - alpha*dbs[2]).detach().clone()

      for x in (self.Ws + self.bs):
        x.requires_grad = True

    def forward(self, x, y, back, alpha=0):
      N = y.shape[0]
      # (1)
      a1 = (self.Ws[0] @ x) + self.bs[0] # = W1*x + b1
      a1.retain_grad()
      h1 = self.f(a1)

      # (2)
      a2 = (self.Ws[1] @ h1) + self.bs[1] # = W2*h1 + b2
      a2.retain_grad()
      h2 = self.f(a2)

      # (3)
      ay = (self.Ws[2] @ h2) + self.bs[2] # = W3*h2 + b3
      y_hat = self.fy(ay)

      # (4)
      J = (1/(N*y.shape[1]))*torch.sum(torch.sum(self.loss_fn(y_hat, y))) # loss
      dWs_inner_products = None
      das_inner_products = None
      if back:
        J.backward()
        dWs, dbs, das = self.backward(N, x, y, a1, h1, a2, h2, ay, y_hat, alpha)
        dWs_inner_products = torch.tensor([get_normalized_inner_product(dW, W.grad) for dW, W in zip(dWs, self.Ws)])
        das_inner_products = torch.tensor([get_normalized_inner_product(da, a.grad) for da, a in zip(das, [a1, a2])])
        self.update_params(dWs, dbs)

      return y_hat, J.item(), dWs_inner_products, das_inner_products

    def backward(self, N, x, y, a1, h1, a2, h2, ay, y_hat, alpha):
      # (5)
      dy_hat = (1/(N*y.shape[1]))*self.dloss_fn(y_hat, y)
      e = self.dfy(ay) * dy_hat # = day

      da1, da2 = self.FA_type(self, e, a1, a2)

      # Gradients for b_i
      db1 = da1
      db2 = da2
      db3 = e # = day

      # (10) updating parameters
      dW1 = da1 @ x.T
      dW2 = da2 @ h1.T
      dW3 = e @ h2.T

      return [dW1, dW2, dW3], [db1, db2, db3], [da1, da2]

    def BP(self, e, a1, a2):
      da2 = (self.Ws[2].T @ e) * self.df(a2) # W3.T*day * df(a2)
      da1 = (self.Ws[1].T @ da2) * self.df(a1) # W2.T*da2 * df(a1)
      return da1, da2

    def FA(self, e, a1, a2):
      # (7) FA
      da2 = (self.Bs[1] @ e) * self.df(a2) # B2*day * df(a2)
      da1 = (self.Bs[0] @ da2) * self.df(a1) # B1*da2 * df(a1)
      return da1, da2

    def DFA(self, e, a1, a2):
      # (8) DFA
      da1 = (self.Bs[0] @ e) * self.df(a1) # B1*day * df(a1)
      da2 = (self.Bs[1] @ e) * self.df(a2) # B2*day * df(a2)
      return da1, da2

    def IFA(self, e, a1, a2):
      # (9) IFA
      da1 = (self.Bs[0] @ e) * self.df(a1) # B1*day * df(a1)
      da2 = (self.Ws[1] @ da1) * self.df(a2) # W2*da1 * df(a2)
      return da1, da2



def init_networks(dims, loss_fn, dloss_fn, random_features=False):
  f = tanh
  df = dtanh
  fy = logistic
  dfy = dlogistic

  Ws = [torch.rand(dims[1], dims[0]).to(device) - 0.5,
        torch.rand(dims[2], dims[1]).to(device) - 0.5,
        torch.rand(dims[3], dims[2]).to(device) - 0.5]
  bs = [torch.rand(dims[1], 1).to(device) - 0.5,
        torch.rand(dims[2], 1).to(device) - 0.5,
        torch.rand(dims[3], 1).to(device) - 0.5]

  BP_model = NeuralNetwork(dims, NeuralNetwork.BP, batch_size, Ws, bs, f, df, fy, dfy, loss_fn, dloss_fn, random_features=True)
  FA_model = NeuralNetwork(dims, NeuralNetwork.FA, batch_size, Ws, bs, f, df, fy, dfy, loss_fn, dloss_fn, random_features=True)
  DFA_model = NeuralNetwork(dims, NeuralNetwork.DFA, batch_size, Ws, bs, f, df, fy, dfy, loss_fn, dloss_fn, random_features=True)
  IFA_model = NeuralNetwork(dims, NeuralNetwork.IFA, batch_size, Ws, bs, f, df, fy, dfy, loss_fn, dloss_fn, random_features=True)
  return BP_model, FA_model, DFA_model, IFA_model



def train_all(dataloader, BP_model, FA_model, DFA_model, IFA_model, back):
    space = ""
    size = len(dataloader.dataset)
    num_batches = size//batch_size
    losses = torch.zeros(4)
    dWs_inner_products, das_inner_products = torch.empty(4, num_batches, 3), torch.empty(4, num_batches, 2)
    for batch, (X, y) in enumerate(dataloader):
      # reshaping the data
      X, y = X.to(device), y.to(device)
      X_manual = X.view(batch_size, input_dim).T
      one_hot_y = torch.zeros((y.shape[0], 10)).to(device)
      for i in range(y.shape[0]):
        one_hot_y[i, y[i]] = 1

      dWs_IP = [0, 0, 0, 0]
      das_IP = [0, 0, 0, 0]

      # training all neural networks
      BP_loss, dWs_IP[0], das_IP[0] = BP_model.forward(X_manual, one_hot_y.T, back, alpha)[1:]
      FA_loss, dWs_IP[1], das_IP[1] = FA_model.forward(X_manual, one_hot_y.T, back, alpha)[1:]
      DFA_loss, dWs_IP[2], das_IP[2]  = DFA_model.forward(X_manual, one_hot_y.T, back, alpha)[1:]
      IFA_loss, dWs_IP[3], das_IP[3]  = IFA_model.forward(X_manual, one_hot_y.T, back, alpha)[1:]

      if back:
        for i in range(4):
          dWs_inner_products[i, batch], das_inner_products[i, batch] = dWs_IP[i], das_IP[i]

      batch_losses = [BP_loss, FA_loss, DFA_loss, IFA_loss]
      for i in range(len(batch_losses)):
        losses[i] += batch_losses[i]

      if batch % 200 == 0:
        print(f"{space}Batch {batch+1}/{num_batches}: BP average loss: {round(BP_loss, 6)}")
        space = "\n"
        print(f"               FA average loss: {round(FA_loss, 6)}")
        print(f"              DFA average loss: {round(DFA_loss, 6)}")
        print(f"              IFA average loss: {round(IFA_loss, 6)}\n")

    return losses/num_batches, test_all(testloader, BP_model, FA_model, DFA_model, IFA_model), dWs_inner_products, das_inner_products

def test_all(dataloader, BP_model, FA_model, DFA_model, IFA_model):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_losses = torch.zeros(4)
  accuracies = torch.zeros(4)
  models = [BP_model, FA_model, DFA_model, IFA_model]
  for X, y in dataloader:
    # reshaping the data
    X, y = X.view(X.shape[0], input_dim).T.to(device), y.to(device)
    one_hot_y = torch.zeros((y.shape[0], 10)).to(device)
    for i in range(y.shape[0]):
      one_hot_y[i, y[i]] = 1

    # calculating loss and accuracy for each model
    for i in range(len(models)):
      pred, loss = models[i].forward(X, one_hot_y.T, False, alpha)[:2]
      test_losses[i] += loss
      accuracies[i] += (pred.argmax(0) == y).type(torch.float).sum().item()

  test_losses /= num_batches
  accuracies *= 100/size
  FA_types = ["BP", "FA", "DFA", "IFA"]
  for i in range(len(FA_types)):
    print(f"{FA_types[i]} Test Error:\n     Accuracy: {round(accuracies[i].item(), 2)}%, Avg loss: {round(test_losses[i].item(), 6)}\n")
  return accuracies



def machine_learn(epochs, dims, loss_fn, dloss_fn, random_features=False):
  BP_model, FA_model, DFA_model, IFA_model = init_networks(dims, loss_fn, dloss_fn, random_features=True)

  size = len(trainloader.dataset)
  num_batches = size//batch_size

  losses = torch.empty(4, epochs)
  accuracies = torch.empty(4, epochs+1)
  dWs_inner_products = torch.empty(4, (epochs+1)*num_batches, 3)
  das_inner_products = torch.empty(4, (epochs+1)*num_batches, 2)

  for t in range(epochs+1):
    print(f"-------------------------------\nEpoch {t}\n-------------------------------")
    if t == 0:
      accuracies[:, t] = test_all(testloader, BP_model, FA_model, DFA_model, IFA_model)
    else:
      back = True
      if t == epochs:
        back = False
      epoch_losses, epoch_accuracies, epoch_dWs_IP, epoch_das_IP = train_all(trainloader, BP_model, FA_model, DFA_model, IFA_model, back)
      losses[:, t-1] = epoch_losses
      accuracies[:, t] = epoch_accuracies
      if back:
        dWs_inner_products[:, (t-1)*num_batches:t*num_batches] = epoch_dWs_IP
        das_inner_products[:, (t-1)*num_batches:t*num_batches] = epoch_das_IP
  print("Done!")
  return losses, accuracies, dWs_inner_products, das_inner_products, num_batches



def graph_data(epochs, num_batches, losses, accuracies, dWs_inner_products, das_inner_products, random_features=True):
  FA_types = ["BP", "FA", "DFA", "IFA"]

  # losses
  ts = range(1, epochs+1)
  for i in range(4):
    plt.plot(ts, losses[i], label=FA_types[i])

  plt.xlabel("Epoch")
  plt.ylabel("Training Loss", fontsize=12)
  plt.title("Training Loss over Time")
  plt.legend()
  plt.show()

  # accuracies
  ts = range(epochs+1)
  for i in range(4):
    plt.plot(ts, accuracies[i], label=FA_types[i])

  plt.xlabel("Epoch")
  plt.ylabel("Testing Accuracy (%)", fontsize=12)
  plt.title("Testing Accuracy over Time")
  plt.legend()
  plt.show()

  # dWs normalized inner products
  ts = torch.arange(0, epochs+1, step=1/num_batches)
  Ws = list(range(3))
  if random_features:
    Ws.remove(1)
  for W in Ws:
    for i in range(4):
      plt.plot(ts, dWs_inner_products[i, :, W], label=FA_types[i])

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Inner Product", fontsize=12)
    plt.title(f"Normalized Inner Product between dW{W+1} in each FA type and BP over Time")
    plt.legend()
    plt.show()

  # das normalized inner_products
  ts = torch.arange(0, epochs+1, step=1/num_batches)
  a_vals = list(range(2))
  if random_features:
    a_vals.remove(1)
  for a in a_vals:
    for i in range(4):
      plt.plot(ts, das_inner_products[i, :, a], label=FA_types[i])

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Inner Product", fontsize=12)
    plt.title(f"Normalized Inner Product between da{a+1} in each FA type and BP over Time")
    plt.legend()
    plt.show()



# Dimensions of Neurons
input_dim = 784
layer1_dim = 128
layer2_dim = 64
output_dim = 10

dims = [input_dim, layer1_dim, layer2_dim, output_dim]

alpha = 0.01 # learning rate
epochs = 100



losses, accuracies, dWs_inner_products, das_inner_products, num_batches = machine_learn(epochs, dims, BCE, dBCE)

graph_data(epochs, num_batches, losses, accuracies, dWs_inner_products, das_inner_products)

losses, accuracies, dWs_inner_products, das_inner_products, num_batches = machine_learn(epochs, dims, MSE, dMSE)

graph_data(epochs, num_batches, losses, accuracies, dWs_inner_products, das_inner_products)



# Dimensions of Neurons for Random Features Model
input_dim = 784
layer1_dim = 500
layer2_dim = 64
output_dim = 10

dims = [input_dim, layer1_dim, layer2_dim, output_dim]

losses, accuracies, dWs_inner_products, das_inner_products, num_batches = machine_learn(epochs, dims, BCE, dBCE, random_features=True)

graph_data(epochs, num_batches, losses, accuracies, dWs_inner_products, das_inner_products)
