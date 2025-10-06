import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(6, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 2)
    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(x)
        return x
class Trainer:
    def __init__(self,mutation_rate,mutation_strength, crossover_rate):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate

    def CrossOver(self, net1, net2):
        child = {}
        for key in net1.keys():
            w1 = net1[key].detach().clone()
            w2 = net2[key].detach().clone()
            mask = torch.rand_like(w1) < self.crossover_rate
            w_child = torch.where(mask, w1, w2)
            child[key] = w_child
        return child

    def Mutatuion(self, child):
        mutation = {}
        for key, param in child.items():
            param = param.clone().detach()
            mask = torch.rand_like(param) < self.mutation_rate
            noise = torch.randn_like(param) * self.mutation_strength
            param = param + mask * noise
            mutation[key] = param
        return mutation

    def Train(self, net1, net2, nets):
        for net in nets:
            child_state = self.Mutatuion(self.CrossOver(net1, net2))
            net.load_state_dict(child_state)