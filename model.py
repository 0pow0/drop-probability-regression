import torch
from torchinfo import summary

class ProbModel(torch.nn.Module):
    def __init__(self) -> None:
        super(ProbModel, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.ff1 = torch.nn.Linear(2, 8)
        self.ff2 = torch.nn.Linear(8, 16)
        self.ff3 = torch.nn.Linear(16, 32)
        self.ff4 = torch.nn.Linear(32, 16)
        self.ff5 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu(self.ff1(x))
        x = self.relu(self.ff2(x))
        x = self.relu(self.ff3(x))
        x = self.relu(self.ff4(x))
        x = self.ff5(x)
        return x

if __name__ == '__main__':
    model = ProbModel()
    summary(model, input_size=(1024, 2), device='cpu')
    # x = torch.tensor([[10.0, 1.0], [2.0, 1.0]])
    # y = model(x)

