import math
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# NOTE: you can scroll down and uncomment the chunk of code with matplotlib to see a visualization of the NN learning process


class CustomMathDataset(Dataset):
    def __init__(self, length, going_in, going_out):
        self.length = length
        self.going_in = going_in
        self.going_out = going_out

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x = torch.tensor([self.going_in(i)], dtype=torch.float32)
        y = torch.tensor([self.going_out(x)], dtype=torch.float32)
        return x, y


training_dataset = CustomMathDataset(
    length=50_000,
    going_in=lambda x: (x / 50_000) * (2 * math.pi),
    going_out=lambda x: torch.sin(x),
)

testing_dataset = CustomMathDataset(
    length=10_000,
    going_in=lambda x: (x / 10_000) * (2 * math.pi),
    going_out=lambda x: torch.sin(x),
)

training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
testing_dataset = DataLoader(testing_dataset, batch_size=64, shuffle=True)

training_ins, training_outs = next(iter(training_dataloader))
print("training_ins shape:", training_ins.size())
print("training_outs shape:", training_outs.size())
print("training_ins[0]:", training_ins[0])
print("training_outs[0]:", training_outs[0])

device = "cpu"
if torch.accelerator.is_available():
    if accelerator := torch.accelerator.current_accelerator():
        device = accelerator.type

print("device:", device)


class MathNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.float()

    def forward(self, x):
        # not needed because input is just one number
        # x = self.flatten(x)
        return self.linear_relu_stack(x)


model = MathNN().to(device)
print(model)


def training_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * batch_size + len(X)
            print(f"[{current:>5d}/{size:>5d}] loss: {loss:>7f}")


def testing_loop(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    first_batch_plotted = False

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # if not first_batch_plotted and epoch is not None:
            #     # Convert to NumPy for plotting
            #     x_np = X.squeeze().numpy()
            #     y_true_np = y.squeeze().numpy()
            #     y_pred_np = pred.squeeze().numpy()

            #     sort_idx = np.argsort(x_np)
            #     x_np = x_np[sort_idx]
            #     y_true_np = y_true_np[sort_idx]
            #     y_pred_np = y_pred_np[sort_idx]

            #     plt.figure(figsize=(8, 4))
            #     plt.plot(x_np, y_true_np, label="sin(x)", color="blue")
            #     plt.plot(x_np, y_pred_np, label="model(x)", color="orange")
            #     plt.title(f"sin(x) vs model(x) | epoch {epoch}")
            #     plt.xlabel("x")
            #     plt.ylabel("y")
            #     plt.legend()
            #     plt.grid(True)
            #     plt.savefig(f"plots/epoch_{epoch:03d}.png")
            #     plt.close()

            #     first_batch_plotted = True

    test_loss /= num_batches

    print(f"test:\n\tloss: {test_loss:>8f}%\n")


learning_rate = 2e-5  # 0.0001
batch_size = 64
epochs = 50

loss_fn = nn.MSELoss()  # Mean Square Error, used for regression
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"epoch {t + 1}/{epochs}")
    training_loop(training_dataloader, model, loss_fn, optimizer)
    testing_loop(testing_dataset, model, loss_fn, epoch=t + 1)

print("finished")
