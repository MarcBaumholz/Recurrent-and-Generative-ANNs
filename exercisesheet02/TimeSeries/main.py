import numpy as np
import torch
import matplotlib.pyplot as plt

from data import generate_memory_task_data, CustomDataset
from net import RNN, LSTM, TCN


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


def train_epoch(model, dataloader, optimizer):
    model.train()

    losses = []
    for x, y in dataloader:
        optimizer.zero_grad()
        predictions = model(x)
        loss = torch.nn.functional.cross_entropy(predictions, y.to(torch.int64))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return torch.tensor(losses).mean()


def valid_epoch(model, dataloader):
    model.eval()

    losses = []
    num_correct = 0
    predictions = []
    with torch.no_grad():
        for x, y in dataloader:
            prediction = model(x)
            predictions.append(prediction)
            loss = torch.nn.functional.cross_entropy(prediction, y.to(torch.int64))
            losses.append(loss.item())
            num_correct += (prediction.argmax(dim=1) == y).sum()
    loss = torch.tensor(losses).mean(),
    accuracy = num_correct / len(dataloader.dataset)
    predictions = torch.cat(predictions)
    return loss, accuracy, predictions


def train_and_valid(model, dataloader_train, dataloader_valid, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    losses_train = []
    losses_valid = []
    for epoch in range(epochs):
        loss_train = train_epoch(
            model=model,
            dataloader=dataloader_train,
            optimizer=optimizer,
        )
        loss_valid, accuracy, predictions = valid_epoch(
            model=model,
            dataloader=dataloader_valid,
        )
        losses_train.append(loss_train)
        losses_valid.append(loss_valid)
    return losses_train, losses_valid, accuracy, predictions


def run(task, model_type):
    # Prepare data
    sequence_length = 30
    num_tokens = 3
    if task == "memory":
        data_train = generate_memory_task_data(
            sequence_length=sequence_length,
            num_tokens=num_tokens,
        )
        data_valid = generate_memory_task_data(
            sequence_length=sequence_length,
            num_tokens=num_tokens,
        )
    elif task == "memorygen":
        data_train = generate_memory_task_data(
            sequence_length=sequence_length,
            allowed_positions=np.arange(sequence_length // 2, sequence_length),
            num_tokens=num_tokens,
        )
        data_valid = generate_memory_task_data(
            sequence_length=sequence_length,
            allowed_positions=np.arange(0, sequence_length // 2),
            num_tokens=num_tokens,
        )

    dataset_train = CustomDataset(*data_train)
    dataset_valid = CustomDataset(*data_valid)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=32,
        shuffle=False,
    )

    # Prepare model
    if model_type == "rnn":
        model = RNN(
            input_size=num_tokens + 2,
            hidden_size=34,
            output_size=num_tokens,
        )
    elif model_type == "lstm":
        model = LSTM(
            input_size=num_tokens + 2,
            hidden_size=16,
            output_size=num_tokens,
        )
    elif model_type == "tcn":
        model = TCN(
            input_size=num_tokens + 2,
            hidden_size=14,
            output_size=num_tokens,
        )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Train and validate
    losses_train, losses_valid, accuracy, predictions = train_and_valid(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        epochs=50,
    )
    print(f"Final accuracy of {model_type} ({num_params} parameters) on {task} task: {accuracy}")

    plt.plot(losses_train, label='Train loss')
    plt.plot(losses_valid, label='Valid loss')
    plt.title(f"{model_type} on {task} task")
    plt.legend()
    plt.savefig(f"{task}_{model_type}_losses.png")
    plt.close()

    # Visualize single sequence with prediction
    index = 0
    x, y = dataset_valid[index]
    prediction = predictions[index]
    if task in ["memory", "memorygen"]:
        x = x.argmax(axis=1)
        prediction = prediction.argmax()
        y += 1
        prediction += 1
    plt.plot(x, color='black', label='Input sequence')
    plt.scatter(len(x), y, color='green', label='Target')
    plt.scatter(len(x), prediction, color='red', label='Prediction')
    plt.title('Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{task}_{model_type}_prediction.png")
    plt.close()


for task in ["memory", "memorygen"]:
    for model in ["rnn", "lstm", "tcn"]:
        run(task, model)
    print()
