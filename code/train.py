import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from models import MLP, count_parameters
from datasets import get_dataloaders


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # Данные
    dataset_cfg = config["dataset"]

    train_loader, test_loader, input_dim, num_classes = get_dataloaders(
        dataset_name=dataset_cfg["name"],
        batch_size=dataset_cfg["batch_size"]
    )

    # Архитектура (пример: две сети с одинаковым W)
    model_cfg = config["models"]["model_a"]

    hidden_dims = [model_cfg["hidden_dim"]] * model_cfg["depth"]

    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes
    ).to(device)

    W = count_parameters(model)
    print(f"Число параметров W = {W}")


    # Конфигурация обучения
    train_cfg = config["training"]

    # Оптимизация
    if train_cfg["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"]
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            momentum=0.9,
            weight_decay=train_cfg["weight_decay"]
        )
    criterion = nn.CrossEntropyLoss()

    # Обучение
    for epoch in range(train_cfg["epochs"]):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )
        print(f"Epoch {epoch + 1}: loss = {loss:.4f}")

    # Сохранение модели
    os.makedirs("../results", exist_ok=True)
    torch.save(model.state_dict(), "../results/model.pt")
    print("Модель сохранена: results/model.pt")


if __name__ == "__main__":
    main()
