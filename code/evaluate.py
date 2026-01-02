import os
import torch
import yaml
import numpy as np
import pandas as pd

from models import MLP
from datasets import get_dataloaders


def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "config.yaml")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise RuntimeError("config.yaml пустой или некорректный")

    return config


@torch.no_grad()
def evaluate_accuracy_and_margin(model, loader, device):
    model.eval()
    correct, total = 0, 0
    margins = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        for i in range(logits.size(0)):
            true_logit = logits[i, y[i]].item()
            other_logits = torch.cat(
                (logits[i, :y[i]], logits[i, y[i] + 1:])
            )
            margins.append(true_logit - other_logits.max().item())

    accuracy = correct / total
    mean_margin = float(np.mean(margins))
    return accuracy, mean_margin


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = config["dataset"]
    model_cfg = config["models"]["model_a"]

    _, test_loader, input_dim, num_classes = get_dataloaders(
        dataset_name=dataset_cfg["name"],
        batch_size=dataset_cfg["batch_size"]
    )

    hidden_dims = [model_cfg["hidden_dim"]] * model_cfg["depth"]

    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(
        torch.load("../results/model.pt", map_location=device)
    )

    accuracy, margin = evaluate_accuracy_and_margin(
        model, test_loader, device
    )

    print(f"Accuracy = {accuracy:.4f}")
    print(f"Mean margin = {margin:.4f}")

    df = pd.DataFrame([{
        "accuracy": accuracy,
        "margin": margin
    }])

    df.to_csv("../results/metrics.csv", index=False)
    print("Метрики сохранены: results/metrics.csv")


if __name__ == "__main__":
    main()
