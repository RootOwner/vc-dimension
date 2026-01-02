import torch
import yaml
import numpy as np
import pandas as pd

from models import MLP
from datasets import get_dataloaders

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def spectral_norm(weight, n_iter=20):
    """
    Приближённая спектральная норма (power iteration).
    """
    w = weight.data
    if w.dim() > 2:
        w = w.view(w.size(0), -1)

    u = torch.randn(w.size(0), 1)
    v = torch.randn(w.size(1), 1)

    for _ in range(n_iter):
        v = torch.matmul(w.t(), u)
        v = v / (v.norm() + 1e-8)
        u = torch.matmul(w, v)
        u = u / (u.norm() + 1e-8)

    sigma = torch.matmul(u.t(), torch.matmul(w, v))
    return sigma.item()


def compute_psi(model):
    """
    Ψ̂(θ) = ∏ ||W_l||_2
    """
    psi = 1.0
    norms = []

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            sn = spectral_norm(module.weight)
            norms.append(sn)
            psi *= sn

    return psi, norms


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = config["dataset"]
    model_cfg = config["models"]["model_a"]

    # Нужны только input_dim и num_classes
    _, _, input_dim, num_classes = get_dataloaders(
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

    psi, norms = compute_psi(model)

    print(f"Ψ̂(θ) = {psi:.6f}")

    df = pd.read_csv("../results/metrics.csv")
    df["psi_hat"] = psi
    df.to_csv("../results/metrics.csv", index=False)

    print("Norm-based сложность добавлена в results/metrics.csv")


if __name__ == "__main__":
    main()
