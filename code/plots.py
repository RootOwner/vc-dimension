import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Пути
    results_path = Path("../results/metrics.csv")
    figures_path = Path("../figures")
    figures_path.mkdir(parents=True, exist_ok=True)

    # Загрузка данных
    df = pd.read_csv(results_path)

    if not {"accuracy", "margin", "psi_hat"}.issubset(df.columns):
        raise ValueError(
            "metrics.csv должен содержать столбцы: accuracy, margin, psi_hat"
        )

    psi = df["psi_hat"]
    acc = df["accuracy"]
    margin = df["margin"]

    # ---------------------------
    # Accuracy vs Psi
    # ---------------------------
    plt.figure(figsize=(5, 4))
    plt.scatter(psi, acc, s=60, alpha=0.8)
    plt.xlabel(r"$\hat{\Psi}(\theta)$")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(figures_path / "accuracy_vs_psi.pdf")
    plt.close()

    # ---------------------------
    # Margin vs Psi
    # ---------------------------
    plt.figure(figsize=(5, 4))
    plt.scatter(psi, margin, s=60, alpha=0.8)
    plt.xlabel(r"$\hat{\Psi}(\theta)$")
    plt.ylabel("Margin")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(figures_path / "margin_vs_psi.pdf")
    plt.close()

    print("Графики сохранены в latex/figures/")
    print(" - accuracy_vs_psi.pdf")
    print(" - margin_vs_psi.pdf")


if __name__ == "__main__":
    main()
