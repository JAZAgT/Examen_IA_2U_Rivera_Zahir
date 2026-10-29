"""
train.py
Pregunta 4 - Ciclo de entrenamiento, metricas de rendimiento y curvas de
aprendizaje para la CNN sobre CIFAR-10.

Autor: Zahir Jadil Rivera Cruz (codigo 022101099h) - Semilla: 99
"""

# --- Importacion de librerias -----------------------------------------------
import os                                    # rutas para guardar las graficas en outputs/
import time                                  # medir cuanto tarda cada epoca

import numpy as np                           # arreglos de etiquetas reales/predichas
import torch                                 # tensores, autograd y deteccion de GPU/CPU
import torch.nn as nn                        # funcion de perdida CrossEntropyLoss
import torch.optim as optim                  # optimizador Adam
import matplotlib.pyplot as plt              # graficar curvas de loss/accuracy
import seaborn as sns                        # heatmap para la matriz de confusion
from sklearn.metrics import confusion_matrix, classification_report  # metricas de clasificacion

# Se reutilizan la carga de datos (dataset.py) y la arquitectura (model.py)
# en lugar de duplicar codigo entre los 3 archivos del repositorio.
from dataset import get_dataloaders, CLASSES, SEED, set_seed, OUTPUTS_DIR
from model import CNNCifar10

# --------------------------------------------------------------------------
# 1. CONFIGURACION DEL ENTRENAMIENTO (Pregunta 4, punto 1)
# --------------------------------------------------------------------------
set_seed(SEED)  # misma semilla individual (99) para los pesos iniciales

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
LEARNING_RATE = 1e-3


def build_training_components(model: nn.Module):
    """Define la funcion de perdida (CrossEntropyLoss) y el optimizador (Adam)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return criterion, optimizer


# --------------------------------------------------------------------------
# 2. BUCLE DE ENTRENAMIENTO POR EPOCAS (Pregunta 4, punto 2)
# --------------------------------------------------------------------------
def run_one_epoch(model, loader, criterion, optimizer=None):
    """Ejecuta una epoca completa. Si optimizer no es None, entrena
    (backprop); si es None, solo evalua (validacion/prueba).

    Esta misma funcion sirve tanto para entrenar como para evaluar: la unica
    diferencia es si se pasa un optimizer (train) o None (val/test), evitando
    tener dos copias casi identicas del bucle for.
    """
    is_training = optimizer is not None
    # model.train() activa Dropout/BatchNorm en modo entrenamiento;
    # model.eval() los desactiva para que la evaluacion sea determinista.
    model.train() if is_training else model.eval()

    running_loss, running_correct, running_total = 0.0, 0, 0

    # Desactiva el calculo de gradientes cuando solo se esta evaluando
    # (ahorra memoria y computo, ya que no habra backward() en ese caso).
    torch.set_grad_enabled(is_training)
    for images, labels in loader:
        # images llega como tensor 4D (batch_size, 3, 32, 32); se mueve al
        # mismo dispositivo (CPU o GPU) que el modelo.
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if is_training:
            # Limpia los gradientes acumulados del paso anterior antes de
            # calcular los nuevos (PyTorch los acumula por defecto).
            optimizer.zero_grad()

        outputs = model(images)             # forward pass -> logits (batch_size, 10)
        loss = criterion(outputs, labels)   # CrossEntropyLoss compara logits vs. etiqueta real

        if is_training:
            loss.backward()   # backpropagation: calcula el gradiente de cada peso
            optimizer.step()  # actualiza los pesos del modelo (paso de Adam)

        # Acumula el loss y los aciertos de este lote para promediar al final
        # de la epoca (running_loss se pondera por tamano de lote).
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)  # clase con mayor logit = prediccion del modelo
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)
    torch.set_grad_enabled(True)  # restaura el estado por defecto para la siguiente llamada

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, epochs: int = EPOCHS):
    """Bucle principal: entrena por 'epochs' epocas, registrando Loss y
    Accuracy de entrenamiento y validacion en cada una (Pregunta 4, punto 2)."""
    criterion, optimizer = build_training_components(model)
    # history guarda la evolucion de las 4 metricas, epoca por epoca, para
    # poder graficarlas despues con plot_loss_accuracy().
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # --- Bucle de entrenamiento por epocas (for epoch in range(EPOCHS)) ---
    for epoch in range(epochs):
        t0 = time.time()
        # Una "epoca" = una pasada completa por TODO el conjunto de entrenamiento,
        # seguida de una evaluacion sobre el conjunto de validacion (sin entrenar).
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer=None)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoca {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"tiempo={elapsed:.1f}s"
        )

    return history


# --------------------------------------------------------------------------
# 3. GENERACION DE ARTEFACTOS GRAFICOS (Pregunta 4, punto 3)
# --------------------------------------------------------------------------
def plot_loss_accuracy(history: dict, save_path: str):
    """Grafica y guarda las curvas de Loss y Accuracy (train/val) en una
    sola figura con dos subgraficas."""
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs_range, history["train_loss"], label="Entrenamiento")
    axes[0].plot(epochs_range, history["val_loss"], label="Validacion")
    axes[0].set_title("Curva de Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_acc"], label="Entrenamiento")
    axes[1].plot(epochs_range, history["val_acc"], label="Validacion")
    axes[1].set_title("Curva de Accuracy")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle(f"Curvas de aprendizaje - CNN CIFAR-10 (semilla={SEED})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train.py] Curvas de aprendizaje guardadas en: {save_path}")


def evaluate_on_test(model, test_loader):
    """Evalua el modelo final sobre el conjunto de prueba independiente y
    devuelve las etiquetas reales y predichas para la matriz de confusion."""
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    """Calcula y guarda la matriz de confusion del conjunto de prueba."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Matriz de confusion - Conjunto de prueba (semilla={SEED})")
    plt.xlabel("Prediccion")
    plt.ylabel("Etiqueta real")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train.py] Matriz de confusion guardada en: {save_path}")


# --------------------------------------------------------------------------
# 4. SCRIPT PRINCIPAL
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[train.py] Dispositivo de computo: {DEVICE}")

    # 1) Carga de datos: reutiliza dataset.py (Pregunta 1) para obtener los
    #    3 DataLoader ya particionados con la semilla individual (99).
    train_loader, val_loader, test_loader = get_dataloaders()

    # 2) Se instancia la arquitectura definida en model.py (Pregunta 2/3) y
    #    se envia al dispositivo de computo (CPU o GPU).
    model = CNNCifar10(num_classes=10).to(DEVICE)

    # 3) Entrenamiento: corre las 15 epocas y devuelve el historial de metricas.
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    # 4) Se guarda la grafica de Loss/Accuracy de todo el entrenamiento.
    plot_loss_accuracy(history, os.path.join(OUTPUTS_DIR, "loss_accuracy.png"))

    # 5) Evaluacion final SOLO sobre el conjunto de prueba (nunca visto antes),
    #    para medir que tan bien generaliza el modelo ya entrenado.
    y_true, y_pred = evaluate_on_test(model, test_loader)
    test_acc = (y_true == y_pred).mean()
    print(f"\n[train.py] Accuracy final sobre el conjunto de PRUEBA: {test_acc:.4f}")
    print("\n[train.py] Reporte de clasificacion (test):")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # 6) Matriz de confusion: muestra que clases se confunden entre si
    #    (usada en el analisis critico del README/informe).
    plot_confusion_matrix(y_true, y_pred, os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))
