"""
model.py
Pregunta 2 (especificacion) y Pregunta 3 - Arquitectura CNN y validacion de
tensores 4D (Batch_Size, Channels, Height, Width).

Autor: Zahir Jadil Rivera Cruz (codigo 022101099h) - Semilla: 99
"""

import torch
import torch.nn as nn


class CNNCifar10(nn.Module):
    """CNN para clasificacion de CIFAR-10 (10 clases, imagenes 3x32x32).

    Arquitectura (Pregunta 2, punto 2):
        Bloque 1: Conv2d(3->32) + ReLU + MaxPool2d(2)   -> 32x16x16
        Bloque 2: Conv2d(32->64) + ReLU + MaxPool2d(2)  -> 64x8x8
        Bloque 3: Conv2d(64->128) + ReLU + MaxPool2d(2) -> 128x4x4
        Flatten -> Linear(128*4*4 -> 256) -> ReLU -> Dropout
                -> Linear(256 -> 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- Bloque convolucional 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
        )

        # --- Bloque convolucional 2 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
        )

        # --- Bloque convolucional 3 (profundiza la jerarquia de features) ---
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )

        # Aplanado dinamico: no se asume un numero fijo, se calcula con
        # nn.Flatten() y se usa un valor de entrada derivado del tamano real.
        self.flatten = nn.Flatten()

        # Tamano de entrada a la capa densa calculado para 128 canales x 4 x 4
        self._flat_features = 128 * 4 * 4

        self.classifier = nn.Sequential(
            nn.Linear(self._flat_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """Forward pass. Si verbose=True, imprime el shape del tensor 4D en
        cada etapa (usado en la validacion de la Pregunta 3)."""
        if verbose:
            print(f"  entrada           : {tuple(x.shape)}")

        x = self.block1(x)
        if verbose:
            print(f"  despues block1    : {tuple(x.shape)}")

        x = self.block2(x)
        if verbose:
            print(f"  despues block2    : {tuple(x.shape)}")

        x = self.block3(x)
        if verbose:
            print(f"  despues block3    : {tuple(x.shape)}")

        x = self.flatten(x)
        if verbose:
            print(f"  despues flatten   : {tuple(x.shape)}")

        x = self.classifier(x)
        if verbose:
            print(f"  salida (logits)   : {tuple(x.shape)}")

        return x


# ----------------------------------------------------------------------------
# Pregunta 3, punto 3: Validacion de Shapes (Aserccion) con un batch simulado.
# ----------------------------------------------------------------------------
def validate_forward_pass(batch_size: int = 8, num_classes: int = 10):
    """Crea un tensor 4D simulado (Batch_Size, Channels, Height, Width) y lo
    pasa por el modelo, imprimiendo los shapes intermedios y verificando con
    assert que la salida final tiene la forma esperada."""
    print(f"\n[model.py] Validando forward pass con batch simulado (N={batch_size})...")

    model = CNNCifar10(num_classes=num_classes)
    model.eval()

    # Tensor 4D: (Batch_Size=8, Channels=3, Height=32, Width=32)
    dummy_input = torch.randn(batch_size, 3, 32, 32)

    with torch.no_grad():
        output = model(dummy_input, verbose=True)

    assert output.shape == (batch_size, num_classes), (
        f"Shape Mismatch: se esperaba ({batch_size}, {num_classes}) y se obtuvo {tuple(output.shape)}"
    )
    print("[model.py] OK: el tensor 4D fluyo correctamente por toda la arquitectura.\n")
    return model


if __name__ == "__main__":
    validate_forward_pass(batch_size=8)
