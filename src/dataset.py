"""
dataset.py
Pregunta 1 - Carga, Visualizacion, Transformaciones y Particion del dataset CIFAR-10.

Autor: Zahir Jadil Rivera Cruz (codigo 022101099h)
Semilla individual: 99 (dos ultimos digitos del codigo de matricula)
"""

# --- Importacion de librerias -----------------------------------------------
import os                                            # manejo de rutas de archivos
import numpy as np                                   # semilla de numpy y operaciones numericas
import torch                                         # tensores y semilla de PyTorch
from torch.utils.data import DataLoader, random_split  # carga por lotes y particion train/val
from torchvision import datasets, transforms         # descarga de CIFAR-10 y transformaciones de imagen
import matplotlib.pyplot as plt                      # graficar la cuadricula de muestras

# --------------------------------------------------------------------------
# 1. SEMILLA INDIVIDUAL (reproducibilidad de shuffle, split y pesos iniciales)
# --------------------------------------------------------------------------
SEED = 99


def set_seed(seed: int = SEED) -> None:
    """Fija la semilla en todas las fuentes de aleatoriedad usadas por el pipeline."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(SEED)

# --------------------------------------------------------------------------
# 2. RUTAS Y CONSTANTES
# --------------------------------------------------------------------------
# Carpeta raiz del proyecto (un nivel arriba de src/), usada para guardar el
# dataset descargado y las imagenes generadas.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CLASSES = (
    "avion", "automovil", "pajaro", "gato", "venado",
    "perro", "rana", "caballo", "barco", "camion",
)

# Media y desviacion estandar estandar de CIFAR-10 por canal RGB.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

BATCH_SIZE = 64
VAL_SPLIT = 0.2  # 20% del conjunto de entrenamiento se reserva para validacion


# --------------------------------------------------------------------------
# 3. PIPELINE DE TRANSFORMACIONES (punto 3 de la Pregunta 1)
# --------------------------------------------------------------------------
# Entrenamiento: incluye Data Augmentation (RandomCrop + RandomHorizontalFlip)
# para mejorar la generalizacion del modelo.
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# Validacion/Prueba: sin augmentation, solo conversion a tensor y normalizacion,
# para evaluar siempre sobre la imagen original.
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


# --------------------------------------------------------------------------
# 4. CARGA DEL DATASET Y PARTICION (puntos 1 y 4 de la Pregunta 1)
# --------------------------------------------------------------------------
def get_dataloaders(batch_size: int = BATCH_SIZE, val_split: float = VAL_SPLIT):
    """Descarga CIFAR-10, aplica el split train/val con la semilla individual
    y devuelve los DataLoader de entrenamiento, validacion y prueba.

    Parametros:
        batch_size: cuantas imagenes trae cada lote (Batch_Size del tensor 4D).
        val_split:  fraccion del conjunto de entrenamiento reservada para validacion (0.2 = 20%).

    Retorna:
        (train_loader, val_loader, test_loader): los 3 DataLoader listos para el bucle de entrenamiento.
    """

    # full_train_raw: las 50000 imagenes de entrenamiento SIN transform todavia
    # (se cargan como PIL.Image); se les aplica el transform despues del split,
    # en _DatasetWithTransform, para poder usar augmentation solo en train.
    full_train_raw = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    # test_dataset: las 10000 imagenes de prueba, siempre con eval_transform
    # (nunca se usan para entrenar ni para elegir hiperparametros).
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=eval_transform)

    n_total = len(full_train_raw)
    n_val = int(n_total * val_split)   # p.ej. 0.2 * 50000 = 10000 imagenes de validacion
    n_train = n_total - n_val          # el resto (40000) queda para entrenamiento

    # random_split usa un generador propio sembrado con SEED: la particion es
    # reproducible y unica para esta semilla individual (99). Sin este generador,
    # cada ejecucion del script generaria una partición distinta.
    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(full_train_raw, [n_train, n_val], generator=generator)

    # Cada subconjunto recibe su propio transform: augmentation (RandomCrop +
    # RandomHorizontalFlip) solo en train_dataset; val_dataset usa eval_transform
    # para medir el desempeño siempre sobre la imagen "limpia", sin distorsionar.
    train_dataset = _DatasetWithTransform(full_train_raw, train_subset.indices, train_transform)
    val_dataset = _DatasetWithTransform(full_train_raw, val_subset.indices, eval_transform)

    # DataLoader agrupa las imagenes en lotes (batches) y entrega tensores 4D
    # de forma (batch_size, 3, 32, 32) a cada iteracion del bucle de entrenamiento.
    # shuffle=True solo en train: en val/test el orden no afecta la evaluacion.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"[dataset.py] Tamano total de entrenamiento original: {n_total}")
    print(f"[dataset.py] Particion -> train: {len(train_dataset)} | val: {len(val_dataset)} | test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


class _DatasetWithTransform(torch.utils.data.Dataset):
    """Envuelve un subconjunto de indices del dataset crudo (PIL) aplicando
    el transform correspondiente (con o sin augmentation).

    Es necesaria porque random_split() devuelve dos Subset que comparten el
    MISMO dataset base; sin esta clase no habria forma de aplicar augmentation
    unicamente al subconjunto de entrenamiento y no al de validacion.
    """

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset  # dataset CIFAR-10 completo (sin transform)
        self.indices = indices            # indices de ESTE subconjunto (train o val)
        self.transform = transform        # pipeline a aplicar (train_transform o eval_transform)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # idx recorre 0..len(self.indices)-1; se traduce al indice real del
        # dataset completo antes de pedir la imagen (todavia en formato PIL).
        image, label = self.base_dataset[self.indices[idx]]
        # Aqui se aplica el transform: convierte la imagen PIL a tensor
        # normalizado (y le agrega augmentation si corresponde).
        return self.transform(image), label


# --------------------------------------------------------------------------
# 5. VISUALIZACION DE MUESTRAS (punto 2 de la Pregunta 1)
# --------------------------------------------------------------------------
def denormalize(tensor_img):
    """Revierte la normalizacion para poder graficar la imagen con colores reales."""
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return tensor_img * std + mean


def show_sample_grid(loader, n_images: int = 12, save_path: str = None, show: bool = True):
    """Muestra (y opcionalmente guarda) una cuadricula con n_images imagenes
    de ejemplo del loader, indicando la clase real de cada una."""
    images, labels = next(iter(loader))
    images, labels = images[:n_images], labels[:n_images]

    cols = 4
    rows = int(np.ceil(n_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n_images:
            img = denormalize(images[i]).permute(1, 2, 0).clamp(0, 1).numpy()
            axes[i].imshow(img)
            axes[i].set_title(CLASSES[labels[i]], fontsize=10)

    fig.suptitle(f"Muestras del dataset CIFAR-10 (semilla={SEED})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[dataset.py] Cuadricula de muestras guardada en: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    # show=False: al ejecutarse como script de consola no se abre ventana
    # grafica interactiva, solo se guarda el archivo PNG en outputs/.
    show_sample_grid(train_loader, n_images=12,
                      save_path=os.path.join(OUTPUTS_DIR, "sample_grid.png"),
                      show=False)
