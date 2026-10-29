# Examen IA - Segunda Unidad: Clasificación de Imágenes mediante CNN, Flujo de Tensores 4D y Análisis de Métricas

**Estudiante:** Zahir Jadil Rivera Cruz
**Código de matrícula:** 022101099h
**Semilla individual:** 99 (dos últimos dígitos del código de matrícula)
**Universidad Andina del Cusco — Departamento Académico de Ingeniería de Sistemas**

---

## 1. Descripción del proyecto

Este repositorio contiene el desarrollo completo de la evaluación de la Segunda Unidad del curso de Inteligencia Artificial: la clasificación de imágenes del dataset **CIFAR-10** mediante una **Red Neuronal Convolucional (CNN)** implementada en PyTorch, documentando el flujo de tensores 4D (`Batch_Size, Channels, Height, Width`) y el análisis crítico de las métricas de entrenamiento.

## 2. Estructura del repositorio

```
Examen_IA_2U_Rivera_Zahir/
├── src/
│   ├── dataset.py          # Carga, transformaciones, partición y visualización (Pregunta 1)
│   ├── model.py             # Arquitectura CNN y validación de tensores 4D (Pregunta 3)
│   └── train.py              # Entrenamiento, métricas y gráficas (Pregunta 4)
├── notebooks/
│   └── experiment.ipynb    # EDA, pruebas y visualización de muestras
├── outputs/
│   ├── sample_grid.png      # Cuadrícula de muestras del dataset
│   ├── loss_accuracy.png   # Curvas de aprendizaje (Loss/Accuracy)
│   └── confusion_matrix.png # Matriz de confusión del conjunto de prueba
└── README.md
```

## 3. Reproducibilidad (semilla individual)

Todo el pipeline usa la semilla **99**, fijada en `src/dataset.py` mediante:

```python
SEED = 99
torch.manual_seed(SEED)
np.random.seed(SEED)
```

Esta semilla determina de forma única el *shuffle* de los datos, la partición train/validación (`torch.Generator().manual_seed(99)` en `random_split`) y la inicialización aleatoria de los pesos de la red.

## 4. Pregunta 2: Justificación teórica — CNN frente a MLP

### 4.1 Por qué un Perceptrón Multicapa (MLP) sería ineficiente para CIFAR-10

CIFAR-10 contiene imágenes a color de 32×32 píxeles con 3 canales RGB, lo que equivale a **3,072 valores de entrada por imagen** (32 × 32 × 3). Un MLP tradicional "aplanaría" cada imagen en un vector de 3,072 dimensiones, descartando por completo la **estructura espacial bidimensional** de los píxeles: dos píxeles vecinos (por ejemplo, los bordes de un ojo en la foto de un gato) se tratarían como entradas independientes, sin relación posicional explícita.

Esto genera tres problemas concretos:

1. **Explosión de parámetros.** Si la primera capa oculta de un MLP tuviera, por ejemplo, 512 neuronas totalmente conectadas a los 3,072 píxeles de entrada, se necesitarían `3072 × 512 ≈ 1.57 millones` de parámetros solo en esa capa. Al apilar varias capas densas, el modelo colapsa en tamaño y se vuelve extremadamente propenso al **overfitting**, dado que el dataset de entrenamiento (50,000 imágenes) no alcanza a "cubrir" un espacio de parámetros tan grande.
2. **Falta de invariancia espacial.** Un MLP aprende pesos distintos para cada posición del píxel. Si un objeto (por ejemplo, un automóvil) aparece desplazado unos píxeles hacia la derecha en una imagen de prueba, el MLP lo interpretaría como un patrón completamente diferente, porque no existe ningún mecanismo que reconozca el mismo patrón visual independientemente de su ubicación.
3. **Ignora la correlación local de los píxeles.** La información relevante de una imagen (bordes, texturas, formas) está codificada en relaciones **locales** entre píxeles vecinos. Un MLP conecta cada entrada con cada neurona sin ningún sesgo hacia la vecindad espacial, por lo que debe "reaprender" desde cero relaciones que, en una imagen, son evidentes por construcción.

### 4.2 Cómo las propiedades de las CNN resuelven esta limitación

Las Redes Neuronales Convolucionales incorporan tres propiedades estructurales que resuelven directamente las limitaciones anteriores:

- **Conexiones locales (campos receptivos).** Cada neurona convolucional solo se conecta a una pequeña región de la imagen (por ejemplo, un kernel de 3×3), en lugar de a todos los píxeles. Esto reduce drásticamente el número de parámetros y obliga al modelo a aprender primero patrones locales simples (bordes, esquinas) que luego se combinan en capas más profundas para representar formas complejas.
- **Weight Sharing (parámetros compartidos).** El mismo kernel (filtro) se desliza sobre toda la imagen, reutilizando los mismos pesos en cada posición espacial. Esto reduce aún más el número de parámetros entrenables y permite que la red **detecte el mismo patrón visual sin importar en qué parte de la imagen aparezca**.
- **Invariancia espacial (gracias al pooling).** Las capas de `MaxPool2d` reducen la resolución espacial conservando las activaciones más fuertes, lo que hace que la red sea más robusta ante pequeños desplazamientos, rotaciones leves o cambios de escala del objeto dentro de la imagen.

En conjunto, estas propiedades permiten que una CNN aprenda **jerarquías de características** (bordes → texturas → partes → objetos completos) con muchos menos parámetros que un MLP equivalente, generalizando mejor sobre imágenes nuevas y siendo computacionalmente más eficiente para datos con estructura espacial como CIFAR-10.

## 5. Pregunta 2: Especificación de la arquitectura CNN

La arquitectura implementada en `src/model.py` (clase `CNNCifar10`) está compuesta por **tres bloques convolucionales** y un clasificador totalmente conectado:

| Capa | Operación | Tensor de salida (C, H, W) |
|---|---|---|
| Entrada | Imagen RGB | (3, 32, 32) |
| Bloque 1 | `Conv2d(3→32, k=3, pad=1)` → `ReLU` → `MaxPool2d(2)` | (32, 16, 16) |
| Bloque 2 | `Conv2d(32→64, k=3, pad=1)` → `ReLU` → `MaxPool2d(2)` | (64, 8, 8) |
| Bloque 3 | `Conv2d(64→128, k=3, pad=1)` → `ReLU` → `MaxPool2d(2)` | (128, 4, 4) |
| Flatten | `nn.Flatten()` | (2048,) |
| FC1 | `Linear(2048 → 256)` → `ReLU` → `Dropout(0.5)` | (256,) |
| FC2 (salida) | `Linear(256 → 10)` | (10,) — logits por clase |

**Diagrama del flujo de tensores:**

```
Input (B,3,32,32)
   │
   ▼
[Conv2d 3→32 + ReLU + MaxPool2d]  →  (B,32,16,16)
   │
   ▼
[Conv2d 32→64 + ReLU + MaxPool2d] →  (B,64,8,8)
   │
   ▼
[Conv2d 64→128 + ReLU + MaxPool2d] → (B,128,4,4)
   │
   ▼
Flatten → (B, 2048)
   │
   ▼
Linear(2048→256) + ReLU + Dropout(0.5)
   │
   ▼
Linear(256→10) → logits (B,10)
```

El `Dropout(0.5)` se añade antes de la capa de salida como técnica de regularización adicional para mitigar el overfitting, dado el tamaño relativamente pequeño de las imágenes de CIFAR-10 frente a la capacidad del modelo.

## 6. Pregunta 3: Validación del flujo de tensores 4D

El archivo `src/model.py` incluye la función `validate_forward_pass()`, que construye un tensor simulado de forma `(Batch_Size=8, Channels=3, Height=32, Width=32)` y lo pasa por el modelo completo, imprimiendo cada shape intermedio y verificando con una aserción (`assert`) que la salida final tenga la forma `(8, 10)`. Esto garantiza, antes de iniciar el entrenamiento real, que no exista un *Shape Mismatch* entre el último bloque convolucional y la primera capa densa.

Salida de consola obtenida al ejecutar `python src/model.py`:

```
[model.py] Validando forward pass con batch simulado (N=8)...
  entrada           : (8, 3, 32, 32)
  despues block1    : (8, 32, 16, 16)
  despues block2    : (8, 64, 8, 8)
  despues block3    : (8, 128, 4, 4)
  despues flatten   : (8, 2048)
  salida (logits)   : (8, 10)
[model.py] OK: el tensor 4D fluyo correctamente por toda la arquitectura.
```

## 7. Pregunta 4: Entrenamiento, métricas y análisis crítico

### 7.1 Configuración

- **Función de pérdida:** `CrossEntropyLoss` (adecuada para clasificación multiclase con logits).
- **Optimizador:** `Adam`, tasa de aprendizaje `1e-3`.
- **Épocas:** 15.
- **Batch size:** 64.
- **Partición:** 80% entrenamiento / 20% validación (sobre las 50,000 imágenes de entrenamiento), más 10,000 imágenes de prueba independientes.
- **Data Augmentation:** `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` solo en el conjunto de entrenamiento.

### 7.2 Resultados obtenidos (semilla 99)

Entrenamiento completo de 15 épocas ejecutado con `python src/train.py`:

| Época | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 1.6802 | 0.3825 | 1.3795 | 0.5050 |
| 5 | 0.9812 | 0.6555 | 0.8203 | 0.7185 |
| 10 | 0.8036 | 0.7199 | 0.6983 | 0.7590 |
| 15 | 0.7181 | 0.7531 | 0.6572 | 0.7771 |

**Accuracy final sobre el conjunto de PRUEBA (10,000 imágenes independientes): 0.7726 (77.26%)**

Reporte de clasificación por clase (conjunto de prueba):

```
              precision    recall  f1-score   support

       avion       0.72      0.85      0.78      1000
   automovil       0.90      0.89      0.90      1000
      pajaro       0.73      0.64      0.68      1000
        gato       0.65      0.54      0.59      1000
      venado       0.71      0.78      0.75      1000
       perro       0.72      0.67      0.69      1000
        rana       0.82      0.84      0.83      1000
     caballo       0.77      0.83      0.80      1000
       barco       0.93      0.78      0.85      1000
      camion       0.77      0.90      0.83      1000

    accuracy                           0.77     10000
   macro avg       0.77      0.77      0.77     10000
weighted avg       0.77      0.77      0.77     10000
```

Ver `outputs/loss_accuracy.png` (curvas de Loss/Accuracy) y `outputs/confusion_matrix.png` (matriz de confusión del conjunto de prueba).

### 7.3 Análisis crítico (overfitting / underfitting)

Con la semilla individual 99, el modelo **no muestra overfitting severo**. La evidencia cuantitativa es la siguiente:

- **Brecha train/val controlada:** al finalizar la época 15, el *train_acc* es 75.31% y el *val_acc* es 77.71%. La diferencia entre ambos es inferior a 3 puntos porcentuales, e incluso el accuracy de validación es ligeramente **superior** al de entrenamiento. Esto es un indicio claro de que el Data Augmentation (RandomCrop + RandomHorizontalFlip) está regularizando correctamente el entrenamiento: al dificultar artificialmente el conjunto de entrenamiento, el modelo generaliza mejor de lo que "memoriza".
- **Convergencia estable del Loss:** tanto el *train_loss* (1.68 → 0.72) como el *val_loss* (1.38 → 0.66) descienden de forma consistente y conjunta a lo largo de las 15 épocas, sin que el *val_loss* vuelva a subir en ninguna época tardía (lo cual sería la señal clásica de overfitting). Ver `outputs/loss_accuracy.png`.
- **Accuracy de prueba consistente con validación:** el accuracy final sobre el conjunto de prueba independiente (77.26%) es prácticamente idéntico al accuracy de validación de la última época (77.71%), confirmando que el modelo generaliza de manera consistente a datos nunca vistos durante el entrenamiento ni la selección de hiperparámetros.
- **Posible underfitting leve:** por otro lado, el accuracy se sigue mostrando una tendencia ascendente hacia la época 15 sin haberse estabilizado del todo (ver pendiente positiva en `loss_accuracy.png`), lo que sugiere que el modelo **podría beneficiarse de más épocas de entrenamiento** o de una arquitectura con mayor capacidad (más filtros o un bloque convolucional adicional) para superar el 80% de accuracy, en lugar de mostrar señales de sobreajuste.
- **Matriz de confusión:** las clases con mayor confusión mutua son *gato* (recall 0.54) y *perro* (recall 0.67), consistente con la alta similitud visual entre ambas categorías en imágenes de baja resolución (32×32 px). En contraste, clases con formas y colores más distintivos como *automovil* (recall 0.89) y *barco* (precision 0.93) obtienen los mejores resultados.

**Conclusión:** con la semilla 99, la combinación de Data Augmentation y Dropout(0.5) logró un entrenamiento bien regularizado, sin evidencia de overfitting; el margen de mejora más probable está en aumentar la capacidad/épocas del modelo (posible underfitting leve) antes que en reducir la complejidad de la red.

## 8. Cómo ejecutar el proyecto

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# 2. Instalar dependencias
pip install torch torchvision matplotlib scikit-learn seaborn jupyter

# 3. Validar la arquitectura (Pregunta 3)
python src/model.py

# 4. Entrenar el modelo y generar las gráficas (Pregunta 4)
python src/train.py

# 5. (Opcional) Explorar el notebook de EDA y pruebas
jupyter notebook notebooks/experiment.ipynb
```

## 9. Enlace del repositorio

[https://github.com/JAZAgT/Examen_IA_2U_Rivera_Zahir](https://github.com/JAZAgT/Examen_IA_2U_Rivera_Zahir)
