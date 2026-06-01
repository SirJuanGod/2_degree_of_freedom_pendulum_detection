# 🔬 Pipeline de Detección de Péndulo Doble Optimizado para GPU/CPU

**Estado**: Código revisado, optimizado y ejecutable ✅

## 📋 Descripción General

Este pipeline automatizado realiza:
1. **Tracking con YOLO + HSV** - Detección de puntos del péndulo en video
2. **Cálculo de ángulos** - Extracción de θ₁ y θ₂ desde trayectorias
3. **Identificación de parámetros** - Multi-shooting con JAX + Diffrax
4. **Análisis de estabilidad** - Linealización y eigenvalores
5. **Generación de reporte** - PDF con modelo matemático

## 🚀 Inicio Rápido

### 1️⃣ Detectar Dispositivos Disponibles

```bash
python utils/device_detector.py
```

**Salida esperada:**
```
PyTorch/YOLO   : CUDA GPU: NVIDIA GeForce RTX 4090 (GPUs disponibles: 1)
JAX/Diffrax    : GPU (1 dispositivo(s))
```

### 2️⃣ Instalar Dependencias

**Para GPU (NVIDIA CUDA 12.x):**
```bash
pip install -r requirements.txt
pip install jax[cuda12_cudnn82]  # Instala JAX para GPU
```

**Para CPU:**
```bash
pip install -r requirements.txt
```

**Para Apple Silicon (Metal):**
```bash
pip install -r requirements.txt
# JAX detectará Metal automáticamente
```

### 3️⃣ Ejecutar Pipeline

```bash
python run_pipeline.py --video mi_video.mp4
```

**Tiempo estimado:**
- Con GPU: **~15-20 minutos**
- Con CPU: **~2-3 horas**

## 📂 Estructura del Proyecto

```
├── run_pipeline.py              # Punto de entrada principal
├── requirements.txt             # Dependencias Python
├── utils/
│   ├── device_detector.py       # Detecta GPU/CPU disponibles
│   └── logging_setup.py         # Configuración de logging
├── src/
│   ├── 00_calibrate_hsv.py      # Calibración de colores HSV
│   ├── 01_tracker.py            # YOLO tracking + HSV ✅ OPTIMIZADO
│   ├── 02_angles.py             # Cálculo de ángulos
│   ├── 03_identification.py     # Identificación JAX ✅ OPTIMIZADO
│   ├── 04_linearization.py      # Análisis de estabilidad ✅ OPTIMIZADO
│   └── 05_report.py             # Reporte PDF ✅ OPTIMIZADO
├── config/
│   └── colors.yaml              # Parámetros HSV de colores
├── model/
│   ├── pendulo.xml              # Modelo MuJoCo
│   └── scene.xml                # Escena MuJoCo
└── output/
    ├── video_tracked.mp4        # Video con tracking anotado
    ├── trajectories.npz         # Trayectorias extraídas
    ├── angles.npz               # Ángulos calculados
    ├── identified_params.yaml    # Parámetros identificados
    ├── linear_models.yaml       # Modelos lineales
    ├── pendulo_modelo.mat       # Modelo MATLAB
    ├── plot_*.png               # Gráficas
    └── pipeline.log             # Log detallado
```

## 🔧 Opciones Avanzadas

### Ejecutar Pasos Individuales

```bash
# Solo tracking
python src/01_tracker.py --video mi_video.mp4

# Solo ángulos (requiere trajectories.npz)
python src/02_angles.py

# Solo identificación (requiere angles.npz)
python src/03_identification.py

# Solo linealización (requiere identified_params.yaml)
python src/04_linearization.py

# Solo reporte
python src/05_report.py
```

### Omitir Pasos

```bash
# Omitir identificación (solo tracking + ángulos)
python run_pipeline.py --video mi_video.mp4 --skip-id

# Omitir linealización
python run_pipeline.py --video mi_video.mp4 --skip-lin
```

### Calibración de Colores HSV

```bash
python src/00_calibrate_hsv.py --video mi_video.mp4 --frame 100
```

Ajusta los sliders para detectar los 3 puntos del péndulo. Los valores se guardan en `config/colors.yaml`.

## 📊 Archivos de Salida

| Archivo | Descripción |
|---------|-----------|
| `video_tracked.mp4` | Video original con puntos detectados |
| `trajectories.npz` | Trayectorias (t, x, y) de los 3 puntos |
| `angles.npz` | Ángulos θ₁, θ₂ y velocidades angulares |
| `identified_params.yaml` | Parámetros identificados: L₁, L₂, m₁, m₂, b₁, b₂ |
| `linear_models.yaml` | Análisis de estabilidad en 4 equilibrios |
| `pendulo_modelo.mat` | Matriz A para MATLAB/Simulink |
| `plot_angles.png` | Gráfica de ángulos vs tiempo |
| `plot_eigenvalues.png` | Mapa de polos (4 equilibrios) |
| `plot_free_response.png` | Respuesta libre lineal |
| `pipeline.log` | Log detallado de ejecución |

## 🐛 Solución de Problemas

### ❌ Error: "CUDA out of memory"
- **Causa**: GPU sin suficiente VRAM
- **Solución**: Ejecutar en CPU: `jax.config.update("jax_platform_name", "cpu")`

### ❌ Error: "angles.npz no encontrado"
- **Causa**: Primer paso (tracking) falló
- **Solución**: Revisar `pipeline.log` y volver a ejecutar con `--video`

### ❌ Error: "No se detectaron 3 colores"
- **Causa**: HSV mal calibrado
- **Solución**: Ejecutar `00_calibrate_hsv.py` y ajustar ranges

### ❌ Ejecución muy lenta en CPU
- **Esperado**: Identificación JAX tarda ~30-60min en CPU
- **Solución**: Instalar JAX con GPU: `pip install jax[cuda12_cudnn82]`

### ⚠️ Advertencia: "reportlab no instalado"
- **Solución**: `pip install reportlab` (opcional, afecta solo PDF)

## 📈 Interpretación de Resultados

### `identified_params.yaml`
```yaml
L1: 0.2500  # Longitud barra 1 (metros)
L2: 0.2500  # Longitud barra 2 (metros)
m1: 0.0500  # Masa eslabón 1 (kg)
m2: 0.0250  # Masa eslabón 2 (kg)
b1: 0.0010  # Damping eslabón 1 (N·m·s/rad)
b2: 0.0010  # Damping eslabón 2 (N·m·s/rad)
g:  9.8100  # Gravedad (m/s²)
cost_final: 1.234e-4  # Error MSE final
optimizer_success: true
```

### `linear_models.yaml`
- **P1_colgante** (θ₁=0°, θ₂=0°): Equilibrio estable
- **P2_semi_inv_b1** (θ₁=180°, θ₂=0°): Equilibrio inestable
- **P3_semi_inv_b2** (θ₁=0°, θ₂=180°): Equilibrio inestable
- **P4_invertido** (θ₁=180°, θ₂=180°): Equilibrio inestable

## 🎯 Optimizaciones Implementadas

| Componente | Problema Original | Solución Implementada |
|-----------|------------------|----------------------|
| **Tracker** | Sin detección GPU | Detección automática + warmup YOLO |
| **JAX** | Sin fallback CPU | Detección de dispositivo + logging |
| **Identificación** | Compilación JIT lenta | Caching + validación de resultados |
| **Linealización** | Imports opcionales faltantes | Fallback con warnings |
| **Reporte** | PDF opcional no manejado | Fallback a texto si no hay reportlab |
| **Pipeline** | Sin logging detallado | Logger completo a archivo y consola |
| **Requirements** | Dependencias incompletas | Especificación completa de versiones |

## 🔐 Validaciones Incorporadas

✅ Verificación de archivo de video  
✅ Detección de GPU/CPU automática  
✅ Validación de parámetros identificados (rangos razonables)  
✅ Detección de NaN e Inf en resultados  
✅ Cobertura de detección (warning si <50%)  
✅ Logging detallado en archivo y consola  
✅ Manejo de excepciones en cada paso  
✅ Fallback para módulos opcionales  

## 📝 Notas de Desarrollo

- **Lenguaje**: Python 3.8+
- **Framework GPU**: JAX + Diffrax (NVIDIA CUDA 12.x, Apple Metal, TPU)
- **Detector**: Ultralytics YOLOv8n
- **ODE Solver**: Diffrax (compatible con JAX)
- **Optimizadores**: Adam (Optax) + Nelder-Mead (SciPy)

## 📚 Referencias

- YOLOv8: https://docs.ultralytics.com
- JAX: https://jax.readthedocs.io
- Diffrax: https://docs.kidger.site/diffrax
- Python Control: https://python-control.readthedocs.io

---

**Última actualización**: Junio 2026  
**Estado de GPU**: ✅ Completamente optimizado para NVIDIA/Apple/TPU
