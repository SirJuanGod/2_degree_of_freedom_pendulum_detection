"""
device_detector.py — Detecta automáticamente GPU/CPU disponibles
Soporta: CUDA (Nvidia), Metal (Apple), TPU (Google Colab)
"""
import warnings
warnings.filterwarnings('ignore')

def detect_torch_device():
    """Detecta GPU para PyTorch/YOLO."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            return "cuda:0", f"CUDA GPU: {gpu_name} (GPUs disponibles: {gpu_count})"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps", "Metal (Apple Silicon)"
        else:
            return "cpu", "CPU"
    except ImportError:
        return "cpu", "PyTorch no instalado"
    except Exception as e:
        return "cpu", f"Error en PyTorch: {e}"

def detect_jax_device():
    """Detecta GPU para JAX/Diffrax."""
    try:
        import jax
        devices = jax.devices()
        if len(devices) == 0:
            return "cpu", "Sin dispositivos disponibles"
        
        primary = devices[0]
        device_type = str(primary).split(':')[0].upper()
        
        if 'gpu' in device_type.lower():
            return "gpu", f"{device_type} ({len(devices)} dispositivo(s))"
        elif 'tpu' in device_type.lower():
            return "tpu", f"{device_type} ({len(devices)} dispositivo(s))"
        else:
            return "cpu", "CPU"
    except ImportError:
        return "cpu", "JAX no instalado"
    except Exception as e:
        return "cpu", f"Error en JAX: {e}"

def detect_mujoco_device():
    """MuJoCo usa GPU automáticamente si está disponible."""
    try:
        import mujoco
        # MuJoCo 2.4+ detecta automáticamente
        return "auto", "MuJoCo detectará GPU automáticamente"
    except ImportError:
        return "none", "MuJoCo no instalado"

def print_summary():
    """Imprime un resumen de dispositivos disponibles."""
    print("\n" + "="*60)
    print("  DEVICE DETECTION SUMMARY")
    print("="*60 + "\n")
    
    torch_dev, torch_msg = detect_torch_device()
    print(f"PyTorch/YOLO   : {torch_msg}")
    print(f"  → Usar device='{torch_dev}'")
    
    jax_dev, jax_msg = detect_jax_device()
    print(f"\nJAX/Diffrax    : {jax_msg}")
    print(f"  → Usar device='{jax_dev}'")
    
    mj_dev, mj_msg = detect_mujoco_device()
    print(f"\nMuJoCo         : {mj_msg}")
    
    print("\n" + "="*60)
    print("RECOMENDACIONES:")
    print("="*60 + "\n")
    
    if torch_dev != "cpu":
        print(f"✓ GPU disponible para YOLO: {torch_msg}")
        print("  → Ejecuta: python run_pipeline.py --video mi_video.mp4")
    else:
        print("⚠ GPU NO disponible para YOLO → ejecutará en CPU")
        print("  Si tienes GPU instalada, revisa drivers CUDA")
    
    if jax_dev != "cpu":
        print(f"\n✓ GPU disponible para JAX: {jax_msg}")
        print("  → Identificación será muy rápida (~5-10 min)")
    else:
        print("\n⚠ GPU NO disponible para JAX → ejecutará en CPU")
        print("  → Identificación tardará ~30-60 min")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print_summary()
