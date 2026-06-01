"""
Pipeline completo — un solo comando.
USO: python run_pipeline.py --video pendulo.mp4

OPTIMIZACIONES:
- Detección automática GPU/CPU
- Mejor manejo de errores en cada paso
- Logging detallado
"""
import argparse, importlib.util, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Importar utilidades
sys.path.insert(0, str(ROOT))
from utils import setup_logging, print_summary

# Configurar logging global
logger = setup_logging("pipeline", ROOT/"output")

def load(name, path):
    """Carga módulo dinámicamente."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def step(n, title):
    """Imprime encabezado de paso."""
    print(f"\n{'─'*70}")
    print(f"  PASO {n}/5 — {title}")
    print(f"{'─'*70}")
    logger.info(f"PASO {n}/5: {title}")

if __name__ == "__main__":
    print_summary()
    
    p = argparse.ArgumentParser(
        description="Pipeline completo de detección de péndulo doble con GPU/CPU"
    )
    p.add_argument("--video",    required=True, help="Ruta al video (ej: pendulo.mp4)")
    p.add_argument("--skip-id",  action="store_true", help="Omitir identificación")
    p.add_argument("--skip-lin", action="store_true", help="Omitir linealización")
    args = p.parse_args()

    # Validar video
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"❌ Video no encontrado: {video_path}")
        sys.exit(1)
    
    logger.info(f"Video: {video_path}")
    logger.info(f"Skip ID: {args.skip_id}, Skip LIN: {args.skip_lin}\n")

    src  = ROOT/"src"
    
    try:
        trk  = load("trk", src/"01_tracker.py")
        ang  = load("ang", src/"02_angles.py")
        idn  = load("idn", src/"03_identification.py")
        lin  = load("lin", src/"04_linearization.py")
        rep  = load("rep", src/"05_report.py")
    except Exception as e:
        logger.error(f"❌ Error cargando módulos: {e}")
        sys.exit(1)

    t0 = time.time()
    failed_steps = []

    # ─────────────────────────────────────────────────────
    # PASO 1: YOLO TRACKING
    # ─────────────────────────────────────────────────────
    step(1, "YOLO TRACKING + FILTRO HSV")
    try:
        trk.run(args.video)
        logger.info("✓ PASO 1 completado")
    except Exception as e:
        logger.error(f"✗ Error en PASO 1: {e}")
        failed_steps.append(("Tracking", e))
        sys.exit(1)

    # ─────────────────────────────────────────────────────
    # PASO 2: CÁLCULO DE ÁNGULOS
    # ─────────────────────────────────────────────────────
    step(2, "CÁLCULO DE ÁNGULOS")
    try:
        ang.run()
        logger.info("✓ PASO 2 completado")
    except Exception as e:
        logger.error(f"✗ Error en PASO 2: {e}")
        failed_steps.append(("Ángulos", e))
        sys.exit(1)

    # ─────────────────────────────────────────────────────
    # PASO 3: IDENTIFICACIÓN (Opcional)
    # ─────────────────────────────────────────────────────
    if not args.skip_id:
        step(3, "IDENTIFICACIÓN DE PARÁMETROS (JAX/GPU)")
        try:
            idn.run()
            logger.info("✓ PASO 3 completado")
        except Exception as e:
            logger.error(f"✗ Error en PASO 3: {e}")
            failed_steps.append(("Identificación", e))
            if args.skip_lin:
                logger.warning("  Omitiendo pasos 3 y 4")
            else:
                sys.exit(1)
    else:
        logger.info("⊘ PASO 3 omitido (--skip-id)")

    # ─────────────────────────────────────────────────────
    # PASO 4: LINEALIZACIÓN (Opcional, requiere PASO 3)
    # ─────────────────────────────────────────────────────
    if not args.skip_lin and not args.skip_id:
        step(4, "LINEALIZACIÓN + ESTABILIDAD")
        try:
            lin.run()
            logger.info("✓ PASO 4 completado")
        except Exception as e:
            logger.error(f"✗ Error en PASO 4: {e}")
            failed_steps.append(("Linealización", e))
    elif args.skip_lin:
        logger.info("⊘ PASO 4 omitido (--skip-lin)")
    else:
        logger.info("⊘ PASO 4 omitido (requiere PASO 3)")

    # ─────────────────────────────────────────────────────
    # PASO 5: REPORTE FINAL
    # ─────────────────────────────────────────────────────
    step(5, "GENERACIÓN DE REPORTE")
    try:
        rep.run()
        logger.info("✓ PASO 5 completado")
    except Exception as e:
        logger.error(f"⚠  Error en PASO 5 (no crítico): {e}")
        failed_steps.append(("Reporte", e))

    # ─────────────────────────────────────────────────────
    # RESUMEN FINAL
    # ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    
    print(f"\n{'═'*70}")
    if not failed_steps:
        print(f"✓ Pipeline completado EXITOSAMENTE en {elapsed/60:.1f} minutos")
        print(f"{'═'*70}\n")
        logger.info(f"✓ Pipeline completado en {elapsed:.1f}s")
    else:
        print(f"⚠  Pipeline completado con ERRORES en {elapsed/60:.1f} minutos")
        print(f"{'═'*70}\n")
        for step_name, error in failed_steps:
            print(f"  ✗ {step_name}: {error}")
        print()

    print(f"Resultados en: {ROOT}/output/")
    logger.info(f"Pipeline finalizado: {len(failed_steps)} error(es)")