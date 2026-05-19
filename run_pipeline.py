"""
Pipeline completo — un solo comando.
USO: python run_pipeline.py --video pendulo.mp4
"""
import argparse, importlib.util, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def step(n, title):
    print(f"\n{'─'*55}\n  PASO {n}/4 — {title}\n{'─'*55}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video",    required=True)
    p.add_argument("--skip-id",  action="store_true")
    p.add_argument("--skip-lin", action="store_true")
    args = p.parse_args()

    src  = ROOT/"src"
    trk  = load("trk", src/"01_tracker.py")
    ang  = load("ang", src/"02_angles.py")
    idn  = load("idn", src/"03_identification.py")
    lin  = load("lin", src/"04_linearization.py")
    rep  = load("rep", src/"05_report.py")

    t0 = time.time()

    step(1, "YOLO TRACKING + FILTRO HSV");      trk.run(args.video)
    step(2, "CÁLCULO DE ÁNGULOS");              ang.run()

    if not args.skip_id:
        step(3, "IDENTIFICACIÓN NO LINEAL");    idn.run()
    if not args.skip_lin and not args.skip_id:
        step(4, "LINEALIZACIÓN + ESTABILIDAD"); lin.run()

    rep.run()
    print(f"\n✓ Pipeline completado en {time.time()-t0:.1f}s")
    print(f"  Resultados en: {ROOT}/output/")