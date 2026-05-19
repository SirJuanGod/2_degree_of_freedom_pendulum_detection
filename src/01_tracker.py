"""
01_tracker.py — versión con doble verificación
================================================
Estrategia por frame:
  PASE 1 → YOLO detecta bounding boxes → aplica filtro HSV dentro del ROI
  PASE 2 → Si un color NO fue encontrado en el paso 1,
            busca directamente en toda la imagen (full-frame HSV scan)

Esto garantiza detección aunque YOLO no encuadre bien la pegatina.
"""
import cv2, numpy as np, yaml, argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"; OUT.mkdir(exist_ok=True)

def load_cfg():
    with open(ROOT/"config"/"colors.yaml") as f:
        return yaml.safe_load(f)

def best_centroid_in_mask(mask):
    """
    Retorna el centroide del contorno MAS GRANDE en la mascara.
    Mas robusto que moments() cuando hay ruido.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    if area < 50:          # demasiado pequeño = ruido
        return None, 0
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None, 0
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])), area

def hsv_mask(img_bgr, lo, hi):
    """Mascara HSV. Maneja el caso especial del ROJO que cruza H=180."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo_arr = np.array(lo, np.uint8)
    hi_arr = np.array(hi, np.uint8)

    # Rojo cruza la frontera H=0/180 → union de dos rangos
    if lo_arr[0] <= 5 and hi_arr[0] <= 15:
        mask1 = cv2.inRange(hsv, lo_arr, hi_arr)
        lo2 = np.array([170, lo_arr[1], lo_arr[2]], np.uint8)
        hi2 = np.array([180, hi_arr[1], hi_arr[2]], np.uint8)
        mask2 = cv2.inRange(hsv, lo2, hi2)
        return cv2.bitwise_or(mask1, mask2)
    return cv2.inRange(hsv, lo_arr, hi_arr)

def find_color_in_roi(frame, box, lo, hi, min_px):
    """PASE 1: busca el color dentro del bounding box de YOLO."""
    H, W = frame.shape[:2]
    x1,y1,x2,y2 = box
    x1,y1 = max(x1,0), max(y1,0)
    x2,y2 = min(x2,W), min(y2,H)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    mask = hsv_mask(roi, lo, hi)
    c, area = best_centroid_in_mask(mask)
    if c and area >= min_px:
        return (x1 + c[0], y1 + c[1])
    return None

def find_color_fullframe(frame, lo, hi, min_px):
    """PASE 2: busca el color en toda la imagen si YOLO no lo encontró."""
    mask = hsv_mask(frame, lo, hi)
    # Morfología para limpiar ruido pequeño
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    c, area = best_centroid_in_mask(mask)
    if c and area >= min_px:
        return c
    return None

def run(video_path):
    cfg    = load_cfg()
    colors = cfg["colors"]
    min_px = cfg["system"]["min_pixels"]

    model = YOLO("yolov8n.pt")
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(OUT/"video_tracked.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H) #type: ignore
    )

    traj = {k: [] for k in colors}

    # Contadores para diagnostico
    found_p1  = {k: 0 for k in colors}   # detectado en YOLO ROI
    found_p2  = {k: 0 for k in colors}   # detectado en full-frame
    missed    = {k: 0 for k in colors}   # no detectado en ningún pase

    t = 0.0; frame_count = 0
    print(f"[01] {W}x{H} @ {fps:.1f}fps  →  {video_path}")
    print(f"[01] Estrategia: YOLO ROI (pase 1) + Full-frame HSV (pase 2)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # ── Pase 1: YOLO
        results = model.track(frame, persist=True,
                               tracker="bytetrack.yaml", verbose=False)
        boxes = []
        if results[0].boxes.id is not None: #type: ignore
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) #type: ignore

        detected_this_frame = {k: None for k in colors}

        # Buscar cada color en los ROI de YOLO
        for box in boxes:
            for name, params in colors.items():
                if detected_this_frame[name] is not None:
                    continue   # ya lo encontramos en otro box
                c = find_color_in_roi(frame, box,
                                      params["hsv_low"],
                                      params["hsv_high"],
                                      min_px)
                if c:
                    detected_this_frame[name] = c #type: ignore
                    found_p1[name] += 1

        # ── Pase 2: Full-frame para los colores no encontrados
        for name, params in colors.items():
            if detected_this_frame[name] is not None:
                continue
            c = find_color_fullframe(frame,
                                     params["hsv_low"],
                                     params["hsv_high"],
                                     min_px)
            if c:
                detected_this_frame[name] = c #type: ignore
                found_p2[name] += 1
            else:
                missed[name] += 1

        # ── Guardar y dibujar
        for name, c in detected_this_frame.items():
            if c:
                traj[name].append((t, c[0], c[1]))
                col = tuple(int(v) for v in colors[name]["bgr_draw"])
                cv2.circle(frame, c, 8, col, -1) #type: ignore
                cv2.circle(frame, c, 10, (255,255,255), 2)  # borde blanco
                cv2.putText(frame, name, (c[0]+12, c[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, col, 2, cv2.LINE_AA)

        # HUD con contadores en tiempo real
        cv2.putText(frame, f"t={t:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255,255,255), 2, cv2.LINE_AA)
        for i, (name, c) in enumerate(detected_this_frame.items()):
            estado = "OK" if c else "---"
            col_hud = (0,255,0) if c else (0,0,255)
            cv2.putText(frame, f"{name}: {estado}", (10, 60 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        col_hud, 2, cv2.LINE_AA)

        writer.write(frame)
        t += 1.0/fps; frame_count += 1

        if frame_count % 200 == 0:
            print(f"  frame {frame_count:5d}  t={t:.2f}s  |  "
                  + "  ".join(f"{k}:{len(v)}" for k,v in traj.items()))

    cap.release(); writer.release()

    # ── Reporte de deteccion
    print(f"\n[01] ══ REPORTE DE DETECCIÓN ══  ({frame_count} frames totales)")
    print(f"{'Color':<10} {'YOLO(p1)':>10} {'Full-frame(p2)':>15} "
          f"{'Perdidos':>10} {'Total':>8} {'Cobertura':>10}")
    print("─"*65)
    for name in colors:
        total = found_p1[name] + found_p2[name]
        cov   = 100*total/frame_count if frame_count > 0 else 0
        print(f"{name:<10} {found_p1[name]:>10} {found_p2[name]:>15} "
              f"{missed[name]:>10} {total:>8} {cov:>9.1f}%")

    if any(found_p1[k]+found_p2[k] < frame_count*0.5 for k in colors):
        print("\n⚠  Algún color tiene < 50% de cobertura.")
        print("   Recalibra ese color con 00_calibrate_hsv.py")

    np.savez(OUT/"trajectories.npz",
             **{k: np.array(v, dtype=np.float64) for k,v in traj.items()}) #type: ignore
    print(f"\n[01] output/trajectories.npz  +  output/video_tracked.mp4")
    for k,v in traj.items(): print(f"  {k}: {len(v)} puntos")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    run(p.parse_args().video)