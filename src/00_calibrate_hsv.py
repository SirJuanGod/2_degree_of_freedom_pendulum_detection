"""
00_calibrate_hsv.py — version baja iluminacion
USO: python src/00_calibrate_hsv.py --video videos/pendulo.mp4 --frame 100
"""
import cv2, numpy as np, argparse

def nothing(x): pass

def enhance(frame_bgr):
    """CLAHE + bilateral para mejorar imagen en baja iluminacion."""
    denoised = cv2.bilateralFilter(frame_bgr, d=9,
                                   sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def calibrate(video_path, frame_number=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"No se pudo leer frame {frame_number}")

    frame_enh = enhance(frame)   # imagen mejorada
    cv2.namedWindow("Original + Mejorada", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mascara",             cv2.WINDOW_NORMAL)
    cv2.namedWindow("Controles HSV",       cv2.WINDOW_NORMAL)

    sliders = [("H_low",0),("S_low",50),("V_low",40),
               ("H_high",15),("S_high",255),("V_high",255)]
    for name, val in sliders:
        cv2.createTrackbar(name, "Controles HSV", val, 255, nothing)

    print("\n[INSTRUCCIONES]")
    print("  Izquierda = frame original  |  Derecha = frame mejorado (CLAHE)")
    print("  Calibra sobre el frame MEJORADO (es lo que usa el tracker)")
    print("  Presiona 'q' para salir\n")

    while True:
        lo = np.array([cv2.getTrackbarPos("H_low",  "Controles HSV"),
                       cv2.getTrackbarPos("S_low",  "Controles HSV"),
                       cv2.getTrackbarPos("V_low",  "Controles HSV")])
        hi = np.array([cv2.getTrackbarPos("H_high", "Controles HSV"),
                       cv2.getTrackbarPos("S_high", "Controles HSV"),
                       cv2.getTrackbarPos("V_high", "Controles HSV")])

        hsv  = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lo, hi)

        # Si es rojo, agregar rango H=170-180
        if lo[0] <= 5 and hi[0] <= 15:
            lo2  = np.array([170, lo[1], lo[2]])
            hi2  = np.array([180, hi[1], hi[2]])
            mask = cv2.bitwise_or(mask,
                                  cv2.inRange(hsv, lo2, hi2))

        result   = cv2.bitwise_and(frame_enh, frame_enh, mask=mask)
        combined = np.hstack([frame, frame_enh])   # original | mejorado

        # Estadisticas de la mascara
        px_count = int(mask.sum() / 255)

        cv2.imshow("Original + Mejorada", combined)
        cv2.imshow("Mascara",             result)

        print(f"\r  low={lo.tolist()}  high={hi.tolist()}  "
              f"pixels_blancos={px_count}   ", end="")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\n\nCopia estos valores en config/colors.yaml:")
    print(f"  hsv_low:  {lo.tolist()}")
    print(f"  hsv_high: {hi.tolist()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frame", type=int, default=100)
    args = p.parse_args()
    calibrate(args.video, args.frame)