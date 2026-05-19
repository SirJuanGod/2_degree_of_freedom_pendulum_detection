"""
Calcula theta1(t) y theta2(t) del péndulo doble desde trayectorias.
USO: python src/02_angles.py
"""
import numpy as np, yaml, matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"

def smooth(s, w=11, poly=3):
    return savgol_filter(s, w, poly) if len(s) >= w else s

def run():
    with open(ROOT/"config"/"colors.yaml") as f:
        cfg = yaml.safe_load(f)
    px_m = float(cfg["system"]["px_per_meter"])

    d = np.load(OUT/"trajectories.npz")
    P, J, T = d["pivot"], d["joint"], d["tip"]
    N = min(len(P), len(J), len(T))
    if N < 10:
        raise RuntimeError(
            f"Solo {N} frames con los 3 colores detectados. "
            "Revisa config/colors.yaml."
        )
    P, J, T = P[:N], J[:N], T[:N]
    time = P[:,0]
    dt   = float(np.mean(np.diff(time)))

    pivot = P[:,1:3].mean(axis=0)   # pivote fijo (media temporal)

    # arctan2(dx, dy) porque en imagen y crece hacia abajo
    theta1 = smooth(np.arctan2(J[:,1]-pivot[0], J[:,2]-pivot[1]))
    theta2 = smooth(np.arctan2(T[:,1]-J[:,1],   T[:,2]-J[:,2]))

    omega1 = np.gradient(theta1, dt) #type: ignore
    omega2 = np.gradient(theta2, dt) #type: ignore
    alpha1 = np.gradient(omega1, dt) 
    alpha2 = np.gradient(omega2, dt)

    L1_m = float(np.mean(np.hypot(J[:,1]-pivot[0], J[:,2]-pivot[1]))) / px_m
    L2_m = float(np.mean(np.hypot(T[:,1]-J[:,1],   T[:,2]-J[:,2])))   / px_m

    print(f"[02] N={N}  dt={dt*1000:.2f}ms  L1={L1_m:.4f}m  L2={L2_m:.4f}m")
    print(f"[02] θ1=[{np.degrees(theta1.min()):.1f}, {np.degrees(theta1.max()):.1f}]°  " #type: ignore
          f"θ2=[{np.degrees(theta2.min()):.1f}, {np.degrees(theta2.max()):.1f}]°") #type: ignore

    max_deg = max(abs(np.degrees(theta1)).max(), abs(np.degrees(theta2)).max()) #type: ignore
    if max_deg > 20:
        print(f"[02] ⚠  Oscilación máxima = {max_deg:.1f}° > 20° → modelo no lineal requerido")

    np.savez(OUT/"angles.npz",
             time=time, dt=np.array([dt]),
             theta1=theta1, theta2=theta2, #type: ignore
             omega1=omega1, omega2=omega2,
             alpha1=alpha1, alpha2=alpha2,
             L1_m=np.array([L1_m]), L2_m=np.array([L2_m]), pivot=pivot)

    fig, ax = plt.subplots(2, 1, figsize=(13,6), sharex=True)
    fig.suptitle("Ángulos — YOLO Tracking", fontweight="bold")
    ax[0].plot(time, np.degrees(theta1), color="#01696f", lw=2, label="θ₁") #type: ignore
    ax[0].axhline(0, color="gray", lw=0.8, ls="--")
    ax[0].set_ylabel("θ₁ [°]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(time, np.degrees(theta2), color="#da7101", lw=2, label="θ₂") #type: ignore
    ax[1].axhline(0, color="gray", lw=0.8, ls="--")
    ax[1].set_ylabel("θ₂ [°]"); ax[1].set_xlabel("Tiempo [s]")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"plot_angles.png", dpi=150); plt.close()
    print("[02] output/angles.npz  +  output/plot_angles.png")

if __name__ == "__main__": run()