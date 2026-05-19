"""
Linealiza el modelo en 4 equilibrios mediante Jacobiano numérico.
USO: python src/04_linearization.py
"""
import numpy as np, yaml, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"

try: import scipy.io as sio;   HAVE_MAT = True
except: HAVE_MAT = False
try: import control as ctrl;   HAVE_CTRL = True
except: HAVE_CTRL = False

def f_nl(x, L1, L2, m1, m2, g):
    th1,th2,w1,w2 = x; d = th1-th2
    den = 2*m1+m2-m2*np.cos(2*d)+1e-14
    dw1 = (-g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)
           -2*np.sin(d)*m2*(w2**2*L2+w1**2*L1*np.cos(d)))/(L1*den)
    dw2 = (2*np.sin(d)*(w1**2*L1*(m1+m2)+g*(m1+m2)*np.cos(th1)
           +w2**2*L2*m2*np.cos(d)))/(L2*den)
    return np.array([w1,w2,dw1,dw2])

def jacobian(x_eq, L1, L2, m1, m2, g, eps=1e-6):
    A = np.zeros((4,4)); f0 = f_nl(x_eq,L1,L2,m1,m2,g)
    for i in range(4):
        xp=x_eq.copy(); xp[i]+=eps
        A[:,i]=(f_nl(xp,L1,L2,m1,m2,g)-f0)/eps
    return A

def stab(eigs):
    if np.all(np.real(eigs)<0): return "ESTABLE",   "#437a22"
    if np.all(np.real(eigs)>0): return "INESTABLE", "#a12c7b"
    return "SILLA",                                  "#da7101"

def run():
    with open(OUT/"identified_params.yaml") as f: p=yaml.safe_load(f)
    L1,L2,m1,m2,g = p["L1"],p["L2"],p["m1"],p["m2"],p["g"]

    EQ = {
        "P1_colgante":    np.array([0.,    0.,   0.,0.]),
        "P2_semi_inv_b1": np.array([np.pi, 0.,   0.,0.]),
        "P3_semi_inv_b2": np.array([0.,    np.pi,0.,0.]),
        "P4_invertido":   np.array([np.pi, np.pi,0.,0.]),
    }
    results = {}; all_eigs = {}; A_stable = None; stab_name = None

    print("[04] ══ ANÁLISIS DE EQUILIBRIOS ══")
    for name, x_eq in EQ.items():
        A    = jacobian(x_eq, L1,L2,m1,m2,g)
        eigs = np.linalg.eigvals(A)
        label, _ = stab(eigs)
        results[name] = {"x_eq":x_eq.tolist(),"A":A.tolist(),
                         "eigenvalues_real":np.real(eigs).tolist(),
                         "eigenvalues_imag":np.imag(eigs).tolist(),
                         "stable":bool(np.all(np.real(eigs)<0))}
        all_eigs[name] = eigs
        print(f"\n  {name}  →  {label}")
        print(f"  Eigenvalores: {np.round(eigs,4)}")
        print(f"  A:\n{np.round(A,5)}")
        if label=="ESTABLE" and A_stable is None:
            A_stable=A; stab_name=name

    import yaml as _y
    with open(OUT/"linear_models.yaml","w") as f: _y.dump(results,f)

    if HAVE_MAT:
        sio.savemat(OUT/"pendulo_modelo.mat",
                    {"L1":L1,"L2":L2,"m1":m1,"m2":m2,"g":g,
                     "A_colgante": np.array(results["P1_colgante"]["A"]),
                     "B": np.zeros((4,1)), "C": np.eye(4), "D": np.zeros((4,1))})
        print("[04] output/pendulo_modelo.mat (para MATLAB/Simulink)")

    # Mapa de polos
    COLS = ["#01696f","#da7101","#a12c7b","#006494"]
    MKS  = ["o","s","^","D"]
    fig, ax = plt.subplots(figsize=(9,8))
    patches = []
    for (nm,eigs), col, mk in zip(all_eigs.items(), COLS, MKS):
        re,im = np.real(eigs), np.imag(eigs)
        ax.scatter(re,im, marker=mk, s=140, color=col, zorder=5)
        for rv,iv in zip(re,im):
            ax.annotate(f"  {rv:.2f}{iv:+.2f}j",(rv,iv),
                        fontsize=7.5, color=col, va="center")
        patches.append(mpatches.Patch(color=col, label=nm))
    lim=18
    ax.fill_betweenx([-lim,lim],-lim,0, alpha=0.05, color="green")
    ax.axvline(0,color="black",lw=1,ls="--",alpha=0.6)
    ax.axhline(0,color="black",lw=1,ls="--",alpha=0.6)
    ax.set_xlim([-lim,lim]); ax.set_ylim([-lim,lim])
    ax.set_xlabel("Parte Real (σ)"); ax.set_ylabel("Parte Imag (jω)")
    ax.set_title("Mapa de Polos — 4 Equilibrios", fontweight="bold")
    ax.legend(handles=patches, fontsize=9, loc="upper right")
    ax.text(-lim*.9, lim*.88, "← Región estable", fontsize=9,
            color="#437a22", alpha=0.8)
    ax.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(OUT/"plot_eigenvalues.png", dpi=150); plt.close()

    # Respuesta libre P1 con python-control
    if HAVE_CTRL and A_stable is not None:
        sys_ss = ctrl.ss(A_stable, np.zeros((4,1)), np.eye(4), np.zeros((4,1)))
        t_sim  = np.linspace(0,6,3000)
        x0     = np.array([np.radians(8),np.radians(4),0.,0.])
        t_out, y_out = ctrl.forced_response(sys_ss, T=t_sim, X0=x0)
        fig,ax2 = plt.subplots(2,1, figsize=(13,6), sharex=True)
        fig.suptitle(f"Respuesta Libre Lineal — {stab_name} (θ₁₀=8°, θ₂₀=4°)",
                     fontweight="bold")
        ax2[0].plot(t_out, np.degrees(y_out[0]), color="#01696f", lw=2, label="θ₁")
        ax2[0].set_ylabel("θ₁ [°]"); ax2[0].legend(); ax2[0].grid(alpha=0.3)
        ax2[1].plot(t_out, np.degrees(y_out[1]), color="#da7101", lw=2, label="θ₂")
        ax2[1].set_ylabel("θ₂ [°]"); ax2[1].set_xlabel("Tiempo [s]")
        ax2[1].legend(); ax2[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT/"plot_free_response.png", dpi=150); plt.close()
        print("[04] output/plot_free_response.png")

    print("[04] output/linear_models.yaml  +  output/plot_eigenvalues.png")

if __name__ == "__main__": run()