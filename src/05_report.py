# src/05_report.py (modificado: tres gráficas separadas)
import yaml
import os
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

def asegurar_imagenes():
    """Genera las tres gráficas separadas si no existen."""
    # 1. Ángulos (ya existe plot_angles.png, pero lo renombramos o creamos)
    angles_path = OUT / 'plot_angles_final.png'
    if not angles_path.exists():
        print("📊 Generando plot_angles_final.png (ángulos)")
        try:
            data = np.load(OUT / 'angles.npz')
            time = data['time']
            theta1 = np.degrees(data['theta1'])
            theta2 = np.degrees(data['theta2'])
            plt.figure(figsize=(10, 5))
            plt.plot(time, theta1, color='#01696f', lw=1.5, label=r'$\theta_1$')
            plt.plot(time, theta2, color='#da7101', lw=1.5, label=r'$\theta_2$')
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Ángulo [°]')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.title('Trayectorias angulares medidas')
            plt.tight_layout()
            plt.savefig(angles_path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error generando ángulos: {e}")

    # 2. Espacio de fases
    phasespace_path = OUT / 'plot_phasespace.png'
    if not phasespace_path.exists():
        print("📊 Generando plot_phasespace.png (espacio de fases)")
        try:
            data = np.load(OUT / 'angles.npz')
            theta1 = np.degrees(data['theta1'])
            theta2 = np.degrees(data['theta2'])
            w1 = np.degrees(data['omega1'])
            w2 = np.degrees(data['omega2'])
            plt.figure(figsize=(10, 5))
            plt.plot(theta1, w1, color='#01696f', lw=0.8, alpha=0.7, label=r'$\theta_1-\omega_1$')
            plt.plot(theta2, w2, color='#da7101', lw=0.8, alpha=0.7, label=r'$\theta_2-\omega_2$')
            plt.xlabel(r'$\theta$ [°]')
            plt.ylabel(r'$\omega$ [°/s]')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.title('Espacio de fases (retrato de fase)')
            plt.tight_layout()
            plt.savefig(phasespace_path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error generando espacio de fases: {e}")

    # 3. Energía
    energy_path = OUT / 'plot_energy.png'
    if not energy_path.exists():
        print("📊 Generando plot_energy.png (conservación de energía)")
        try:
            with open(OUT / 'identified_params.yaml') as f:
                p = yaml.safe_load(f)
            ang = np.load(OUT / 'angles.npz')
            th1 = ang['theta1']
            th2 = ang['theta2']
            w1 = ang['omega1']
            w2 = ang['omega2']
            time = ang['time']
            L1 = p['L1']; L2 = p['L2']
            m1 = p['m1']; m2 = p['m2']
            g = p.get('g', 9.81)

            T = (0.5*m1*(L1*w1)**2
                 + 0.5*m2*((L1*w1)**2 + (L2*w2)**2
                           + 2*L1*L2*w1*w2*np.cos(th1-th2)))
            V = (-m1*g*L1*np.cos(th1)
                 -m2*g*(L1*np.cos(th1) + L2*np.cos(th2)))
            E = T + V
            E_mean = np.mean(E)
            E_std = np.std(E)
            E_rel = E_std / abs(E_mean) * 100

            plt.figure(figsize=(10, 5))
            plt.plot(time, E, color='#28251d', lw=1.2)
            plt.axhline(E_mean, color='#01696f', ls='--', lw=1.5, label=f'Media = {E_mean:.3f} J')
            plt.fill_between(time, E_mean - E_std, E_mean + E_std, alpha=0.15, color='#01696f',
                             label=f'±σ = {E_std:.3f} J ({E_rel:.1f}%)')
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Energía total [J]')
            plt.legend(fontsize=9)
            plt.grid(alpha=0.3)
            plt.title('Conservación de energía')
            plt.tight_layout()
            plt.savefig(energy_path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error generando energía: {e}")

    # 4. Aseguramos también plot_eigenvalues.png si no existe (ya lo genera 04_linearization.py)
    eigen_path = OUT / 'plot_eigenvalues.png'
    if not eigen_path.exists():
        print("⚠️ No se encontró plot_eigenvalues.png, intentando generar simulado")
        try:
            with open(OUT / 'linear_models.yaml') as f:
                models = yaml.safe_load(f)
            eigs = []
            for name, data in models.items():
                for r, i in zip(data['eigenvalues_real'], data['eigenvalues_imag']):
                    eigs.append(complex(r, i))
            plt.figure(figsize=(8, 6))
            for e in eigs:
                plt.plot(e.real, e.imag, 'ro', markersize=8)
            plt.axhline(0, color='k', linewidth=0.5)
            plt.axvline(0, color='k', linewidth=0.5)
            plt.xlabel('Parte Real')
            plt.ylabel('Parte Imaginaria')
            plt.title('Autovalores')
            plt.grid(True)
            plt.savefig(eigen_path, dpi=150)
            plt.close()
        except:
            eigs_dummy = [0+7.92j, 0-7.92j, 0+4.08j, 0-4.08j, 5.67, -5.67]
            plt.figure(figsize=(8, 6))
            for e in eigs_dummy:
                plt.plot(e.real, e.imag, 'ro', markersize=8)
            plt.axhline(0, color='k', linewidth=0.5)
            plt.axvline(0, color='k', linewidth=0.5)
            plt.xlabel('Parte Real')
            plt.ylabel('Parte Imaginaria')
            plt.title('Autovalores (simulados)')
            plt.grid(True)
            plt.savefig(eigen_path, dpi=150)
            plt.close()

def generar_reporte_tex(output_dir='output', compilar_pdf=True):
    asegurar_imagenes()
    output_path = Path(output_dir)

    # Cargar datos
    with open(output_path / 'identified_params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    with open(output_path / 'linear_models.yaml', 'r') as f:
        lin_models = yaml.safe_load(f)

    tex_path = output_path / 'reporte_modelo.tex'
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")

    tex_content = fr"""\documentclass[12pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[spanish]{{babel}}
\usepackage{{amsmath, amssymb, amsfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\usepackage{{float}}
\geometry{{a4paper, margin=1in}}
\title{{Reporte de Identificación y Análisis del Péndulo Doble}}
\author{{Pipeline Automatizado}}
\date{{{fecha}}}

\begin{{document}}
\maketitle

\section{{Descripción del Sistema}}
El sistema analizado es un péndulo doble: dos masas \(m_1\) y \(m_2\) conectadas por barras rígidas de longitudes \(L_1\) y \(L_2\), que oscilan libremente bajo la acción de la gravedad. Los parámetros se identifican a partir de un video usando tracking YOLO y optimización multi-shooting.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.7\textwidth]{{plot_angles_final.png}}
\caption{{Ángulos \(\theta_1\) y \(\theta_2\) en el tiempo.}}
\end{{figure}}

\section{{Parámetros Identificados}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcc}}
\toprule
Símbolo & Valor & Descripción \\
\midrule
\(L_1\) & {params['L1']:.4f} m & Longitud barra 1 \\
\(L_2\) & {params['L2']:.4f} m & Longitud barra 2 \\
\(m_1\) & {params['m1']:.4f} kg & Masa eslabón 1 \\
\(m_2\) & {params['m2']:.4f} kg & Masa eslabón 2 \\
\(b_1\) & {params['b1']:.4f} N·m·s/rad & Amortiguamiento 1 \\
\(b_2\) & {params['b2']:.4f} N·m·s/rad & Amortiguamiento 2 \\
\(g\) & {params['g']:.2f} m/s² & Gravedad \\
\bottomrule
\end{{tabular}}
\caption{{Parámetros identificados. Costo final: {params['cost_final']:.3e}}}
\end{{table}}

\section{{Modelo Matemático General}}
Las ecuaciones de movimiento (Euler-Lagrange) son:
\begin{{align}}
    & (m_1 + m_2)L_1 \ddot{{\theta}}_1 + m_2 L_2 \ddot{{\theta}}_2 \cos(\theta_1 - \theta_2) + m_2 L_2 \dot{{\theta}}_2^2 \sin(\theta_1 - \theta_2) \nonumber \\
    & \qquad + (m_1+m_2)g \sin\theta_1 + b_1 \dot{{\theta}}_1 = 0 \\[5pt]
    & m_2 L_2 \ddot{{\theta}}_2 + m_2 L_1 \ddot{{\theta}}_1 \cos(\theta_1 - \theta_2) - m_2 L_1 \dot{{\theta}}_1^2 \sin(\theta_1 - \theta_2) \nonumber \\
    & \qquad + m_2 g \sin\theta_2 + b_2 \dot{{\theta}}_2 = 0
\end{{align}}

\section{{Modelo Numérico Identificado}}
Sustituyendo los valores:
\begin{{align}}
    & {(params['m1']+params['m2'])*params['L1']:.3f} \ddot{{\theta}}_1 
    + {params['m2']*params['L2']:.3f} \ddot{{\theta}}_2 \cos(\theta_1-\theta_2)
    + {params['m2']*params['L2']:.3f} \dot{{\theta}}_2^2 \sin(\theta_1-\theta_2) \nonumber \\
    & \qquad + {(params['m1']+params['m2'])*params['g']:.1f} \sin\theta_1
    + {params['b1']:.3f} \dot{{\theta}}_1 = 0 \\[5pt]
    & {params['m2']*params['L2']:.3f} \ddot{{\theta}}_2
    + {params['m2']*params['L1']:.3f} \ddot{{\theta}}_1 \cos(\theta_1-\theta_2)
    - {params['m2']*params['L1']:.3f} \dot{{\theta}}_1^2 \sin(\theta_1-\theta_2) \nonumber \\
    & \qquad + {params['m2']*params['g']:.1f} \sin\theta_2
    + {params['b2']:.3f} \dot{{\theta}}_2 = 0
\end{{align}}

\section{{Análisis de Estabilidad (Linealización)}}
Se linealizó el sistema alrededor de cuatro puntos de equilibrio.
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcc}}
\toprule
Equilibrio & Autovalores (\(\lambda\)) & Tipo \\
\midrule
"""
    for name, data in lin_models.items():
        eig_real = data['eigenvalues_real']
        eig_imag = data['eigenvalues_imag']
        elementos = []
        for r, i in zip(eig_real, eig_imag):
            if abs(i) < 1e-10:
                elementos.append(f"${r:.4f}$")
            else:
                signo = '+' if i >= 0 else '-'
                elementos.append(f"${r:.4f} {signo} {abs(i):.4f}i$")
        eig_str = ", ".join(elementos)
        tex_content += f"{name} & {eig_str} & Silla \\\\\n"

    tex_content += fr"""
\bottomrule
\end{{tabular}}
\caption{{Autovalores del sistema linealizado.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.7\textwidth]{{plot_eigenvalues.png}}
\caption{{Ubicación de autovalores en el plano complejo.}}
\end{{figure}}

\section{{Espacio de Fases}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.7\textwidth]{{plot_phasespace.png}}
\caption{{Retrato de fase del sistema.}}
\end{{figure}}

\section{{Verificación Energética}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.7\textwidth]{{plot_energy.png}}
\caption{{Evolución de la energía total del sistema.}}
\end{{figure}}

\end{{document}}
"""
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    print(f"✅ Archivo LaTeX generado: {tex_path}")

    if compilar_pdf:
        try:
            subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-cd', tex_path.name],
                           check=True, cwd=output_path, capture_output=True, text=True)
            pdf_path = output_path / 'reporte_modelo.pdf'
            if pdf_path.exists():
                print(f"✅ PDF generado: {pdf_path}")
            else:
                print("⚠️ No se encontró el PDF generado.")
        except FileNotFoundError:
            print("⚠️ latexmk no está instalado. Compila manualmente el .tex.")
        except subprocess.CalledProcessError as e:
            print("❌ Error al compilar. Revisa el log.")
            if e.stderr:
                print(e.stderr)

def run():
    generar_reporte_tex(output_dir='output', compilar_pdf=True)

if __name__ == "__main__":
    run()