"""
05_report.py — reporte final compatible con costo por energia+frecuencias
 + generación de PDF con modelo matemático
"""
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

# ---- NUEVO: imports para PDF ----
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, Image as RLImage
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from io import BytesIO


ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"


# ---- NUEVO: estilos y helpers PDF ----
def S(name, **kw):
    return ParagraphStyle(name, **kw)


TEAL   = HexColor('#01696f')
DARK   = HexColor('#28251d')
MUTED  = HexColor('#7a7974')
ORANGE = HexColor('#da7101')
PURPLE = HexColor('#7a39bb')
BG_EQ  = HexColor('#eef5f4')
BG_NUM = HexColor('#fdf3e7')
BG_SS  = HexColor('#f5f0fa')
WARN   = HexColor('#fdf3e7')
WARN_B = HexColor('#da7101')
WHITE  = colors.white
DIVIDER = HexColor('#dcd9d5')


title_s = S('title', fontName='Helvetica-Bold', fontSize=20,
            textColor=DARK, alignment=TA_CENTER, spaceAfter=2)
subtitle_s = S('sub', fontName='Helvetica', fontSize=10,
               textColor=MUTED, alignment=TA_CENTER, spaceAfter=6)
h2_s = S('h2', fontName='Helvetica-Bold', fontSize=13,
         textColor=TEAL, spaceBefore=10, spaceAfter=4)
body_s = S('body', fontName='Helvetica', fontSize=9.5,
           textColor=DARK, leading=15, alignment=TA_JUSTIFY, spaceAfter=4)
small_s = S('small', fontName='Helvetica-Oblique', fontSize=8.5,
            textColor=MUTED, spaceAfter=2)
eq_s = S('eq', fontName='Helvetica-Bold', fontSize=10,
         textColor=HexColor('#0f3638'), alignment=TA_CENTER,
         leading=16, spaceAfter=2)
eq_num_s = S('eqnum', fontName='Helvetica', fontSize=8.5,
             textColor=MUTED, alignment=TA_CENTER, spaceAfter=0)
num_eq_s = S('numeq', fontName='Helvetica-Bold', fontSize=9.5,
             textColor=HexColor('#4b2614'), alignment=TA_CENTER,
             leading=15, spaceAfter=2)
warn_s = S('warn', fontName='Helvetica', fontSize=9,
           textColor=HexColor('#964219'), leading=14,
           alignment=TA_LEFT, spaceAfter=4)
ss_s = S('ss', fontName='Helvetica-Bold', fontSize=10,
         textColor=HexColor('#431673'), leading=17,
         alignment=TA_LEFT, spaceAfter=0)


def eq_table(lines, bg_color, border_color, num):
    data = [[Paragraph('<br/>'.join(lines),
                       eq_s if bg_color == BG_EQ else num_eq_s)]]
    t = Table(data, colWidths=[14.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('ROUNDEDCORNERS', [6]),
        ('BOX', (0, 0), (-1, -1), 1.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    num_p = Paragraph(f'<font color="#7a7974">({num})</font>', eq_num_s)
    return Table(
        [[t, num_p]],
        colWidths=[14.5*cm, 1.0*cm],
        style=[
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ],
    )


def param_table(L1, L2, m1, m2, g):
    rows = [
        ['Símbolo', 'Valor', 'Descripción'],
        ['L₁', f'{L1:.4f} m', 'Longitud barra 1'],
        ['L₂', f'{L2:.4f} m', 'Longitud barra 2'],
        ['m₁', f'{m1:.4f} kg', 'Masa del eslabón 1'],
        ['m₂', f'{m2:.4f} kg', 'Masa del eslabón 2'],
        ['g',  f'{g:.4f} m/s²', 'Aceleración gravitacional'],
        ['Δ',  'θ₁ − θ₂', 'Diferencia de ángulos'],
    ]
    t = Table(rows, colWidths=[2.5*cm, 3.5*cm, 9.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TEAL),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9.5),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TEXTCOLOR', (0, 1), (-1, -1), DARK),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, HexColor('#f0f7f6')]),
        ('GRID', (0, 0), (-1, -1), 0.5, DIVIDER),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


def make_diagram():
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    fig.patch.set_facecolor('#f9f8f5')
    ax.set_facecolor('#f9f8f5')
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-2.2, 0.5)
    ax.axis('off')

    ax.hlines(0.22, -0.55, 0.55, colors='#28251d', lw=3)
    for xi in np.linspace(-0.5, 0.5, 6):
        ax.plot([xi, xi-0.14], [0.22, 0.40],
                '-', color='#7a7974', lw=1)

    ax.plot(0, 0, 's', ms=9, color='#28251d', zorder=5)

    t1 = np.radians(32)
    x1 = np.sin(t1)
    y1 = -np.cos(t1)
    ax.plot([0, x1], [0, y1], '-', color='#01696f', lw=2.5)
    ax.plot(x1, y1, 'o', ms=11, color='#da7101', zorder=5)
    ax.annotate('m₁', (x1, y1), (x1+0.13, y1+0.05), fontsize=9,
                color='#28251d',
                arrowprops=dict(arrowstyle='-',
                                color='#dcd9d5', lw=0.5))
    ax.text(x1/2+0.11, y1/2, 'L₁', fontsize=8.5,
            color='#01696f', fontweight='bold')

    ta = np.linspace(-np.pi/2, t1, 40)
    ax.plot(0.30*np.sin(ta), -0.30*np.cos(ta),
            color='#7a7974', lw=1.2, ls='--')
    ax.text(0.06, -0.40, 'θ₁', fontsize=9,
            color='#7a7974', fontstyle='italic')

    t2 = np.radians(-24)
    x2 = x1 + np.sin(t1+t2)*0.88
    y2 = y1 - np.cos(t1+t2)*0.88
    ax.plot([x1, x2], [y1, y2], '-', color='#01696f', lw=2.5)
    ax.plot(x2, y2, 'o', ms=11, color='#a12c7b', zorder=5)
    ax.annotate('m₂', (x2, y2), (x2+0.10, y2-0.10), fontsize=9,
                color='#28251d',
                arrowprops=dict(arrowstyle='-',
                                color='#dcd9d5', lw=0.5))
    ax.text((x1+x2)/2+0.10, (y1+y2)/2, 'L₂', fontsize=8.5,
            color='#01696f', fontweight='bold')

    ax.text(0, -2.08, 'Fig. 1 — Diagrama del péndulo doble',
            ha='center', fontsize=7.5,
            color='#7a7974', fontstyle='italic')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=160,
                bbox_inches='tight', facecolor='#f9f8f5')
    plt.close()
    buf.seek(0)
    return buf


def generar_pdf(L1, L2, m1, m2, g, cost_val,
                E_mean, E_std, E_rel):
    doc = SimpleDocTemplate(
        str(OUT / "reporte_modelo.pdf"),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    story = []

    # Título
    story.append(Paragraph(
        "Identificación del Modelo Matemático", title_s))
    story.append(Paragraph(
        "Péndulo Doble — Sistema No Lineal &nbsp;·&nbsp; "
        "Pipeline YOLO + Optimización Energética",
        subtitle_s))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=TEAL, spaceAfter=10))

    # Sección 1
    story.append(Paragraph("1. Descripción del Sistema",
                           h2_s))
    diag_buf = make_diagram()
    diag_img = RLImage(diag_buf, width=5.5*cm, height=5.5*cm)

    desc_text = Paragraph(
        "El sistema analizado es un <b>péndulo doble</b>: dos masas "
        "(m₁ y m₂) conectadas por barras rígidas de longitud L₁ y L₂, "
        "que oscilan libremente bajo la acción de la gravedad. Posee "
        "<b>dos grados de libertad</b>: el ángulo θ₁ del primer eslabón "
        "respecto a la vertical y el ángulo θ₂ del segundo eslabón.<br/><br/>"
        "El modelo se obtiene aplicando las <b>ecuaciones de Euler-Lagrange</b> "
        "al Lagrangiano L = T − V. Los parámetros se identifican a partir del "
        "video usando tracking YOLO y conservación de energía.",
        body_s
    )

    sec1_table = Table(
        [[desc_text, diag_img]],
        colWidths=[10.0*cm, 5.5*cm],
        style=[
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (0, -1), 0),
            ('RIGHTPADDING', (0, 0), (0, -1), 8),
            ('LEFTPADDING', (1, 0), (1, -1), 4),
            ('RIGHTPADDING', (1, 0), (1, -1), 0),
        ]
    )
    story.append(sec1_table)
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=DIVIDER, spaceAfter=6))

    # Sección 2
    story.append(Paragraph("2. Parámetros Identificados",
                           h2_s))
    story.append(param_table(L1, L2, m1, m2, g))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=DIVIDER, spaceAfter=6))

    # Sección 3: modelo general
    story.append(Paragraph("3. Modelo Matemático General",
                           h2_s))
    story.append(Paragraph(
        "Aplicando Euler-Lagrange se obtienen las "
        "<b>ecuaciones diferenciales de segundo orden</b> del péndulo doble:",
        body_s))

    story.append(Spacer(1, 0.2*cm))
    story.append(eq_table([
        "θ̈₁  =  −g·(2m₁+m₂)·sin θ₁  −  m₂·g·sin(θ₁−2θ₂)  −  2·m₂·sin Δ·[ θ̇₂²·L₂ + θ̇₁²·L₁·cos Δ ]",
        "                                        ──────────────────────────────────────────────",
        "                                        L₁ · [ 2m₁ + m₂ − m₂·cos(2Δ) ]"
    ], BG_EQ, TEAL, "1"))

    story.append(Spacer(1, 0.25*cm))
    story.append(eq_table([
        "θ̈₂  =  2·sin Δ · [ θ̇₁²·L₁·(m₁+m₂)  +  g·(m₁+m₂)·cos θ₁  +  θ̇₂²·L₂·m₂·cos Δ ]",
        "                                        ────────────────────────────────────────────",
        "                                        L₂ · [ 2m₁ + m₂ − m₂·cos(2Δ) ]"
    ], BG_EQ, TEAL, "2"))

    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        "<i>Donde  Δ = θ₁ − θ₂  es la diferencia de ángulos entre los eslabones.</i>",
        small_s))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=DIVIDER, spaceAfter=6))

    # Sección 4: modelo numérico
    story.append(Paragraph("4. Modelo con Valores Numéricos",
                           h2_s))
    story.append(Paragraph(
        "Sustituyendo los parámetros identificados en las ecuaciones (1) y (2):",
        body_s))

    Dc = 2*m1 + m2
    den = f"{Dc:.3f} − {m2:.3f}·cos(2Δ)"

    story.append(Spacer(1, 0.2*cm))
    story.append(eq_table([
        f"θ̈₁  =  {-(g*(2*m1+m2)):.2f}·sin θ₁  −  {m2*g:.2f}·sin(θ₁−2θ₂)  −  {2*m2:.2f}·sin Δ·[ {L2:.3f}·θ̇₂² + {L1:.3f}·θ̇₁²·cos Δ ]",
        "                            ───────────────────────────────────────────",
        f"                            {L1:.3f} · [ {den} ]"
    ], BG_NUM, ORANGE, "3"))

    story.append(Spacer(1, 0.25*cm))
    story.append(eq_table([
        f"θ̈₂  =  2·sin Δ · [ {L1:.3f}·{m1+m2:.3f}·θ̇₁²  +  {g*(m1+m2):.2f}·cos θ₁  +  {L2*m2:.3f}·θ̇₂²·cos Δ ]",
        "                            ───────────────────────────────────────────",
        f"                            {L2:.3f} · [ {den} ]"
    ], BG_NUM, ORANGE, "4"))

    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=DIVIDER, spaceAfter=6))

    # Sección 5: espacio de estados
    story.append(Paragraph("5. Representación en Espacio de Estados",
                           h2_s))
    story.append(Paragraph(
        "Definiendo el vector de estado  x = [θ₁, θ₂, ω₁, ω₂]ᵀ  con  ω₁ = θ̇₁  y  ω₂ = θ̇₂:",
        body_s))

    ss_lines = [
        "ẋ = f(x)   →   Sistema de 4 ecuaciones de primer orden:",
        "",
        "    θ̇₁  =  ω₁",
        "    θ̇₂  =  ω₂",
        "    ω̇₁  =  f₁(θ₁, θ₂, ω₁, ω₂)   →  Ec. (3)",
        "    ω̇₂  =  f₂(θ₁, θ₂, ω₁, ω₂)   →  Ec. (4)",
    ]
    ss_data = [[Paragraph('<br/>'.join(ss_lines), ss_s)]]
    ss_t = Table(ss_data, colWidths=[15.5*cm])
    ss_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_SS),
        ('BOX', (0, 0), (-1, -1), 1.5, PURPLE),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(ss_t)
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=DIVIDER, spaceAfter=6))

    # Nota caos
    warn_data = [[Paragraph(
        "<b>⚠  Comportamiento Caótico</b><br/>"
        "Para oscilaciones mayores a 20° el péndulo doble es caótico: "
        "pequeñas variaciones en las condiciones iniciales producen "
        "trayectorias completamente distintas a largo plazo. La identificación "
        "es válida, pero la predicción a largo horizonte es limitada.",
        warn_s)]]
    w_t = Table(warn_data, colWidths=[15.5*cm])
    w_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), WARN),
        ('BOX', (0, 0), (-1, -1), 1.2, WARN_B),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(w_t)
    story.append(Spacer(1, 0.5*cm))

    # Pie
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=DIVIDER, spaceAfter=3))
    story.append(Paragraph(
        "Generado automáticamente · Pipeline YOLO Tracking + Identificación Paramétrica "
        "&nbsp;&nbsp;|&nbsp;&nbsp; SYS2 — USC Palmira",
        ParagraphStyle(
            'foot', fontName='Helvetica', fontSize=7.5,
            textColor=MUTED, alignment=TA_CENTER
        )
    ))

    doc.build(story)
    print(f"[05] {OUT/'reporte_modelo.pdf'}")


# ----------------------------------------------------------------
#  CÓDIGO ORIGINAL + LLAMADA A generar_pdf(...)
# ----------------------------------------------------------------
def run():
    # Cargar parámetros identificados
    with open(OUT/"identified_params.yaml") as f:
        p = yaml.safe_load(f)

    with open(ROOT/"config"/"colors.yaml") as f:
        cfg = yaml.safe_load(f)

    ang = np.load(OUT/"angles.npz")

    L1 = p["L1"];  L2 = p["L2"]
    m1 = p["m1"];  m2 = p["m2"]
    g  = p["g"]

    # Clave flexible: acepta tanto 'cost_final' como 'cost_mse'
    cost_val = p.get("cost_final",
               p.get("cost_mse",
               p.get("cost", float("nan"))))

    # Energía con parámetros identificados
    th1 = ang["theta1"]; th2 = ang["theta2"]
    w1  = ang["omega1"];  w2  = ang["omega2"]

    T = (0.5*m1*(L1*w1)**2
         + 0.5*m2*((L1*w1)**2 + (L2*w2)**2
                   + 2*L1*L2*w1*w2*np.cos(th1-th2)))
    V = (-m1*g*L1*np.cos(th1)
         -m2*g*(L1*np.cos(th1) + L2*np.cos(th2)))
    E = T + V

    E_mean = np.mean(E)
    E_std  = np.std(E)
    E_rel  = E_std / abs(E_mean) * 100   # % variación (ideal = 0%)

    # Ecuaciones del modelo (consola)
    print("\n" + "═"*60)
    print("  REPORTE FINAL — PIPELINE YOLO + IDENTIFICACIÓN")
    print("═"*60)
    print(f"\n[PARÁMETROS FÍSICOS]")
    print(f"  L1 = {L1:.6f} m")
    print(f"  L2 = {L2:.6f} m")
    print(f"  m1 = {m1:.6f} kg")
    print(f"  m2 = {m2:.6f} kg")
    print(f"  g  = {g:.4f} m/s²")

    print(f"\n[CALIDAD DEL AJUSTE]")
    print(f"  Costo final    = {cost_val:.4e}  (adimensional)")
    print(f"  Energía media  = {E_mean:.4f} J")
    print(f"  Energía std    = {E_std:.4f} J")
    print(f"  Variación E    = {E_rel:.2f}%  (ideal = 0%)")

    if E_rel < 5:
        calidad = "✅ EXCELENTE"
    elif E_rel < 15:
        calidad = "✅ BUENO"
    elif E_rel < 30:
        calidad = "⚠  ACEPTABLE"
    else:
        calidad = "❌ REVISAR — posible error en tracking"
    print(f"  Calidad        = {calidad}")

    print(f"\n[MODELO MATEMÁTICO — PÉNDULO DOBLE NO LINEAL]")
    print(f"  Lagrangiano del sistema (Euler-Lagrange):\n")
    den = "(2m1+m2-m2·cos(2Δ))"
    print(f"  θ1'' = [-g(2m1+m2)sin(θ1) - m2·g·sin(θ1-2θ2)")
    print(f"          - 2sin(Δ)·m2·(θ2'²·L2 + θ1'²·L1·cos(Δ))]")
    print(f"         / [L1·{den}]\n")
    print(f"  θ2'' = [2sin(Δ)·(θ1'²·L1·(m1+m2)")
    print(f"          + g·(m1+m2)·cos(θ1) + θ2'²·L2·m2·cos(Δ))]")
    print(f"         / [L2·{den}]\n")
    print(f"  Donde Δ = θ1 - θ2")

    print(f"\n[MODELO CON VALORES NUMÉRICOS]")
    den_str = f"({2*m1+m2:.4f} - {m2:.4f}·cos(2Δ))"
    print(f"  θ1'' = [-{g*(2*m1+m2):.4f}·sin(θ1) - {m2*g:.4f}·sin(θ1-2θ2)")
    print(f"          - {2*m2:.4f}·sin(Δ)·(θ2'²·{L2:.4f} + θ1'²·{L1:.4f}·cos(Δ))]")
    print(f"         / [{L1:.4f}·{den_str}]\n")
    print(f"  θ2'' = [2·sin(Δ)·(θ1'²·{L1:.4f}·{m1+m2:.4f}")
    print(f"          + {g*(m1+m2):.4f}·cos(θ1) + θ2'²·{L2:.4f}·{m2:.4f}·cos(Δ))]")
    print(f"         / [{L2:.4f}·{den_str}]")

    # ── Gráfico de energía total ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"Reporte final — L1={L1:.4f}m  L2={L2:.4f}m  "
        f"m1={m1:.4f}kg  m2={m2:.4f}kg",
        fontweight="bold"
    )

    time = ang["time"]

    # θ1 y θ2
    axes[0].plot(time, np.degrees(th1), color="#01696f",
                 lw=1.5, label="θ1")
    axes[0].plot(time, np.degrees(th2), color="#da7101",
                 lw=1.5, label="θ2")
    axes[0].set_xlabel("Tiempo [s]")
    axes[0].set_ylabel("Ángulo [°]")
    axes[0].set_title("Trayectorias angulares medidas")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Espacio de fases θ vs ω
    axes[1].plot(np.degrees(th1), np.degrees(w1),
                 color="#01696f", lw=0.8, alpha=0.7, label="θ1-ω1")
    axes[1].plot(np.degrees(th2), np.degrees(w2),
                 color="#da7101", lw=0.8, alpha=0.7, label="θ2-ω2")
    axes[1].set_xlabel("θ [°]")
    axes[1].set_ylabel("ω [°/s]")
    axes[1].set_title("Espacio de fases (retrato de fase)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # Energía total
    axes[2].plot(time, E, color="#28251d", lw=1.2)
    axes[2].axhline(E_mean, color="#01696f",
                    ls="--", lw=1.5,
                    label=f"Media = {E_mean:.3f} J")
    axes[2].fill_between(time,
                         E_mean - E_std, E_mean + E_std,
                         alpha=0.15, color="#01696f",
                         label=f"±std = {E_std:.3f} J ({E_rel:.1f}%)")
    axes[2].set_xlabel("Tiempo [s]")
    axes[2].set_ylabel("Energía total [J]")
    axes[2].set_title("Conservación de energía")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT/"plot_report.png", dpi=150)
    plt.close()

    print(f"\n[05] output/plot_report.png")
    print("═"*60 + "\n")

    # ---- NUEVO: generar PDF usando los mismos parámetros ----
    generar_pdf(L1, L2, m1, m2, g, cost_val,
                E_mean, E_std, E_rel)


if __name__ == "__main__":
    run()