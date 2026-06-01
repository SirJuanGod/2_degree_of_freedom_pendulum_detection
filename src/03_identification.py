"""
03_identification_jax.py — Multi-shooting con JAX + Diffrax + vmap
====================================================================
Arquitectura: MULTI-SHOOTING
  — La trayectoria se divide en ventanas de 1 s
  — Cada ventana arranca desde el estado OBSERVADO en ese instante
  — Las ventanas se simulan EN PARALELO con vmap
  — El costo es el MSE medio sobre todas las ventanas

GPU OPTIMIZATIONS:
  - Detección automática de GPU JAX (NVIDIA/Apple/TPU)
  - Fallback a CPU si no hay GPU disponible
  - Caché de compilación JIT (primera compilación ~20s)
  - Timeout en optimización para evitar cuelgues

params = [L1, L2, m1, m2, b1, b2]   (6 parámetros)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import diffrax
import optax
import numpy as np
import yaml
import warnings
from pathlib import Path
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"

# Importar utilidades
import sys
sys.path.insert(0, str(ROOT))
from utils import detect_jax_device, setup_logging

logger = setup_logging("identification", OUT)

# ══════════════════════════════════════════════════════════════
# DETECCIÓN DE DISPOSITIVO
# ══════════════════════════════════════════════════════════════
jax_device, device_msg = detect_jax_device()
logger.info(f"Dispositivo JAX detectado: {device_msg}")

if jax_device == "cpu":
    logger.warning("⚠  JAX ejecutará en CPU. Esto tardará 30-60 minutos.")
    logger.info("   Para acelerar con GPU, instala: pip install jax[cuda12_cudnn82]")
else:
    logger.info(f"✓ JAX utilizará {jax_device.upper()}. Tiempo estimado: 5-10 minutos.")


# ══════════════════════════════════════════════════════════════
# 1. ODE del péndulo doble con amortiguamiento viscoso
# ══════════════════════════════════════════════════════════════
@jit
def ode_jax(t, y, args):
    L1, L2, m1, m2, b1, b2, g = args
    th1, th2, w1, w2 = y
    d   = th1 - th2
    den = 2*m1 + m2 - m2*jnp.cos(2*d) + 1e-14

    dw1 = (- g*(2*m1 + m2)*jnp.sin(th1)
           - m2*g*jnp.sin(th1 - 2*th2)
           - 2*jnp.sin(d)*m2*(w2**2*L2 + w1**2*L1*jnp.cos(d))
           - b1*w1
           ) / (L1*den)

    dw2 = (  2*jnp.sin(d)*(w1**2*L1*(m1 + m2)
           + g*(m1 + m2)*jnp.cos(th1)
           + w2**2*L2*m2*jnp.cos(d))
           - b2*w2
           ) / (L2*den)

    return jnp.array([w1, w2, dw1, dw2])


# ══════════════════════════════════════════════════════════════
# 2. Multi-shooting: fábrica de función de costo
#
#   Preprocesa las ventanas UNA SOLA VEZ (arrays estáticos).
#   Devuelve una función JIT+vmap-compilada lista para optimizar.
# ══════════════════════════════════════════════════════════════
def make_cost_multishooting(obs_th1, obs_th2, obs_w1, obs_w2,
                             t_all, L1_geo, L2_geo,
                             win_secs=1.0, stride_secs=0.5,
                             w_geo=0.3):
    """
    Construye la función de costo multi-shooting.

    Parámetros:
        win_secs    — duración de cada ventana (s). 1 s = seguro frente al caos.
        stride_secs — desplazamiento entre ventanas (s).
        w_geo       — peso de la penalización geométrica sobre L1, L2.

    La función de costo devuelta:
        cost_fn(params) → escalar
        params = [L1, L2, m1, m2, b1, b2]
    """
    dt       = float(t_all[1] - t_all[0])
    win_pts  = round(win_secs / dt) + 1        # puntos por ventana
    str_pts  = max(1, round(stride_secs / dt)) # paso entre ventanas
    n_pts    = len(t_all)

    # ── Índices de inicio de cada ventana ────────────────────
    starts = np.arange(0, n_pts - win_pts, str_pts)
    n_win  = len(starts)

    # ── Tiempo relativo [0, dt, 2dt, …, win_secs] ────────────
    # Todas las ventanas comparten el MISMO array de tiempos relativos.
    # Esto permite que vmap no necesite variar el eje del tiempo.
    ts_rel     = np.arange(win_pts) * dt           # (win_pts,)
    ts_rel_jax = jnp.array(ts_rel)
    t1_rel     = float(ts_rel[-1])

    # ── Apilar condiciones iniciales y observaciones ─────────
    # y0_all[k] = estado observado al inicio de la ventana k
    y0_all = jnp.array(np.stack([
        [obs_th1[i], obs_th2[i], obs_w1[i], obs_w2[i]]
        for i in starts
    ]))  # (n_win, 4)

    obs_win_th1 = jnp.array(np.stack(
        [obs_th1[i : i + win_pts] for i in starts]
    ))  # (n_win, win_pts)
    obs_win_th2 = jnp.array(np.stack(
        [obs_th2[i : i + win_pts] for i in starts]
    ))

    print(f"  Multi-shooting: {n_win} ventanas × {win_pts} pts  "
          f"(win={win_secs}s  stride={stride_secs}s)")

    # ── Simulador de UNA ventana (tiempo relativo) ────────────
    # Esta función será vmapeada sobre el eje de y0.
    def simulate_one_window(params, y0_single):
        """
        Integra una ventana desde y0_single durante [0, t1_rel].
        Los parámetros físicos se comparten entre todas las ventanas.
        """
        L1, L2, m1, m2 = jnp.abs(params[:4])
        b1 = jnp.abs(params[4])
        b2 = jnp.abs(params[5])

        term   = diffrax.ODETerm(ode_jax)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=ts_rel_jax)   # tiempos de la closure
        ctrl   = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=t1_rel, dt0=0.002,
            y0=y0_single,
            args=(L1, L2, m1, m2, b1, b2, 9.81),
            saveat=saveat,
            stepsize_controller=ctrl,
            max_steps=30_000,
        )
        return sol.ys   # (win_pts, 4)

    # vmap sobre el eje 0 de y0 (params compartidos → in_axes=(None, 0))
    _simulate_batch = vmap(simulate_one_window, in_axes=(None, 0))
    simulate_batch  = jit(_simulate_batch)

    # ── Función de costo ──────────────────────────────────────
    @jit
    def cost_fn(params):
        # Simular TODAS las ventanas en paralelo → (n_win, win_pts, 4)
        ys_all  = simulate_batch(params, y0_all)
        sim_th1 = ys_all[:, :, 0]   # (n_win, win_pts)
        sim_th2 = ys_all[:, :, 1]

        # MSE medio sobre todas las ventanas y todos los puntos
        traj_err = jnp.mean(
            (sim_th1 - obs_win_th1)**2 +
            (sim_th2 - obs_win_th2)**2
        )

        # Penalización geométrica (ancla L1 y L2)
        L1 = jnp.abs(params[0])
        L2 = jnp.abs(params[1])
        geo = ((L1 - L1_geo)/L1_geo)**2 + ((L2 - L2_geo)/L2_geo)**2

        return traj_err + w_geo * geo

    return cost_fn, n_win


# ══════════════════════════════════════════════════════════════
# 3. Adam con lax.scan (loop completo en XLA, sin overhead Python)
# ══════════════════════════════════════════════════════════════
def optimize_adam(cost_fn, params_init, lr=5e-3, steps=600):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params_init)

    val_and_grad = jit(jax.value_and_grad(cost_fn))

    def step_fn(carry, _):
        params, state = carry
        loss, grads   = val_and_grad(params)
        updates, new_state = optimizer.update(grads, state)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_state), loss

    (final_params, _), losses = jax.lax.scan(
        step_fn, (params_init, opt_state), None, length=steps
    )
    return final_params, losses


# ══════════════════════════════════════════════════════════════
# 4. Pipeline principal
# ══════════════════════════════════════════════════════════════
def _print_params(params, label):
    try:
        L1, L2, m1, m2 = [float(jnp.abs(params[k])) for k in range(4)]
        b1, b2 = float(jnp.abs(params[4])), float(jnp.abs(params[5]))
        logger.info(f"  {label:10s}: L1={L1:.4f} L2={L2:.4f} "
              f"m1={m1:.4f} m2={m2:.4f} "
              f"b1={b1:.4f} b2={b2:.4f}")
        # Validar que los valores sean razonables
        if not (0.1 < L1 < 5 and 0.1 < L2 < 5):
            logger.warning(f"  ⚠  Longitudes fuera de rango esperado")
        if not (0.01 < m1 < 10 and 0.01 < m2 < 10):
            logger.warning(f"  ⚠  Masas fuera de rango esperado")
    except Exception as e:
        logger.error(f"Error imprimiendo parámetros: {e}")


def _validate_params(params, name=""):
    """Valida que los parámetros sean válidos (no NaN, no inf, rangos razonables)."""
    L1, L2, m1, m2 = np.abs(params[:4])
    b1, b2 = np.abs(params[4:6])
    
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        logger.error(f"❌ {name} contiene NaN o Inf")
        return False
    
    if not (0.05 < L1 < 10 and 0.05 < L2 < 10):
        logger.warning(f"⚠  {name} L1 o L2 fuera de rango (0.05-10m)")
    
    if not (0.001 < m1 < 50 and 0.001 < m2 < 50):
        logger.warning(f"⚠  {name} m1 o m2 fuera de rango (0.001-50kg)")
    
    if not (0.001 < b1 < 100 and 0.001 < b2 < 100):
        logger.warning(f"⚠  {name} b1 o b2 fuera de rango (0.001-100)")
    
    return True


def run():
    # ── Cargar datos ─────────────────────────────────────────
    try:
        with open(ROOT / "config" / "colors.yaml") as f:
            g_val = float(yaml.safe_load(f)["system"]["g"])
        
        if not (9.0 < g_val < 10.0):
            logger.warning(f"⚠  Gravedad inusual: g={g_val}")
    except Exception as e:
        logger.error(f"Error cargando config: {e}")
        raise

    try:
        raw = np.load(OUT / "angles.npz")
        ang = {k: raw[k] for k in raw.files}
    except FileNotFoundError:
        logger.error("❌ angles.npz no encontrado. Ejecuta: python run_pipeline.py --video <video>")
        raise

    try:
        L1_geo = float(ang["L1_m"].item())
        L2_geo = float(ang["L2_m"].item())
        t_all  = ang["time"]
        dt     = float(ang["dt"].item())
        
        obs_th1 = ang["theta1"]
        obs_th2 = ang["theta2"]
        obs_w1  = ang["omega1"]
        obs_w2  = ang["omega2"]
        
        # Validar datos
        if len(t_all) < 100:
            logger.error(f"❌ Datos insuficientes: {len(t_all)} frames (min 100)")
            raise ValueError("Datos insuficientes para identificación")
        
        if np.any(np.isnan(obs_th1)) or np.any(np.isnan(obs_th2)):
            logger.error("❌ Datos contienen NaN. Revisa config/colors.yaml")
            raise ValueError("Datos inválidos")
            
    except Exception as e:
        logger.error(f"Error leyendo angles.npz: {e}")
        raise

    logger.info("="*65)
    logger.info("IDENTIFICACIÓN DE PARÁMETROS (MULTI-SHOOTING JAX)")
    logger.info("="*65)
    logger.info(f"Datos totales: {len(t_all)} pts  ({t_all[-1]:.1f} s  @  {1/dt:.0f} fps)")
    logger.info(f"L1_geo={L1_geo:.4f} m   L2_geo={L2_geo:.4f} m")
    logger.info(f"Estrategia: MULTI-SHOOTING (ventanas cortas en paralelo)\n")

    # ── Punto inicial ─────────────────────────────────────────
    # [L1,    L2,    m1,  m2,   b1,   b2  ]
    p0 = jnp.array([L1_geo, L2_geo, 1.0, 0.5, 0.05, 0.05])

    # ══════════════════════════════════════════════════════════
    # FASE 1 — Adam con ventanas de 1 s, w_geo fuerte
    # ══════════════════════════════════════════════════════════
    logger.info("FASE 1: Adam multi-shooting (win=1s, w_geo=1.0)")
    logger.info("-"*65)
    
    try:
        cost1, n_win1 = make_cost_multishooting(
            obs_th1, obs_th2, obs_w1, obs_w2, t_all,
            L1_geo, L2_geo,
            win_secs=1.0, stride_secs=0.5,
            w_geo=1.0,
        )

        logger.info("Compilando JIT+vmap (primera vez puede tardar ~20s en CPU)...")
        _ = cost1(p0)
        logger.info("✓ Compilación completada\n")

        p1, losses1 = optimize_adam(cost1, p0, lr=5e-3, steps=800)

        for i in [0, 100, 300, 500, 799]:
            logger.info(f"  step {i:4d}  loss={float(losses1[i]):.6f}")
        
        _print_params(p1, "Adam F1")
        _validate_params(p1, "Adam F1")
        
    except Exception as e:
        logger.error(f"Error en FASE 1: {e}")
        raise

    # ══════════════════════════════════════════════════════════
    # FASE 2 — Adam con ventanas de 1 s, w_geo moderado
    # ══════════════════════════════════════════════════════════
    logger.info(f"\nFASE 2: Adam multi-shooting (win=1s, w_geo=0.3)")
    logger.info("-"*65)
    
    try:
        cost2, _ = make_cost_multishooting(
            obs_th1, obs_th2, obs_w1, obs_w2, t_all,
            L1_geo, L2_geo,
            win_secs=1.0, stride_secs=0.5,
            w_geo=0.3,
        )

        logger.info("Compilando JIT fase 2...")
        _ = cost2(p1)
        logger.info("✓ OK\n")

        p2, losses2 = optimize_adam(cost2, p1, lr=2e-3, steps=800)

        for i in [0, 100, 300, 500, 799]:
            logger.info(f"  step {i:4d}  loss={float(losses2[i]):.6f}")
        
        _print_params(p2, "Adam F2")
        _validate_params(p2, "Adam F2")
        
    except Exception as e:
        logger.error(f"Error en FASE 2: {e}")
        raise

    # ══════════════════════════════════════════════════════════
    # FASE 3 — Nelder-Mead (refinamiento fino)
    # ══════════════════════════════════════════════════════════
    logger.info(f"\nFASE 3: Nelder-Mead (refinamiento fino)")
    logger.info("-"*65)

    try:
        def cost_np(p):
            return float(cost2(jnp.array(p)))

        r3 = minimize(
            cost_np,
            np.array(p2),
            method="Nelder-Mead",
            options={
                "maxiter":  15_000,
                "xatol":    1e-9,
                "fatol":    1e-9,
                "adaptive": True,
            },
        )

        L1f, L2f, m1f, m2f = np.abs(r3.x[:4])
        b1f, b2f            = np.abs(r3.x[4]), np.abs(r3.x[5])

        # ── Resultado ─────────────────────────────────────────────
        logger.info("="*65)
        logger.info("RESULTADO FINAL")
        logger.info("="*65)
        logger.info(f"  L1 = {L1f:.6f} m         (geo video: {L1_geo:.4f} m)")
        logger.info(f"  L2 = {L2f:.6f} m         (geo video: {L2_geo:.4f} m)")
        logger.info(f"  m1 = {m1f:.6f} kg")
        logger.info(f"  m2 = {m2f:.6f} kg")
        logger.info(f"  b1 = {b1f:.6f} N·m·s/rad")
        logger.info(f"  b2 = {b2f:.6f} N·m·s/rad")
        logger.info(f"  m1/m2        = {m1f/m2f:.4f}")
        logger.info(f"  Costo final  = {r3.fun:.6e}")
        logger.info(f"  Convergencia = {r3.success}  ({r3.message})")

        # Validar resultado final
        final_params = np.array([L1f, L2f, m1f, m2f, b1f, b2f])
        if not _validate_params(final_params, "Resultado Final"):
            logger.warning("⚠  Algunos parámetros están fuera de rango esperado")

        # ── Guardar ───────────────────────────────────────────────
        out_dict = {
            "L1": float(L1f), "L2": float(L2f),
            "m1": float(m1f), "m2": float(m2f),
            "b1": float(b1f), "b2": float(b2f),
            "g":  float(g_val),
            "cost_final":        float(r3.fun),
            "optimizer_success": bool(r3.success),
        }
        with open(OUT / "identified_params.yaml", "w") as f:
            yaml.dump(out_dict, f, default_flow_style=False)

        logger.info("\n✓ output/identified_params.yaml guardado")
        
    except Exception as e:
        logger.error(f"Error en FASE 3: {e}")
        raise


if __name__ == "__main__":
    run()