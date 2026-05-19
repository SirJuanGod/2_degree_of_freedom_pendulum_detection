"""
03_identification_jax.py — Multi-shooting con JAX + Diffrax + vmap
====================================================================
Arquitectura: MULTI-SHOOTING
  — La trayectoria (62 s) se divide en ~122 ventanas de 1 s
  — Cada ventana arranca desde el estado OBSERVADO en ese instante
  — Las 122 ventanas se simulan EN PARALELO con vmap
  — El costo es el MSE medio sobre todas las ventanas

Por qué multi-shooting resuelve el problema anterior:
  — El péndulo doble es caótico: errores de params → trayectorias
    completamente distintas después de ~2-3 s (efecto mariposa)
  — Con ventanas de 1 s el gradiente está bien condicionado
  — vmap sobre ventanas = misma velocidad que simular 1 sola

Correcciones vs versión anterior (horizonte completo):
  1. Ventanas cortas de 1 s → gradientes estables, sin divergencia caótica
  2. vmap paralelo sobre todas las ventanas → igual de rápido
  3. w_geo más fuerte en fase 1 → L1, L2 no se alejan de la geo
  4. Punto inicial de masas más alto → explorar región correcta
  5. Nelder-Mead adaptive=True + 15k iter → convergencia garantizada

params = [L1, L2, m1, m2, b1, b2]   (6 parámetros)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import diffrax
import optax
import numpy as np
import yaml
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output"


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
    L1, L2, m1, m2 = [float(jnp.abs(params[k])) for k in range(4)]
    b1, b2 = float(jnp.abs(params[4])), float(jnp.abs(params[5]))
    print(f"  {label:10s}: L1={L1:.4f} L2={L2:.4f} "
          f"m1={m1:.4f} m2={m2:.4f} "
          f"b1={b1:.4f} b2={b2:.4f}")


def run():
    # ── Cargar datos ─────────────────────────────────────────
    with open(ROOT / "config" / "colors.yaml") as f:
        g_val = float(yaml.safe_load(f)["system"]["g"])

    raw = np.load(OUT / "angles.npz")
    ang = {k: raw[k] for k in raw.files}

    L1_geo = float(ang["L1_m"].item())
    L2_geo = float(ang["L2_m"].item())
    t_all  = ang["time"]          # (3750,)
    dt     = float(ang["dt"].item())

    obs_th1 = ang["theta1"]
    obs_th2 = ang["theta2"]
    obs_w1  = ang["omega1"]
    obs_w2  = ang["omega2"]

    print(f"[03-JAX] Dispositivo  : {jax.devices()[0]}")
    print(f"[03-JAX] Datos totales: {len(t_all)} pts  "
          f"({t_all[-1]:.1f} s  @  {1/dt:.0f} fps)")
    print(f"[03-JAX] L1_geo={L1_geo:.4f} m   L2_geo={L2_geo:.4f} m")
    print(f"[03-JAX] Estrategia: MULTI-SHOOTING (ventanas cortas en paralelo)\n")

    # ── Punto inicial ─────────────────────────────────────────
    # [L1,    L2,    m1,  m2,   b1,   b2  ]
    p0 = jnp.array([L1_geo, L2_geo, 1.0, 0.5, 0.05, 0.05])

    # ══════════════════════════════════════════════════════════
    # FASE 1 — Adam con ventanas de 1 s, w_geo fuerte
    #   Objetivo: convergencia inicial estable sin que L1,L2 deriven
    # ══════════════════════════════════════════════════════════
    print("[03-JAX] ── FASE 1: Adam multi-shooting (win=1s, w_geo=1.0) ──")
    cost1, n_win1 = make_cost_multishooting(
        obs_th1, obs_th2, obs_w1, obs_w2, t_all,
        L1_geo, L2_geo,
        win_secs=1.0, stride_secs=0.5,
        w_geo=1.0,    # ancla fuerte: L1, L2 cerca de estimación geométrica
    )

    print("[03-JAX] Compilando JIT+vmap (primera vez ~20 s en CPU)...")
    _ = cost1(p0)
    print("  OK\n")

    p1, losses1 = optimize_adam(cost1, p0, lr=5e-3, steps=800)

    for i in [0, 100, 300, 500, 799]:
        print(f"  step {i:4d}  loss={float(losses1[i]):.6f}")
    _print_params(p1, "Adam F1")

    # ══════════════════════════════════════════════════════════
    # FASE 2 — Adam con ventanas de 1 s, w_geo moderado
    #   Objetivo: ajustar masas y damping con más libertad
    # ══════════════════════════════════════════════════════════
    print(f"\n[03-JAX] ── FASE 2: Adam multi-shooting (win=1s, w_geo=0.3) ──")
    cost2, _ = make_cost_multishooting(
        obs_th1, obs_th2, obs_w1, obs_w2, t_all,
        L1_geo, L2_geo,
        win_secs=1.0, stride_secs=0.5,
        w_geo=0.3,
    )

    print("[03-JAX] Compilando JIT fase 2...")
    _ = cost2(p1)
    print("  OK\n")

    p2, losses2 = optimize_adam(cost2, p1, lr=2e-3, steps=800)

    for i in [0, 100, 300, 500, 799]:
        print(f"  step {i:4d}  loss={float(losses2[i]):.6f}")
    _print_params(p2, "Adam F2")

    # ══════════════════════════════════════════════════════════
    # FASE 3 — Nelder-Mead (refinamiento fino, mismo costo)
    #   Adaptive=True: escala el simplex al espacio 6D correctamente
    # ══════════════════════════════════════════════════════════
    print(f"\n[03-JAX] ── FASE 3: Nelder-Mead (refinamiento fino) ──")

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
            "adaptive": True,   # escala automática del simplex a 6D
        },
    )

    L1f, L2f, m1f, m2f = np.abs(r3.x[:4])
    b1f, b2f            = np.abs(r3.x[4]), np.abs(r3.x[5])

    # ── Resultado ─────────────────────────────────────────────
    print(f"\n[03-JAX] ══ RESULTADO FINAL ══")
    print(f"  L1 = {L1f:.6f} m         (geo video: {L1_geo:.4f} m)")
    print(f"  L2 = {L2f:.6f} m         (geo video: {L2_geo:.4f} m)")
    print(f"  m1 = {m1f:.6f} kg")
    print(f"  m2 = {m2f:.6f} kg")
    print(f"  b1 = {b1f:.6f} N·m·s/rad")
    print(f"  b2 = {b2f:.6f} N·m·s/rad")
    print(f"  m1/m2        = {m1f/m2f:.4f}")
    print(f"  Costo final  = {r3.fun:.6e}")
    print(f"  Convergencia = {r3.success}  ({r3.message})")

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

    print("\n[03-JAX] → output/identified_params.yaml guardado")


if __name__ == "__main__":
    run()