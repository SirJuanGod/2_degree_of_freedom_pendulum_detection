import mujoco
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Ruta al archivo de modelo MuJoCo
model_path = os.path.join("model", "scene.xml")

# Cargar el modelo MuJoCo
print(f"Cargando modelo desde: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print(f"\nInfo del modelo:")
print(f"  - Cuerpos: {model.nbody}")
print(f"  - Articulaciones: {model.njnt}")
print(f"  - Grados de libertad: {model.nq}")
print(f"  - Velocidades: {model.nv}")
print(f"  - Actuadores: {model.nu}")

# Número de steps a ejecutar
num_steps = 500
dt = model.opt.timestep

print(f"\nEjecutando {num_steps} pasos de simulación con renderizado...")
print(f"Timestep: {dt}s\n")

# Configurar renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# Almacenar frames para visualización
frames = []
data_history = {
    'time': [],
    'qpos': [],
    'qvel': [],
    'step': []
}

# Aplicar velocidades iniciales para mover el center
print("Aplicando condiciones iniciales...")
if model.nq > 0:
    data.qpos[0] = 0.0  # Posición inicial en x
if model.nv > 0:
    data.qvel[0] = 1.0  # Velocidad inicial en x
if model.nv > 1:
    data.qvel[1] = 0.5  # Velocidad inicial en y/rotación

print("Recopilando datos...")
for step in range(num_steps):
    # Obtener estado actual
    q = data.qpos.copy()
    qd = data.qvel.copy()
    
    # Ejecutar paso de simulación
    mujoco.mj_step(model, data)
    
    # Guardar datos cada 2 pasos
    if step % 2 == 0:
        # Renderizar frame
        renderer.update_scene(data)
        frame = renderer.render()
        frames.append(frame)
        
        # Guardar datos históricos
        data_history['time'].append(data.time)
        data_history['qpos'].append(q.copy())
        data_history['qvel'].append(qd.copy())
        data_history['step'].append(step)
    
    # Mostrar progreso
    if (step + 1) % 100 == 0:
        print(f"  Paso {step + 1}/{num_steps} completado")

print("-" * 50)
print(f"\nSimulación completada")
print(f"Pasos ejecutados: {num_steps}")
print(f"Frames guardados: {len(frames)}")
print(f"Tiempo total simulado: {data.time:.3f}s")
print(f"Estado final:")
print(f"  Posiciones (qpos): {data.qpos}")
print(f"  Velocidades (qvel): {data.qvel}")
print(f"  Aceleraciones (qacc): {data.qacc}")

# Visualizar
print("\nCreando visualización...")
fig, ax = plt.subplots(figsize=(10, 8))

def update_frame(frame_idx):
    ax.clear()
    ax.imshow(frames[frame_idx])
    ax.set_title(f"Paso: {data_history['step'][frame_idx]} | Tiempo: {data_history['time'][frame_idx]:.3f}s | Pos: {data_history['qpos'][frame_idx][0]:.4f}")
    ax.axis('off')

# Crear animación
anim = FuncAnimation(fig, update_frame, frames=len(frames), interval=50, repeat=True)

plt.tight_layout()
print("Mostrando animación (cierra la ventana para terminar)...")
plt.show()
