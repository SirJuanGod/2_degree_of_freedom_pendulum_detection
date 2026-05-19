import mujoco
import numpy as np
import cv2

model = mujoco.MjModel.from_xml_path("model/pendulo.xml")
data  = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

WIDTH, HEIGHT = 1280, 720
FPS           = 60
cam_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "vista_frontal")
output_path   = "videos/pendulo.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video  = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))

DURACION_MAX   = 60.0
UMBRAL_VEL     = 0.005
TIEMPO_ESTABLE = 2.0
dt             = model.opt.timestep
steps_per_frame = max(1, int(1.0 / (FPS * dt)))

t_estable_inicio = None
frame_count      = 0

print(f"Grabando... (máx {DURACION_MAX}s) → {output_path}")

# ── Context manager para cerrar renderer correctamente ─────────
with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
    while data.time < DURACION_MAX:

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cam_id)
        frame_rgb = renderer.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
        frame_count += 1

        vel_max = np.max(np.abs(data.qvel[:]))
        if vel_max < UMBRAL_VEL:
            if t_estable_inicio is None:
                t_estable_inicio = data.time
            elif data.time - t_estable_inicio >= TIEMPO_ESTABLE:
                print(f"\n✅ Estabilizado en t={data.time:.2f}s")
                break
        else:
            t_estable_inicio = None

        if frame_count % FPS == 0:
            print(f"  t={data.time:.1f}s | vel_max={vel_max:.4f} rad/s")

video.release()
print(f"\n🎬 Listo → {output_path}")
print(f"   Duración: {data.time:.2f}s | Frames: {frame_count}")