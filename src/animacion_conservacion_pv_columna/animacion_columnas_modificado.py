from dataclasses import dataclass, field
from pathlib import Path
import os

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pyvista as pv

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
CONFIG = {
    # IMPORTANTE: Cambia esta ruta a una carpeta válida en tu ordenador
    "ruta_salida": Path(r"C:\Users\guill\Desktop\Guillermo\Aeroespacial\Cursos\4º - Cuarto VAMOOO Q ACABMOO\TFG\CodigoTFG\Columnitas"),
    "fps": 24,
    "duracion_seg": 25,
    "n_columnas": 5,
    "f_coriolis": 1e-4,
    "volumen_ref": 10.0,
    "h_media": 5.0,
    "amp_h": 2.0,
    "resolucion": (1920, 1080), # Resolución por defecto para vídeos finales
    "resolucion_qhd": (2560, 1440),
    "indice_destacado": 0,
    "k_repulsion": 9.5,
    "c_repulsion": 2.0,
    "k_centro": 1.15,
    "c_arrastre": 0.85,
    "margen_contacto": 0.12,
    "angulo_desde_vertical_deg": 60.0,
    "azimut_deg": -45.0,
    "distancia_camara": 21.0,
    "separacion_minima": 0.04,
    "desplazamiento_escena_derecha": 0.0,
    "z_centro_comun": 0.0,  
    
    # --- AJUSTES DE ENCUADRE MODIFICADOS PARA CORREGIR DESPLAZAMIENTO ---
    "hud_ancho_reservado_individual": 0.30, # REDUCIDO (antes 0.50): Mueve la columna indiv. hacia el centro
    "factor_acercamiento_individual": 1.15, # Mantiene zoom alejado
    "hud_ancho_reservado_grupal": 0.30,     # REDUCIDO (antes 0.50): Mueve el grupo hacia el centro
    "factor_acercamiento_grupal": 1.25,     # Mantiene zoom alejado
    "margen_derecho_escena": 0.05,
    "margen_superior_escena": 0.13, 
    "margen_inferior_escena": 0.10,
    "margen_izquierdo_extra_escena": 0.0,
    "escala_giro_visual": 12000.0,
    "bg_lower": "#0B132B",
    "bg_upper": "#1C2541",

    # --- PARÁMETROS DE CONTRASTE CROMÁTICO EXTREMO ---
    "color_lento_circundante": "#7A8088",  
    "color_rapido_circundante": "#F4F7FB", 
    "color_lento_central": "#0B4F9C",      
    "color_rapido_central": "#00F5D4",     
    "opacidad_circundantes": 0.65,
    "opacidad_central": 0.95,

    # --- CONFIGURACIÓN ESTÉTICA DEL HUD ---
    "hud_panel_color": "#0F172A",
    "hud_text_color": "#F8FAFC",
    "hud_accent_color": "#38BDF8",
    "hud_pos_x": 40,          
    "hud_pos_y_top": 40,      
    "hud_ancho": 950,         
    "hud_padding": 35,
    "hud_font_title": 26,     
    "hud_font_sub": 19,
    "hud_font_norm": 17,
    "hud_line_h": 45, # Altura generosa entre filas de la tabla         
}

PERFILES_RENDER = {
    "rapido": {
        "nombre": "Rápido Pruebas (720p)",
        "resolucion_preview": (1280, 720),
        "resolucion_video": (1280, 720),
        "fps": 10,
        "duracion_seg": 6,
        "calidad_video": 4,
        "macro_block_size": 2,
        "sufijo_archivo": "_preview",
    },
    "alta_calidad": {
        "nombre": "Alta Calidad Resultado Final (1080p)",
        "resolucion_preview": (1280, 720),
        "resolucion_video": (1920, 1080),
        "fps": 24,
        "duracion_seg": 25,
        "calidad_video": 8,
        "macro_block_size": 1,
        "sufijo_archivo": "",
    },
}

CONFIG_ACTIVA: dict = {}

def seleccionar_modo_exportacion() -> str:
    print("\n" + "=" * 70)
    print("SELECCIONAR MODO DE EXPORTACIÓN")
    print("=" * 70)
    print("\nOpciones disponibles:")
    print("  [1] Rápido / Pruebas (720p, baja FPS)")
    print("  [2] Alta calidad / Resultado final (1080p, 24 FPS)")
    
    while True:
        opcion = input("\nIngresa tu opción (1 o 2): ").strip()
        if opcion == "1": return "rapido"
        elif opcion == "2": return "alta_calidad"
        else: print("❌ Opción inválida. Por favor, ingresa 1 o 2.")

def obtener_fuente_elegante() -> str | None:
    candidatas = [
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\seguisym.ttf"),
        Path(r"C:\Windows\Fonts\consola.ttf"),
    ]
    for fuente in candidatas:
        if fuente.exists(): return str(fuente)
    return None

HUD_FONT_FILE = obtener_fuente_elegante()

@dataclass
class ColumnaGeofisica:
    indice: int
    posicion: np.ndarray
    fase: float
    frecuencia: float
    amp_local: float
    q_cte: float
    velocidad: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    fuerza: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    r: float = 1.0
    h: float = 1.0
    omega: float = 0.0
    zeta: float = 0.0
    theta_visual: float = 0.0

    def actualizar_estado(self, t: float) -> None:
        self.h = CONFIG["h_media"] + self.amp_local * np.sin(self.frecuencia * t + self.fase)
        self.h = max(self.h, 1e-6)
        self.r = np.sqrt(CONFIG["volumen_ref"] / (np.pi * self.h))
        self.zeta = self.q_cte * self.h - CONFIG["f_coriolis"]
        self.omega = 0.5 * self.zeta

    def integrar(self, dt: float) -> None:
        self.velocidad += self.fuerza * dt
        self.posicion += self.velocidad * dt
        self.theta_visual += self.omega * dt * CONFIG.get("escala_giro_visual", 12000.0)
        self.fuerza[:] = 0.0

def hex_a_rgb_f(hex_color: str) -> tuple[float, float, float]:
    valor = hex_color.lstrip("#")
    return tuple(int(valor[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

def color_por_estado_columna(col: ColumnaGeofisica, destacada: bool = False) -> tuple[str, float]:
    h_media = CONFIG["h_media"]
    amp = col.amp_local
    x_lineal = (col.h - (h_media - amp)) / (2.0 * amp)
    x_lineal = float(np.clip(x_lineal, 0.0, 1.0))
    
    if destacada:
        c_l = CONFIG.get("color_lento_central", "#0B4F9C")  
        c_r = CONFIG.get("color_rapido_central", "#00F5D4") 
        k = 12.0   
        x0 = 0.5  
        v = 1.0 / (1.0 + np.exp(-k * (x_lineal - x0)))
        v_min = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
        v_max = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
        intensidad = (v - v_min) / (v_max - v_min)
    else:
        c_l = CONFIG.get("color_lento_circundante", "#7A8088")
        c_r = CONFIG.get("color_rapido_circundante", "#F4F7FB")
        intensidad = x_lineal 
    
    intensidad = float(np.clip(intensidad, 0.0, 1.0))
    rgb_l, rgb_r = np.array(hex_a_rgb_f(c_l)), np.array(hex_a_rgb_f(c_r))
    color = rgb_l * (1.0 - intensidad) + rgb_r * intensidad
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255.0)) for c in np.clip(color, 0.0, 1.0))), intensidad

def dibujar_eje_discontinuo(plotter: pv.Plotter, col: ColumnaGeofisica) -> None:
    z_center = CONFIG["z_centro_comun"]
    z_min = z_center - col.h * 0.65
    z_max = z_center + col.h * 0.65
    n_dashes = 14
    z_points = np.linspace(z_min, z_max, n_dashes * 2)
    
    for i in range(0, len(z_points)-1, 2):
        z_mid = (z_points[i] + z_points[i+1]) / 2.0
        h_dash = abs(z_points[i+1] - z_points[i]) * 0.6 
        cil_dash = pv.Cylinder(center=(col.posicion[0], col.posicion[1], z_mid), direction=(0, 0, 1), radius=0.012, height=h_dash, resolution=12)
        plotter.add_mesh(cil_dash, color="#E2E8F0", ambient=0.6, pbr=False)

def dibujar_rod_y_rings(plotter: pv.Plotter, col: ColumnaGeofisica) -> None:
    max_omega = 0.5 * abs(CONFIG["f_coriolis"]) * (CONFIG["amp_h"] / CONFIG["h_media"])
    omega_ratio = abs(col.omega) / (max_omega + 1e-12)
    
    z_center = CONFIG["z_centro_comun"]
    pos_top = z_center + 0.5 * col.h
    
    rod = pv.Cylinder(center=(col.posicion[0], col.posicion[1], pos_top + 0.7 * col.r), direction=(0, 0, 1), radius=0.04 * col.r, height=1.6 * col.r, resolution=32)
    plotter.add_mesh(rod, color="#AAAAAA", pbr=True, metallic=0.8, roughness=0.2, specular=0.8, smooth_shading=True)
    
    thresholds = [0.0, 0.4, 0.7] 
    base_offset = 0.3 * col.r 
    stack_spacing = 0.4 * col.r 
    white_hex = "#FFFFFF" 
    giro = col.theta_visual
    signo = 1 if col.omega >= 0 else -1
    
    if signo >= 0:
        angulo_inicio = giro + np.radians(25)
        angulo_fin = giro + np.radians(335)
    else:
        angulo_inicio = giro - np.radians(25)
        angulo_fin = giro - np.radians(335)
    
    for i in range(3):
        if omega_ratio >= thresholds[i]:
            offset = base_offset + i * stack_spacing
            puntos_theta = np.linspace(angulo_inicio, angulo_fin, 90)
            path = pv.Spline(points=np.column_stack((col.posicion[0] + 0.35 * col.r * np.cos(puntos_theta), col.posicion[1] + 0.35 * col.r * np.sin(puntos_theta), np.full(90, pos_top + offset))))
            tube = path.tube(radius=0.03 * col.r)
            plotter.add_mesh(tube, color=white_hex, opacity=1.0, smooth_shading=True, pbr=False, roughness=1.0, ambient=0.5) 
            
            tip_pos = np.array([col.posicion[0] + 0.35 * col.r * np.cos(angulo_fin), col.posicion[1] + 0.35 * col.r * np.sin(angulo_fin), pos_top + offset])
            dir_vec = np.array([-np.sin(angulo_fin) * signo, np.cos(angulo_fin) * signo, 0.0])
            norm = np.linalg.norm(dir_vec)
            if norm > 0: dir_vec /= norm
            
            height_cone = 0.15 * col.r
            center_cone = tip_pos + dir_vec * (height_cone / 2.0)
            cone = pv.Cone(center=center_cone, direction=dir_vec, radius=0.06 * col.r, height=height_cone, resolution=16)
            plotter.add_mesh(cone, color=white_hex, opacity=1.0, smooth_shading=True, pbr=False, ambient=0.5)

def intensidad_compresion_central(col_central: ColumnaGeofisica) -> float:
    h_media = CONFIG["h_media"]
    amp_local = col_central.amp_local
    h_max = h_media + amp_local
    h_min = h_media - amp_local
    r_min = np.sqrt(CONFIG["volumen_ref"] / (np.pi * h_max))
    r_max = np.sqrt(CONFIG["volumen_ref"] / (np.pi * h_min))
    intensidad = 1.0 - (col_central.r - r_min) / max(r_max - r_min, 1e-6)
    return float(np.clip(intensidad, 0.0, 1.0))

def dibujar_flechas_compresion(plotter: pv.Plotter, columnas: list[ColumnaGeofisica], t: float) -> None:
    idx_central = CONFIG["indice_destacado"]
    col_central = columnas[idx_central]
    intensidad = intensidad_compresion_central(col_central)
    c_low = np.array(hex_a_rgb_f("#8B0000"))
    c_high = np.array(hex_a_rgb_f("#FF0000"))
    color_rgb = c_low * (1.0 - intensidad) + c_high * intensidad
    color_hex = "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255.0)) for c in np.clip(color_rgb, 0.0, 1.0)))
    
    opacidad = 0.85 + 0.15 * intensidad 
    longitud_base = 0.6 + 0.6 * intensidad   
    grosor_shaft = 0.08 + 0.04 * intensidad  
    grosor_tip = 0.18 + 0.08 * intensidad    
    
    z_mid = CONFIG["z_centro_comun"]
    radio_ref = np.sqrt(CONFIG["volumen_ref"] / (np.pi * CONFIG["h_media"]))
    
    for col in columnas:
        if col.indice == idx_central: continue
        delta = col_central.posicion - col.posicion
        dist = np.linalg.norm(delta)
        if dist < 1e-6: continue
        dir_vec = delta / dist
        
        longitud_real = radio_ref * longitud_base
        margen = 0.15 * radio_ref
        punto_fin = col_central.posicion - dir_vec * (col_central.r + margen)
        punto_inicio = punto_fin - dir_vec * longitud_real
        
        flecha = pv.Arrow(start=(punto_inicio[0], punto_inicio[1], z_mid), direction=(dir_vec[0], dir_vec[1], 0.0), scale=longitud_real, tip_length=0.3, tip_radius=grosor_tip, shaft_radius=grosor_shaft, tip_resolution=30, shaft_resolution=30)
        plotter.add_mesh(flecha, color=color_hex, opacity=opacidad, ambient=0.5, pbr=False, smooth_shading=True)

MAPA_RUIDO = None
def obtener_mapa_ruido(size: int = 512) -> np.ndarray:
    global MAPA_RUIDO
    if MAPA_RUIDO is None:
        x = np.linspace(0.0, 4.0 * np.pi, size)
        y = np.linspace(0.0, 4.0 * np.pi, size)
        xx, yy = np.meshgrid(x, y)
        ondas = 0.55 * np.sin(xx + 1.3 * yy) + 0.45 * np.cos(1.7 * xx - 0.8 * yy)
        ruido = 0.25 * np.sin(3.2 * xx + 2.1 * yy)
        mapa = ondas + ruido
        MAPA_RUIDO = (mapa - mapa.min()) / (mapa.max() - mapa.min())
    return MAPA_RUIDO

def generar_textura_tintada(hex_color: str) -> pv.Texture:
    mapa = obtener_mapa_ruido()
    rgb = np.array(hex_a_rgb_f(hex_color))
    factor = 0.5 + 0.5 * mapa
    r = (rgb[0] * factor * 255).astype(np.uint8)
    g = (rgb[1] * factor * 255).astype(np.uint8)
    b = (rgb[2] * factor * 255).astype(np.uint8)
    arr = np.ascontiguousarray(np.dstack([r, g, b]))
    return pv.numpy_to_texture(arr)

def preparar_datos_hud(col: ColumnaGeofisica, t: float, titulo: str = "PANEL DE CONTROL FÍSICO", subtitulo: str = "COLUMNA CENTRAL") -> dict:
    color_hex, intensidad = color_por_estado_columna(col, destacada=True)
    intensidad_comp = intensidad_compresion_central(col)
    
    return {
        "titulo": titulo.upper(),
        "subtitulo": subtitulo.upper(),
        "estado": "MODO: SHALLOW-WATER / PV CONSERVADA",
        "filas": [
            ("Tiempo de simulación", "t", f"{t:6.2f}", "s"),
            ("Altura de columna", "h", f"{col.h:6.3f}", "m"),
            ("Radio efectivo", "r", f"{col.r:6.3f}", "m"),
            ("Velocidad angular", "ω", f"{col.omega:.3e}", "rad/s"),
            ("Vorticidad relativa", "ζ", f"{col.zeta:.3e}", "s⁻¹"),
            ("Vorticidad potencial", "q", f"{col.q_cte:.3e}", "m⁻¹s⁻¹"),
            ("Compresión lateral", "C_lat", f"{np.clip(intensidad_comp*100,0,100):5.1f}", "%"),
            ("Intensidad cromática", "I", f"{intensidad*100:5.1f}", "%"),
        ],
    }

# =============================================================================
# FUNCION MAESTRA DE DIBUJO DEL HUD (ESCALADO DINÁMICO)
# =============================================================================
def dibujar_hud_academico(plotter: pv.Plotter, datos: dict, name: str = "hud_panel") -> None:
    win_w, win_h = plotter.window_size
    
    # FACTOR DE ESCALA: Asegura que el panel sea idéntico en todas las resoluciones
    scale = win_h / 720.0 
    
    pos_x = int(CONFIG.get("hud_pos_x", 40) * scale)
    pos_y_top = win_h - int(CONFIG.get("hud_pos_y_top", 40) * scale)
    padding = int(CONFIG.get("hud_padding", 35) * scale)
    
    f_tit = int(CONFIG.get("hud_font_title", 26) * scale)
    f_sub = int(CONFIG.get("hud_font_sub", 19) * scale)
    f_norm = int(CONFIG.get("hud_font_norm", 17) * scale)
    line_h = int(CONFIG.get("hud_line_h", 45) * scale)
    
    n_filas = len(datos["filas"])
    
    # --- CÁLCULO DE POSICIONES VERTICALES MEDIANTE "CURSOR" ---
    curr_y = pos_y_top - padding
    
    y_tit = curr_y
    curr_y -= (f_tit + int(15 * scale))
    
    y_sub = curr_y
    curr_y -= (f_sub + int(15 * scale))
    
    y_mod = curr_y
    curr_y -= (f_norm + int(10 * scale))
    
    y_sep = curr_y
    curr_y -= (f_norm + int(15 * scale))
    
    y_head = curr_y
    curr_y -= (f_norm + int(20 * scale))
    
    start_y = curr_y
    
    # --- FONDO DE LA TABLA ---
    bg_text = "\n".join(["█" * 62 for _ in range(24 + n_filas * 2)])
    pos_y_bg = start_y - (n_filas * line_h) - int(40 * scale)

    try:
        plotter.add_text(bg_text, position=(pos_x - int(15 * scale), pos_y_bg), font_size=int(18 * scale), color=CONFIG.get("hud_panel_color", "#0F172A"), font_file=HUD_FONT_FILE, name=f"{name}_bg", viewport=False, shadow=False)
    except Exception:
        plotter.add_text(bg_text, position=(pos_x - int(15 * scale), pos_y_bg), font_size=int(18 * scale), color=CONFIG.get("hud_panel_color", "#0F172A"), name=f"{name}_bg", viewport=False, shadow=False)

    # --- TEXTOS DE CABECERA Y SEPARADOR ---
    plotter.add_text(datos["titulo"], position=(pos_x + padding, y_tit), font_size=f_tit, color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE, name=f"{name}_tit", viewport=False, shadow=True)
    plotter.add_text(datos["subtitulo"], position=(pos_x + padding, y_sub), font_size=f_sub, color=CONFIG.get("hud_text_color", "#F8FAFC"), font_file=HUD_FONT_FILE, name=f"{name}_sub", viewport=False)
    plotter.add_text(datos["estado"], position=(pos_x + padding, y_mod), font_size=max(1, f_norm - int(2 * scale)), color="#94A3B8", font_file=HUD_FONT_FILE, name=f"{name}_mod", viewport=False)
    plotter.add_text("⎯" * 50, position=(pos_x + padding, y_sep), font_size=f_norm, color="#334155", font_file=HUD_FONT_FILE, name=f"{name}_sep", viewport=False)

    # --- CÁLCULO DE POSICIONES HORIZONTALES (Escaladas dinámicamente) ---
    x_mag = pos_x + padding
    x_sim = pos_x + int(290 * scale)
    x_val = pos_x + int(400 * scale)
    x_uni = pos_x + int(500 * scale)
    
    plotter.add_text("MAGNITUD", position=(x_mag, y_head), font_size=max(1, f_norm - 1), color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE, name=f"{name}_h1", viewport=False)
    plotter.add_text("SÍMBOLO", position=(x_sim, y_head), font_size=max(1, f_norm - 1), color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE, name=f"{name}_h2", viewport=False)
    plotter.add_text("VALOR", position=(x_val, y_head), font_size=max(1, f_norm - 1), color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE, name=f"{name}_h3", viewport=False)
    plotter.add_text("UNIDAD", position=(x_uni, y_head), font_size=max(1, f_norm - 1), color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE, name=f"{name}_h4", viewport=False)

    # --- DIBUJAR DATOS ---
    for i, fila in enumerate(datos["filas"]):
        y_i = start_y - i * line_h
        plotter.add_text(fila[0], position=(x_mag, y_i), font_size=f_norm, color="#CBD5E1", font_file=HUD_FONT_FILE, name=f"{name}_lbl_{i}", viewport=False)
        plotter.add_text(fila[1], position=(x_sim, y_i), font_size=f_norm, color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE, name=f"{name}_sym_{i}", viewport=False)
        plotter.add_text(fila[2], position=(x_val, y_i), font_size=f_norm, color=CONFIG.get("hud_text_color", "#F8FAFC"), font_file=HUD_FONT_FILE, name=f"{name}_val_{i}", viewport=False)
        plotter.add_text(fila[3], position=(x_uni, y_i), font_size=f_norm, color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE, name=f"{name}_uni_{i}", viewport=False)

# =============================================================================
# PIPELINE Y RENDERS DE CÁMARA
# =============================================================================

def validar_configuracion() -> None:
    if CONFIG["fps"] <= 0: raise ValueError("CONFIG['fps'] debe ser mayor que cero.")
    if CONFIG["duracion_seg"] < 20: raise ValueError("CONFIG['duracion_seg'] debe ser al menos 20 s.")
    if CONFIG["n_columnas"] != 5: raise ValueError("CONFIG['n_columnas'] debe ser exactamente 5.")

def configurar_ffmpeg_imageio() -> str:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe
    return ffmpeg_exe

def inicializar_columnas() -> list[ColumnaGeofisica]:
    radio_ref = np.sqrt(CONFIG["volumen_ref"] / (np.pi * CONFIG["h_media"]))
    paso = 2.35 * radio_ref
    posiciones = [
        np.array([0.0, 0.0]),
        np.array([-paso, 0.0]),
        np.array([paso, 0.0]),
        np.array([0.0, paso]),
        np.array([0.0, -paso]),
    ]
    shift_x = float(CONFIG.get("desplazamiento_escena_derecha", 0.0))
    posiciones = [p + np.array([shift_x, 0.0]) for p in posiciones]
    q_cte = CONFIG["f_coriolis"] / CONFIG["h_media"]
    columnas = []
    for k, pos in enumerate(posiciones):
        col = ColumnaGeofisica(
            indice=k, posicion=pos.copy(), fase=0.75 * k, frecuencia=0.48 + 0.07 * k,
            amp_local=CONFIG["amp_h"] * (0.78 + 0.06 * k), q_cte=q_cte,
        )
        col.actualizar_estado(0.0)
        columnas.append(col)
    return columnas

def resolver_contactos(columnas: list[ColumnaGeofisica], dt: float) -> None:
    centro_objetivo = np.mean([c.posicion for c in columnas], axis=0)
    for col in columnas:
        col.fuerza += -CONFIG["k_centro"] * (col.posicion - centro_objetivo)
        col.fuerza += -CONFIG["c_arrastre"] * col.velocidad
    for i in range(len(columnas)):
        for j in range(i + 1, len(columnas)):
            a, b = columnas[i], columnas[j]
            delta = b.posicion - a.posicion
            dist = max(np.linalg.norm(delta), 1e-8)
            direccion = delta / dist
            contacto = a.r + b.r
            umbral = contacto + CONFIG["margen_contacto"]
            if dist < umbral:
                penetracion = max(contacto - dist, 0.0)
                vel_rel = np.dot((b.velocidad - a.velocidad), direccion)
                fuerza_n = max(CONFIG["k_repulsion"] * penetracion - CONFIG["c_repulsion"] * vel_rel, 0.0)
                fuerza = fuerza_n * direccion
                a.fuerza -= fuerza
                b.fuerza += fuerza
    for col in columnas:
        col.integrar(dt)
    evitar_intersecciones(columnas, iteraciones=5)

def evitar_intersecciones(columnas: list[ColumnaGeofisica], iteraciones: int = 4) -> None:
    separacion = CONFIG["separacion_minima"]
    for _ in range(iteraciones):
        for i in range(len(columnas)):
            for j in range(i + 1, len(columnas)):
                a, b = columnas[i], columnas[j]
                delta = b.posicion - a.posicion
                dist = max(np.linalg.norm(delta), 1e-9)
                direccion = delta / dist
                distancia_objetivo = a.r + b.r + separacion
                if dist < distancia_objetivo:
                    correccion = 0.5 * (distancia_objetivo - dist) * direccion
                    a.posicion -= correccion
                    b.posicion += correccion

def crear_plotter(window_size: tuple[int, int] | None = None) -> pv.Plotter:
    pv.set_plot_theme("document")
    pv.global_theme.smooth_shading = True
    resolucion_default = CONFIG_ACTIVA.get("resolucion_video", CONFIG["resolucion"])
    plotter = pv.Plotter(off_screen=True, window_size=resolucion_default if window_size is None else window_size)
    plotter.enable_depth_peeling(10)
    plotter.set_background(CONFIG.get("bg_lower", "#0B132B"), top=CONFIG.get("bg_upper", "#1C2541"))
    plotter.add_light(pv.Light(position=(10, -12, 18), focal_point=(0, 0, 2), intensity=1.15, light_type="scene light"))
    plotter.add_light(pv.Light(position=(-12, 8, 10), focal_point=(0, 0, 2), intensity=0.55, light_type="scene light"))
    plotter.add_light(pv.Light(position=(0, 15, 12), focal_point=(0, 0, 3), intensity=0.45, light_type="scene light"))
    plotter.add_light(pv.Light(intensity=0.18, light_type="headlight"))
    plotter.camera.up = (0, 0, 1)
    return plotter

def configurar_camara(plotter: pv.Plotter, foco_xyz: tuple[float, float, float] = (0.0, 0.0, 3.0), distancia: float | None = None) -> None:
    elev = np.deg2rad(90.0 - CONFIG["angulo_desde_vertical_deg"])
    azim = np.deg2rad(CONFIG["azimut_deg"])
    dist = CONFIG["distancia_camara"] if distancia is None else distancia
    foco_x, foco_y, foco_z = foco_xyz
    radio_horizontal = dist * np.cos(elev)
    x = foco_x + radio_horizontal * np.cos(azim)
    y = foco_y + radio_horizontal * np.sin(azim)
    z = foco_z + dist * np.sin(elev)
    plotter.camera.position = (x, y, z)
    plotter.camera.focal_point = (foco_x, foco_y, foco_z)
    plotter.camera.clipping_range = (0.1, 500.0)

def calcular_bounds_columna(col: ColumnaGeofisica) -> tuple[float, float, float, float, float, float]:
    z_center = CONFIG["z_centro_comun"]
    return (float(col.posicion[0] - col.r), float(col.posicion[0] + col.r), float(col.posicion[1] - col.r), float(col.posicion[1] + col.r), float(z_center - 0.5 * col.h), float(z_center + 0.5 * col.h))

def calcular_bounds_columnas(columnas: list[ColumnaGeofisica]) -> tuple[float, float, float, float, float, float]:
    xmin = ymin = zmin = np.inf
    xmax = ymax = zmax = -np.inf
    for col in columnas:
        cxmin, cxmax, cymin, cymax, czmin, czmax = calcular_bounds_columna(col)
        xmin, xmax = min(xmin, cxmin), max(xmax, cxmax)
        ymin, ymax = min(ymin, cymin), max(ymax, cymax)
        zmin, zmax = min(zmin, czmin), max(zmax, czmax)
    return float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)

def calcular_foco_y_distancia_con_margenes(
    bounds_xyz: tuple[float, float, float, float, float, float], *, resolucion: tuple[int, int] | None = None,
    hud_ancho_reservado: float | None = None, margen_derecho_escena: float | None = None,
    margen_superior_escena: float | None = None, margen_inferior_escena: float | None = None,
    margen_izquierdo_extra_escena: float | None = None, camera_view_angle_deg: float | None = None, factor_acercamiento: float = 1.0,
) -> tuple[tuple[float, float, float], float]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds_xyz
    centro = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5], dtype=float)

    if resolucion is None: resolucion = CONFIG_ACTIVA.get("resolucion_video", CONFIG["resolucion"])
    if camera_view_angle_deg is None: camera_view_angle_deg = 30.0

    # Usamos valores por defecto si son None
    left_reserved = float(hud_ancho_reservado if hud_ancho_reservado is not None else 0.36)
    left_reserved += float(margen_izquierdo_extra_escena if margen_izquierdo_extra_escena is not None else 0.0)
    right_reserved = float(margen_derecho_escena if margen_derecho_escena is not None else 0.06)
    top_reserved = float(margen_superior_escena if margen_superior_escena is not None else 0.08)
    bottom_reserved = float(margen_inferior_escena if margen_inferior_escena is not None else 0.10)

    usable_w = max(1e-6, 1.0 - left_reserved - right_reserved)
    usable_h = max(1e-6, 1.0 - top_reserved - bottom_reserved)
    
    # Objetivo: centrar el modelo en el espacio "usable"
    target_x = left_reserved + 0.5 * usable_w
    target_y = bottom_reserved + 0.5 * usable_h

    aspect = float(resolucion[0]) / float(resolucion[1])
    vfov = np.deg2rad(float(camera_view_angle_deg))
    hfov = 2.0 * np.arctan(np.tan(vfov * 0.5) * aspect)

    elev = np.deg2rad(90.0 - CONFIG["angulo_desde_vertical_deg"])
    azim = np.deg2rad(CONFIG["azimut_deg"])
    forward = np.array([-np.cos(elev) * np.cos(azim), -np.cos(elev) * np.sin(azim), -np.sin(elev)], dtype=float)
    forward /= np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, up_world); right /= np.linalg.norm(right)
    up_cam = np.cross(right, forward); up_cam /= np.linalg.norm(up_cam)

    corners = np.array([
        [xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
        [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax],
    ], dtype=float)
    rel = corners - centro
    u_extent = float(np.max(np.abs(rel @ right)))
    v_extent = float(np.max(np.abs(rel @ up_cam)))

    margen_seguridad = 1.02
    dist_x = u_extent / (np.tan(hfov * 0.5) * max(1e-6, usable_w * 0.5))
    dist_y = v_extent / (np.tan(vfov * 0.5) * max(1e-6, usable_h * 0.5))
    
    # IMPORTANTE: Aquí aplicamos el factor. Si factor_acercamiento > 1.0, la cámara se ALEJA.
    dist = max(dist_x, dist_y, CONFIG["distancia_camara"] * 0.55) * margen_seguridad
    dist *= float(factor_acercamiento)

    desired_u = (target_x - 0.5) * 2.0 * dist * np.tan(hfov * 0.5)
    desired_v = (target_y - 0.5) * 2.0 * dist * np.tan(vfov * 0.5)
    foco = centro - desired_u * right - desired_v * up_cam
    return (float(foco[0]), float(foco[1]), float(foco[2])), float(dist)

def parametros_camara_grupal(columnas: list[ColumnaGeofisica]) -> tuple[tuple[float, float, float], float]:
    bounds = calcular_bounds_columnas(columnas)
    return calcular_foco_y_distancia_con_margenes(
        bounds,
        resolucion=CONFIG_ACTIVA.get("resolucion_video", CONFIG["resolucion"]),
        # Usamos los valores de CONFIG que acabamos de modificar
        hud_ancho_reservado=CONFIG.get("hud_ancho_reservado_grupal", 0.30), 
        margen_derecho_escena=CONFIG.get("margen_derecho_escena", 0.05),
        margen_superior_escena=CONFIG.get("margen_superior_escena", 0.18),
        margen_inferior_escena=CONFIG.get("margen_inferior_escena", 0.05),
        margen_izquierdo_extra_escena=CONFIG.get("margen_izquierdo_extra_escena", 0.0),
        camera_view_angle_deg=CONFIG.get("angulo_desde_vertical_deg", 60.0),
        factor_acercamiento=CONFIG.get("factor_acercamiento_grupal", 1.25),
    )

def dibujar_columna(plotter: pv.Plotter, col: ColumnaGeofisica, destacada: bool, t: float, es_grupal: bool = False) -> None:
    z_center = CONFIG["z_centro_comun"]
    centro = (col.posicion[0], col.posicion[1], z_center)
    cilindro = pv.Cylinder(center=centro, direction=(0, 0, 1), radius=col.r, height=col.h, resolution=96, capping=False)
    
    color_actual, _ = color_por_estado_columna(col, destacada)
    textura_tintada = generar_textura_tintada(color_actual)
    
    if destacada:
        opacity, metallic, roughness = CONFIG.get("opacidad_central", 0.95), 0.35, 0.40
        dibujar_rod_y_rings(plotter, col)
    else:
        opacity, metallic, roughness = CONFIG.get("opacidad_circundantes", 0.65), 0.15, 0.65
        
    if es_grupal: opacity *= 0.5
        
    plotter.add_mesh(cilindro, color=color_actual, opacity=opacity, pbr=True, metallic=metallic, roughness=roughness, specular=0.20, smooth_shading=True)

    top = pv.Disc(center=(col.posicion[0], col.posicion[1], z_center + 0.5 * col.h), inner=0.0, outer=col.r, c_res=120).texture_map_to_plane(inplace=False)
    bot = pv.Disc(center=(col.posicion[0], col.posicion[1], z_center - 0.5 * col.h), inner=0.0, outer=col.r, c_res=120).texture_map_to_plane(inplace=False)
    opacity_tapas = min(opacity + 0.15, 1.0) if not destacada else opacity
    
    plotter.add_mesh(top, color=color_actual, texture=textura_tintada, pbr=True, metallic=0.18, roughness=0.38, opacity=opacity_tapas)
    plotter.add_mesh(bot, color=color_actual, texture=textura_tintada, pbr=True, metallic=0.18, roughness=0.38, opacity=opacity_tapas)
    dibujar_eje_discontinuo(plotter, col)

# --- FUNCIONES DE RENDER Y PIPELINE FINAL ---
def agregar_firma(plotter: pv.Plotter) -> None:
    win_h = plotter.window_size[1]
    scale = win_h / 720.0
    pos_x_firma = int((CONFIG.get("hud_pos_x", 40) + CONFIG.get("hud_padding", 35)) * scale)
    # Firma pequeña en la esquina inferior izquierda
    plotter.add_text("Guillermo Alba Buitrón", position=(pos_x_firma, int(40 * scale)), viewport=False, font_size=int(14 * scale), color="#FFFFFF", font_file=HUD_FONT_FILE, name="footer", shadow=True)

def render_animacion_columna_central() -> None:
    plotter = crear_plotter(window_size=CONFIG_ACTIVA.get("resolucion_video", CONFIG["resolucion"]))
    sufijo = CONFIG_ACTIVA["sufijo_archivo"]
    salida_video = CONFIG["ruta_salida"] / f"animacion_columna_central{sufijo}.mp4"
    salida_png = CONFIG["ruta_salida"] / f"fotograma_columna_central{sufijo}.png"
    col = ColumnaGeofisica(indice=CONFIG["indice_destacado"], posicion=np.array([0.0, 0.0]), fase=0.0, frecuencia=0.62, amp_local=CONFIG["amp_h"] * 0.95, q_cte=CONFIG["f_coriolis"] / CONFIG["h_media"])
    total_frames = CONFIG_ACTIVA["fps"] * CONFIG_ACTIVA["duracion_seg"]
    dt = 1.0 / CONFIG_ACTIVA["fps"]
    
    with imageio.get_writer(salida_video, fps=CONFIG_ACTIVA["fps"], codec="libx264", quality=CONFIG_ACTIVA["calidad_video"], macro_block_size=CONFIG_ACTIVA["macro_block_size"]) as writer:
        for i in range(total_frames):
            t = i * dt
            col.actualizar_estado(t)
            
            foco, dist = calcular_foco_y_distancia_con_margenes(
                calcular_bounds_columna(col), resolucion=CONFIG_ACTIVA.get("resolucion_video", CONFIG["resolucion"]),
                # Usamos los nuevos valores corregidos (0.30)
                hud_ancho_reservado=CONFIG.get("hud_ancho_reservado_individual", 0.30), margen_derecho_escena=CONFIG.get("margen_derecho_escena", 0.05),
                margen_superior_escena=CONFIG.get("margen_superior_escena", 0.18), margen_inferior_escena=CONFIG.get("margen_inferior_escena", 0.05),
                margen_izquierdo_extra_escena=CONFIG.get("margen_izquierdo_extra_escena", 0.0), camera_view_angle_deg=CONFIG.get("angulo_desde_vertical_deg", 60.0),
                factor_acercamiento=CONFIG.get("factor_acercamiento_individual", 1.15),
            )
            configurar_camara(plotter, foco_xyz=foco, distancia=dist)
            plotter.clear_actors()
            
            dibujar_columna(plotter, col, destacada=True, t=t, es_grupal=False)
            dibujar_hud_academico(plotter, preparar_datos_hud(col, t, titulo="PANEL FÍSICO", subtitulo="COLUMNA CENTRAL"), "hud_individual")
            agregar_firma(plotter)
            
            frame = plotter.screenshot(return_img=True)
            writer.append_data(frame)
            col.theta_visual += col.omega * dt * CONFIG.get("escala_giro_visual", 12000.0)
            if i == total_frames // 2:
                imageio.imwrite(salida_png, frame)
    plotter.close()

def render_animacion_grupal() -> None:
    plotter = crear_plotter(window_size=CONFIG_ACTIVA.get("resolucion_video", CONFIG["resolucion"]))
    sufijo = CONFIG_ACTIVA["sufijo_archivo"]
    salida_video = CONFIG["ruta_salida"] / f"animacion_grupal_5_columnas{sufijo}.mp4"
    salida_png = CONFIG["ruta_salida"] / f"fotograma_grupal_5_columnas{sufijo}.png"
    columnas = inicializar_columnas()
    total_frames = CONFIG_ACTIVA["fps"] * CONFIG_ACTIVA["duracion_seg"]
    dt = 1.0 / CONFIG_ACTIVA["fps"]
    
    with imageio.get_writer(salida_video, fps=CONFIG_ACTIVA["fps"], codec="libx264", quality=CONFIG_ACTIVA["calidad_video"], macro_block_size=CONFIG_ACTIVA["macro_block_size"]) as writer:
        for i in range(total_frames):
            t = i * dt
            for col in columnas: col.actualizar_estado(t)
            resolver_contactos(columnas, dt)
            
            # parametros_camara_grupal ya usa internamente CONFIG corregido
            foco, dist = parametros_camara_grupal(columnas)
            configurar_camara(plotter, foco_xyz=foco, distancia=dist)
            plotter.clear_actors()
            
            for col in columnas:
                dibujar_columna(plotter, col, destacada=(col.indice == CONFIG["indice_destacado"]), t=t, es_grupal=True)
            dibujar_flechas_compresion(plotter, columnas, t)
            dibujar_hud_academico(plotter, preparar_datos_hud(columnas[CONFIG["indice_destacado"]], t, titulo="PANEL FÍSICO", subtitulo="SISTEMA MULTI-COLUMNA"), "hud_grupal")
            agregar_firma(plotter)
            
            frame = plotter.screenshot(return_img=True)
            writer.append_data(frame)
            if i == total_frames // 2:
                imageio.imwrite(salida_png, frame)
    plotter.close()

def render_preview_columna_central() -> None:
    plotter = crear_plotter(window_size=CONFIG_ACTIVA["resolucion_preview"])
    sufijo = CONFIG_ACTIVA["sufijo_archivo"]
    salida_png = CONFIG["ruta_salida"] / f"preview_columna_central{sufijo}.png"
    col = ColumnaGeofisica(indice=CONFIG["indice_destacado"], posicion=np.array([0.0, 0.0]), fase=0.0, frecuencia=0.62, amp_local=CONFIG["amp_h"] * 0.95, q_cte=CONFIG["f_coriolis"] / CONFIG["h_media"])
    col.actualizar_estado(0.0)

    foco, dist = calcular_foco_y_distancia_con_margenes(
        calcular_bounds_columna(col), resolucion=CONFIG_ACTIVA["resolucion_preview"],
        # Nuevos valores corregidos
        hud_ancho_reservado=CONFIG.get("hud_ancho_reservado_individual", 0.30), margen_derecho_escena=CONFIG.get("margen_derecho_escena", 0.05),
        margen_superior_escena=CONFIG.get("margen_superior_escena", 0.18), margen_inferior_escena=CONFIG.get("margen_inferior_escena", 0.05),
        margen_izquierdo_extra_escena=CONFIG.get("margen_izquierdo_extra_escena", 0.0), camera_view_angle_deg=CONFIG.get("angulo_desde_vertical_deg", 60.0),
        factor_acercamiento=CONFIG.get("factor_acercamiento_individual", 1.15),
    )
    configurar_camara(plotter, foco_xyz=foco, distancia=dist)
    dibujar_columna(plotter, col, destacada=True, t=0.0, es_grupal=False)
    dibujar_hud_academico(plotter, preparar_datos_hud(col, 0.0, titulo="PANEL FÍSICO", subtitulo="COLUMNA CENTRAL"), "hud_preview_ind")
    agregar_firma(plotter)

    # Captura directa a archivo
    plotter.screenshot(salida_png)
    plotter.close()

def render_preview_grupal() -> None:
    plotter = crear_plotter(window_size=CONFIG_ACTIVA["resolucion_preview"])
    sufijo = CONFIG_ACTIVA["sufijo_archivo"]
    salida_png = CONFIG["ruta_salida"] / f"preview_grupal{sufijo}.png"
    columnas = inicializar_columnas()
    for col in columnas: col.actualizar_estado(0.0)

    foco, dist = calcular_foco_y_distancia_con_margenes(
        calcular_bounds_columnas(columnas), resolucion=CONFIG_ACTIVA["resolucion_preview"],
        # Nuevos valores corregidos
        hud_ancho_reservado=CONFIG.get("hud_ancho_reservado_grupal", 0.30), margen_derecho_escena=CONFIG.get("margen_derecho_escena", 0.05),
        margen_superior_escena=CONFIG.get("margen_superior_escena", 0.18), margen_inferior_escena=CONFIG.get("margen_inferior_escena", 0.05),
        margen_izquierdo_extra_escena=CONFIG.get("margen_izquierdo_extra_escena", 0.0), camera_view_angle_deg=CONFIG.get("angulo_desde_vertical_deg", 60.0),
        factor_acercamiento=CONFIG.get("factor_acercamiento_grupal", 1.25),
    )
    configurar_camara(plotter, foco_xyz=foco, distancia=dist)
    for col in columnas:
        dibujar_columna(plotter, col, destacada=(col.indice == CONFIG["indice_destacado"]), t=0.0, es_grupal=True)
    dibujar_flechas_compresion(plotter, columnas, 0.0)
    dibujar_hud_academico(plotter, preparar_datos_hud(columnas[CONFIG["indice_destacado"]], 0.0, titulo="PANEL FÍSICO", subtitulo="SISTEMA MULTI-COLUMNA"), "hud_preview_grup")
    agregar_firma(plotter)

    plotter.screenshot(salida_png)
    plotter.close()

def mostrar_resumen_configuracion() -> None:
    print("\nRESUMEN DE CONFIGURACIÓN ACTIVA")
    print("-" * 30)
    print(f"Modo:             {CONFIG_ACTIVA['nombre']}")
    print(f"Resolución Video: {CONFIG_ACTIVA['resolucion_video']}")
    print(f"FPS:              {CONFIG_ACTIVA['fps']}")
    print(f"Duración:         {CONFIG_ACTIVA['duracion_seg']} s")
    print(f"Reserva HUD indiv: {CONFIG['hud_ancho_reservado_individual']*100:.0f}%")
    print(f"Reserva HUD grupal:{CONFIG['hud_ancho_reservado_grupal']*100:.0f}%")
    print("-" * 30)

def main() -> None:
    validar_configuracion()
    ffmpeg_exe = configurar_ffmpeg_imageio()
    CONFIG["ruta_salida"].mkdir(parents=True, exist_ok=True)
    
    global CONFIG_ACTIVA
    modo_seleccionado = seleccionar_modo_exportacion()
    CONFIG_ACTIVA = PERFILES_RENDER[modo_seleccionado]
    
    print(f"\nFFmpeg en uso: {ffmpeg_exe}")
    mostrar_resumen_configuracion()
    
    print("\nINICIALIZANDO PIPELINE DE RENDER - COMPOSICIÓN CORREGIDA")
    print("=" * 70)
    
    print("\n[1/4] Generando preview de columna central...")
    render_preview_columna_central()
    sufijo = CONFIG_ACTIVA["sufijo_archivo"]
    print(f"      ✓ Preview columna central guardada: preview_columna_central{sufijo}.png")
    
    print("\n[2/4] Generando preview grupal...")
    render_preview_grupal()
    print(f"      ✓ Preview grupal guardada: preview_grupal{sufijo}.png")
    
    print("\n[3/4] Iniciando render de vídeo individual...")
    render_animacion_columna_central()
    print(f"      ✓ Vídeo de columna individual completado: animacion_columna_central{sufijo}.mp4")
    
    print("\n[4/4] Iniciando render de vídeo grupal...")
    render_animacion_grupal()
    print(f"      ✓ Vídeo grupal completado: animacion_grupal_5_columnas{sufijo}.mp4")
    
    print("\n" + "=" * 70)
    print("RENDER FINALIZADO CORRECTAMENTE")
    print("=" * 70)

if __name__ == "__main__":
    main()