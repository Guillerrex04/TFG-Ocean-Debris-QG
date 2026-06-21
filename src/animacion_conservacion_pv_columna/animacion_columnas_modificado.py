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
    "ruta_salida": Path(r"C:\Users\guill\Desktop\Guillermo\Aeroespacial\Cursos\TFG\CodigoTFG\Columnitas"),
    "fps": 24,
    "duracion_seg": 25,
    "n_columnas": 5,
    "f_coriolis": 1e-4,
    "volumen_ref": 10.0,
    "h_media": 5.0,
    "amp_h": 2.0,
    "resolucion": (1920, 1080),
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

    "hud_ancho_reservado_individual": 0.30,
    "factor_acercamiento_individual": 1.15,
    "hud_ancho_reservado_grupal": 0.30,
    "factor_acercamiento_grupal": 1.25,
    "margen_derecho_escena": 0.05,
    "margen_superior_escena": 0.13,
    "margen_inferior_escena": 0.10,
    "margen_izquierdo_extra_escena": 0.0,
    "escala_giro_visual": 120000.0,
    "bg_lower": "#0B132B",
    "bg_upper": "#1C2541",

    "color_lento_circundante": "#000000",
    "color_rapido_circundante": "#F4F7FB",
    "color_lento_central": "#FF0000",
    "color_rapido_central": "#00F5D4",
    "opacidad_circundantes": 0.65,
    "opacidad_central": 0.95,

    "hud_panel_color": "#0F172A",
    "hud_text_color": "#F8FAFC",
    "hud_accent_color": "#38BDF8",
    "hud_pos_x": 40,
    "hud_pos_y_top": 40,
    "hud_ancho": 950,
    "hud_padding": 24,
    "hud_font_title": 18,
    "hud_font_sub": 12,
    "hud_font_norm": 11,
    "hud_line_h": 24,

    # --- Marcador angular de giro (sector incrustado) ---
    "sector_angulo_deg": 30,
    "sector_color": "#FFD700",
    "sector_opacidad": 0.9,
}

PERFILES_RENDER = {
    "rapido": {
        "nombre": "Rapido Pruebas (720p)",
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

# ---------------------------------------------------------------------------
# Resoluciones de malla optimizadas (calidad visual ~identica en 1080p)
# ---------------------------------------------------------------------------
_MESH_RES = {
    "cyl": 48,       # antes 96
    "disc": 60,      # antes 120
    "dash": 6,       # antes 12
    "rod": 16,       # antes 32
    "ring_path": 24, # antes 90
    "ring_tube": 8,  # antes 30 (n_sides)
    "cone": 10,      # antes 16
    "arrow_tip": 12, # antes 30
    "arrow_shaft": 12, # antes 30
}

# ---------------------------------------------------------------------------
# Cache de texturas (single-slot: ultimo color)
# ---------------------------------------------------------------------------
_TEX_CACHE_COLOR = None
_TEX_CACHE_VALUE = None

# ---------------------------------------------------------------------------
# Mallas unitarias reutilizables (se crean una vez, se escalan/posicionan)
# ---------------------------------------------------------------------------
_UNIT_CYL = None
_UNIT_DISC = None
_UNIT_DASH = None
_UNIT_ROD = None

def _init_unit_meshes():
    global _UNIT_CYL, _UNIT_DISC, _UNIT_DASH, _UNIT_ROD
    if _UNIT_CYL is not None:
        return
    _UNIT_CYL = pv.Cylinder(center=(0,0,0), direction=(0,0,1),
                            radius=1.0, height=1.0,
                            resolution=_MESH_RES["cyl"], capping=False)
    _UNIT_DISC = pv.Disc(center=(0,0,0), inner=0.0, outer=1.0,
                         c_res=_MESH_RES["disc"])
    _UNIT_DISC = _UNIT_DISC.texture_map_to_plane(inplace=False)
    _UNIT_DASH = pv.Cylinder(center=(0,0,0), direction=(0,0,1),
                             radius=0.012, height=1.0,
                             resolution=_MESH_RES["dash"])
    _UNIT_ROD = pv.Cylinder(center=(0,0,0), direction=(0,0,1),
                            radius=1.0, height=1.6,
                            resolution=_MESH_RES["rod"])


def _crear_malla_cuna(angulo_deg: float = 30, radio: float = 1.002) -> pv.PolyData:
    """Malla CERRADA de cuna cilindrica (radio=radio, altura=1, centrada en origen).

    Incluye: superficie lateral + tapa superior + tapa inferior + 2 caras radiales.
    El radio > 1 evita z-fighting con el cilindro principal.
    Escalar con actor.SetScale(r, r, h) y rotar con SetOrientation(0,0,theta).
    """
    ang_rad = np.deg2rad(angulo_deg)
    n_arc = 8
    theta = np.linspace(0, ang_rad, n_arc + 1)
    ct = radio * np.cos(theta)
    st = radio * np.sin(theta)

    pts = np.zeros((2 * n_arc + 4, 3), dtype=np.float64)
    for i in range(n_arc + 1):
        pts[i] = [ct[i], st[i], 0.5]            # arco superior
        pts[n_arc + 1 + i] = [ct[i], st[i], -0.5]  # arco inferior
    pts[2 * n_arc + 2] = [0.0, 0.0, 0.5]         # centro superior
    pts[2 * n_arc + 3] = [0.0, 0.0, -0.5]        # centro inferior

    off_inf = n_arc + 1
    cs = 2 * n_arc + 2
    ci = 2 * n_arc + 3

    faces = []
    # Lateral (quads): arco_sup[i] -> arco_sup[i+1] -> arco_inf[i+1] -> arco_inf[i]
    for i in range(n_arc):
        faces.extend([4, i, i + 1, off_inf + i + 1, off_inf + i])
    # Tapa superior (triangulos): centro -> arco_sup[i] -> arco_sup[i+1]
    for i in range(n_arc):
        faces.extend([3, cs, i, i + 1])
    # Tapa inferior (triangulos, winding invertido): centro -> arco_inf[i+1] -> arco_inf[i]
    for i in range(n_arc):
        faces.extend([3, ci, off_inf + i + 1, off_inf + i])
    # Cara radial theta=0
    faces.extend([4, 0, cs, ci, off_inf])
    # Cara radial theta=ang_deg
    faces.extend([4, n_arc, off_inf + n_arc, ci, cs])

    mesh = pv.PolyData(pts, np.array(faces, dtype=np.int64))
    mesh.compute_normals(cell_normals=True, auto_orient_normals=True, inplace=True)
    return mesh

def seleccionar_modo_exportacion() -> str:
    print("\n" + "=" * 70)
    print("SELECCIONAR MODO DE EXPORTACION")
    print("=" * 70)
    print("\nOpciones disponibles:")
    print("  [1] Rapido / Pruebas (720p, baja FPS)")
    print("  [2] Alta calidad / Resultado final (1080p, 24 FPS)")

    while True:
        opcion = input("\nIngresa tu opcion (1 o 2): ").strip()
        if opcion == "1": return "rapido"
        elif opcion == "2": return "alta_calidad"
        else: print("Opcion invalida. Por favor, ingresa 1 o 2.")

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

def intensidad_compresion_central(col_central: ColumnaGeofisica) -> float:
    h_media = CONFIG["h_media"]
    amp_local = col_central.amp_local
    h_max = h_media + amp_local
    h_min = h_media - amp_local
    r_min = np.sqrt(CONFIG["volumen_ref"] / (np.pi * h_max))
    r_max = np.sqrt(CONFIG["volumen_ref"] / (np.pi * h_min))
    intensidad = 1.0 - (col_central.r - r_min) / max(r_max - r_min, 1e-6)
    return float(np.clip(intensidad, 0.0, 1.0))

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
    global _TEX_CACHE_COLOR, _TEX_CACHE_VALUE
    if hex_color == _TEX_CACHE_COLOR:
        return _TEX_CACHE_VALUE
    mapa = obtener_mapa_ruido()
    rgb = np.array(hex_a_rgb_f(hex_color))
    factor = 0.5 + 0.5 * mapa
    r = (rgb[0] * factor * 255).astype(np.uint8)
    g = (rgb[1] * factor * 255).astype(np.uint8)
    b = (rgb[2] * factor * 255).astype(np.uint8)
    arr = np.ascontiguousarray(np.dstack([r, g, b]))
    _TEX_CACHE_VALUE = pv.numpy_to_texture(arr)
    _TEX_CACHE_COLOR = hex_color
    return _TEX_CACHE_VALUE

def preparar_datos_hud(col: ColumnaGeofisica, t: float, titulo: str = "PANEL DE CONTROL FISICO", subtitulo: str = "COLUMNA CENTRAL") -> dict:
    color_hex, intensidad = color_por_estado_columna(col, destacada=True)
    intensidad_comp = intensidad_compresion_central(col)

    return {
        "titulo": titulo.upper(),
        "subtitulo": subtitulo.upper(),
        "estado": "MODO: SHALLOW-WATER / PV CONSERVADA",
        "filas": [
            ("Tiempo de simulacion", "t", f"{t:6.2f}", "s"),
            ("Altura de columna", "h", f"{col.h:6.3f}", "m"),
            ("Radio efectivo", "r", f"{col.r:6.3f}", "m"),
            ("Velocidad angular", chr(969), f"{col.omega:.3e}", "rad/s"),
            ("Vorticidad relativa", chr(950), f"{col.zeta:.3e}", "s-1"),
            ("Vorticidad potencial", "q", f"{col.q_cte:.3e}", "m-1s-1"),
            ("Compresion lateral", "C_lat", f"{np.clip(intensidad_comp*100,0,100):5.1f}", "%"),
            ("Intensidad cromatica", "I", f"{intensidad*100:5.1f}", "%"),
        ],
    }

# ---------------------------------------------------------------------------
# HUD background pre-computado (no recalcular por frame)
# ---------------------------------------------------------------------------
_HUD_BG_TEXT = "\n".join(["\u2588" * 62 for _ in range(40)])

def dibujar_hud_academico(plotter: pv.Plotter, datos: dict, name: str = "hud_panel") -> None:
    win_w, win_h = plotter.window_size
    scale = win_h / 720.0

    pos_x = int(CONFIG.get("hud_pos_x", 40) * scale)
    pos_y_top = win_h - int(CONFIG.get("hud_pos_y_top", 40) * scale)
    padding = int(CONFIG.get("hud_padding", 35) * scale)

    f_tit = int(CONFIG.get("hud_font_title", 26) * scale)
    f_sub = int(CONFIG.get("hud_font_sub", 19) * scale)
    f_norm = int(CONFIG.get("hud_font_norm", 17) * scale)
    line_h = int(CONFIG.get("hud_line_h", 45) * scale)

    n_filas = len(datos["filas"])
    curr_y = pos_y_top - padding

    y_tit = curr_y; curr_y -= (f_tit + int(15 * scale))
    y_sub = curr_y; curr_y -= (f_sub + int(15 * scale))
    y_mod = curr_y; curr_y -= (f_norm + int(10 * scale))
    y_sep = curr_y; curr_y -= (f_norm + int(15 * scale))
    y_head = curr_y; curr_y -= (f_norm + int(20 * scale))
    start_y = curr_y

    pos_y_bg = start_y - (n_filas * line_h) - int(40 * scale)
    try:
        plotter.add_text(_HUD_BG_TEXT, position=(pos_x - int(15 * scale), pos_y_bg),
                         font_size=int(18 * scale), color=CONFIG.get("hud_panel_color", "#0F172A"),
                         font_file=HUD_FONT_FILE, name=f"{name}_bg", viewport=False, shadow=False)
    except Exception:
        plotter.add_text(_HUD_BG_TEXT, position=(pos_x - int(15 * scale), pos_y_bg),
                         font_size=int(18 * scale), color=CONFIG.get("hud_panel_color", "#0F172A"),
                         name=f"{name}_bg", viewport=False, shadow=False)

    plotter.add_text(datos["titulo"], position=(pos_x + padding, y_tit), font_size=f_tit,
                     color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE,
                     name=f"{name}_tit", viewport=False, shadow=True)
    plotter.add_text(datos["subtitulo"], position=(pos_x + padding, y_sub), font_size=f_sub,
                     color=CONFIG.get("hud_text_color", "#F8FAFC"), font_file=HUD_FONT_FILE,
                     name=f"{name}_sub", viewport=False)
    plotter.add_text(datos["estado"], position=(pos_x + padding, y_mod),
                     font_size=max(1, f_norm - int(2 * scale)), color="#94A3B8",
                     font_file=HUD_FONT_FILE, name=f"{name}_mod", viewport=False)
    plotter.add_text("\u23AF" * 50, position=(pos_x + padding, y_sep), font_size=f_norm,
                     color="#334155", font_file=HUD_FONT_FILE, name=f"{name}_sep", viewport=False)

    x_mag = pos_x + padding
    x_sim = pos_x + int(270 * scale)
    x_val = pos_x + int(370 * scale)
    x_uni = pos_x + int(480 * scale)

    plotter.add_text("MAGNITUD", position=(x_mag, y_head), font_size=max(1, f_norm - 1),
                     color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE,
                     name=f"{name}_h1", viewport=False)
    plotter.add_text("SIMBOLO", position=(x_sim, y_head), font_size=max(1, f_norm - 1),
                     color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE,
                     name=f"{name}_h2", viewport=False)
    plotter.add_text("VALOR", position=(x_val, y_head), font_size=max(1, f_norm - 1),
                     color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE,
                     name=f"{name}_h3", viewport=False)
    plotter.add_text("UNIDAD", position=(x_uni, y_head), font_size=max(1, f_norm - 1),
                     color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE,
                     name=f"{name}_h4", viewport=False)

    for i, fila in enumerate(datos["filas"]):
        y_i = start_y - i * line_h
        plotter.add_text(fila[0], position=(x_mag, y_i), font_size=f_norm, color="#CBD5E1",
                         font_file=HUD_FONT_FILE, name=f"{name}_lbl_{i}", viewport=False)
        plotter.add_text(fila[1], position=(x_sim, y_i), font_size=f_norm,
                         color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE,
                         name=f"{name}_sym_{i}", viewport=False)
        plotter.add_text(fila[2], position=(x_val, y_i), font_size=f_norm,
                         color=CONFIG.get("hud_text_color", "#F8FAFC"), font_file=HUD_FONT_FILE,
                         name=f"{name}_val_{i}", viewport=False)
        plotter.add_text(fila[3], position=(x_uni, y_i), font_size=f_norm,
                         color=CONFIG.get("hud_accent_color", "#38BDF8"), font_file=HUD_FONT_FILE,
                         name=f"{name}_uni_{i}", viewport=False)

# =============================================================================
# COLUMN SCENE - Pool de actores reutilizables (NUCLEO DE LA OPTIMIZACION)
# =============================================================================

class ColumnScene:
    """Crea todos los actores UNA VEZ; cada frame solo actualiza propiedades.

    Reemplaza: dibujar_columna, dibujar_eje_discontinuo, dibujar_rod_y_rings,
    dibujar_flechas_compresion. Elimina clear_actors() + add_mesh() por frame.
    """
    __slots__ = ('plotter', 'cols', 'arrows', 'arrow_color', 'arrow_opacity')

    def __init__(self, plotter: pv.Plotter, n_columnas: int):
        _init_unit_meshes()
        self.plotter = plotter
        sector_mesh = _crear_malla_cuna(CONFIG.get("sector_angulo_deg", 30))
        sector_color = CONFIG.get("sector_color", "#FFD700")
        sector_opacidad = CONFIG.get("sector_opacidad", 0.9)
        self.cols = []
        for _ in range(n_columnas):
            col = {}
            # --- Cilindro principal ---
            col['cyl'] = plotter.add_mesh(
                _UNIT_CYL, pbr=True, metallic=0.15, roughness=0.65,
                specular=0.20, smooth_shading=True)
            # --- Tapas ---
            col['top'] = plotter.add_mesh(
                _UNIT_DISC, pbr=True, metallic=0.18, roughness=0.38)
            col['bot'] = plotter.add_mesh(
                _UNIT_DISC, pbr=True, metallic=0.18, roughness=0.38)
            # --- Eje discontinuo (14 dash cylinders) ---
            col['dashes'] = [
                plotter.add_mesh(_UNIT_DASH, color="#E2E8F0",
                                 ambient=0.6, pbr=False)
                for _ in range(14)]
            # --- Rod (varilla superior) ---
            col['rod'] = plotter.add_mesh(
                _UNIT_ROD, color="#AAAAAA", pbr=True, metallic=0.8,
                roughness=0.2, specular=0.8, smooth_shading=True)
            col['rod'].VisibilityOff()
            # --- Marcador sectorial (angulo de giro incrustado) ---
            col['sector'] = plotter.add_mesh(
                sector_mesh, color=sector_color,
                opacity=sector_opacidad,
                pbr=False, ambient=0.5, smooth_shading=True)
            # --- Anillos (3 por columna, ocultos por defecto) ---
            col['rings'] = []
            for __ in range(3):
                dummy = pv.Cylinder(radius=0.001, height=0.001)
                t = plotter.add_mesh(
                    dummy, color="#FFFFFF", smooth_shading=True,
                    pbr=False, roughness=1.0, ambient=0.5)
                k = plotter.add_mesh(
                    dummy, color="#FFFFFF", smooth_shading=True,
                    pbr=False, ambient=0.5)
                t.VisibilityOff()
                k.VisibilityOff()
                col['rings'].append((t, k))
            self.cols.append(col)

        # --- Flechas de compresion (hasta 4) ---
        self.arrows = []
        self.arrow_color = None
        self.arrow_opacity = 1.0
        for _ in range(4):
            dummy = pv.Cylinder(radius=0.001, height=0.001)
            a = plotter.add_mesh(dummy, color="#FF0000", opacity=0.85,
                                 ambient=0.5, pbr=False, smooth_shading=True)
            a.VisibilityOff()
            self.arrows.append(a)

    # ------------------------------------------------------------------
    def update_column(self, idx: int, col: ColumnaGeofisica,
                      destacada: bool, t: float, es_grupal: bool) -> None:
        c = self.cols[idx]
        zc = CONFIG["z_centro_comun"]

        color_hex, _ = color_por_estado_columna(col, destacada)
        rgb = hex_a_rgb_f(color_hex)

        # --- Cilindro ---
        cyl = c['cyl']
        cyl.SetPosition(float(col.posicion[0]), float(col.posicion[1]), float(zc))
        cyl.SetScale(float(col.r), float(col.r), float(col.h))
        prop = cyl.GetProperty()
        prop.SetColor(*rgb)
        if destacada:
            op = CONFIG.get("opacidad_central", 0.95)
            prop.SetOpacity(op * (0.5 if es_grupal else 1.0))
            prop.SetMetallic(0.35)
            prop.SetRoughness(0.40)
        else:
            op = CONFIG.get("opacidad_circundantes", 0.65)
            prop.SetOpacity(op * (0.5 if es_grupal else 1.0))
            prop.SetMetallic(0.15)
            prop.SetRoughness(0.65)

        # --- Tapas ---
        op_disc = min(op + 0.15, 1.0) if not destacada else op
        for key, sign in (('top', 0.5), ('bot', -0.5)):
            d = c[key]
            d.SetPosition(float(col.posicion[0]), float(col.posicion[1]),
                          float(zc + sign * col.h))
            d.SetScale(float(col.r), float(col.r), 1.0)
            dp = d.GetProperty()
            dp.SetColor(*rgb)
            dp.SetOpacity(op_disc)

        # Textura solo en columna destacada (evita subida a GPU para las demas)
        if destacada:
            tex = generar_textura_tintada(color_hex)
            c['top'].SetTexture(tex)
            c['bot'].SetTexture(tex)
        else:
            c['top'].SetTexture(None)
            c['bot'].SetTexture(None)

        # --- Eje discontinuo ---
        z_min = zc - col.h * 0.65
        z_max = zc + col.h * 0.65
        z_pts = np.linspace(z_min, z_max, 28)
        dash_h = abs(z_pts[1] - z_pts[0]) * 0.6
        cx, cy = float(col.posicion[0]), float(col.posicion[1])
        dashes = c['dashes']
        for i in range(14):
            zm = (z_pts[2*i] + z_pts[2*i+1]) * 0.5
            d = dashes[i]
            d.SetPosition(cx, cy, float(zm))
            d.SetScale(1.0, 1.0, float(dash_h))

        # --- Rod ---
        rod = c['rod']
        if destacada:
            pos_top = zc + 0.5 * col.h
            rod.SetPosition(float(col.posicion[0]), float(col.posicion[1]),
                            float(pos_top + 0.7 * col.r))
            rod.SetScale(float(0.04 * col.r), float(0.04 * col.r), float(col.r))
            rod.VisibilityOn()
        else:
            rod.VisibilityOff()

        # --- Anillos ---
        if destacada:
            self._update_rings(c, col)
        else:
            for tube_act, cone_act in c['rings']:
                tube_act.VisibilityOff()
                cone_act.VisibilityOff()

        # --- Marcador sectorial (solo en columna destacada) ---
        sector = c['sector']
        if destacada:
            sector.SetPosition(float(col.posicion[0]), float(col.posicion[1]), float(zc))
            sector.SetScale(float(col.r), float(col.r), float(col.h))
            sector.SetOrientation(0, 0, float(np.degrees(col.theta_visual)))
            sector.GetProperty().SetOpacity(op * (0.5 if es_grupal else 1.0))
            sector.VisibilityOn()
        else:
            sector.VisibilityOff()

    # ------------------------------------------------------------------
    def _update_rings(self, c: dict, col: ColumnaGeofisica) -> None:
        max_omega = 0.5 * abs(CONFIG["f_coriolis"]) * (CONFIG["amp_h"] / CONFIG["h_media"])
        omega_ratio = abs(col.omega) / (max_omega + 1e-12)
        zc = CONFIG["z_centro_comun"]
        pos_top = zc + 0.5 * col.h
        giro = col.theta_visual
        signo = 1 if col.omega >= 0 else -1

        if signo >= 0:
            ang_ini = giro + np.radians(25)
            ang_fin = giro + np.radians(335)
        else:
            ang_ini = giro - np.radians(25)
            ang_fin = giro - np.radians(335)

        thresholds = [0.0, 0.4, 0.7]
        base_offset = 0.3 * col.r
        stack_spacing = 0.4 * col.r
        n_pts = _MESH_RES["ring_path"]
        cx, cy = float(col.posicion[0]), float(col.posicion[1])

        for i in range(3):
            tube_act, cone_act = c['rings'][i]
            if omega_ratio >= thresholds[i]:
                offset = base_offset + i * stack_spacing
                theta = np.linspace(ang_ini, ang_fin, n_pts)
                r_ring = 0.35 * col.r
                pts = np.column_stack((
                    cx + r_ring * np.cos(theta),
                    cy + r_ring * np.sin(theta),
                    np.full(n_pts, float(pos_top + offset)),
                ))
                path = pv.Spline(points=pts)
                tube = path.tube(radius=0.03 * col.r, n_sides=_MESH_RES["ring_tube"])
                tube_act.mapper.SetInputData(tube)
                tube_act.VisibilityOn()

                tip_pos = np.array([cx + r_ring * np.cos(ang_fin),
                                    cy + r_ring * np.sin(ang_fin),
                                    float(pos_top + offset)])
                dir_v = np.array([-np.sin(ang_fin) * signo,
                                   np.cos(ang_fin) * signo, 0.0])
                nrm = np.linalg.norm(dir_v)
                if nrm > 1e-12:
                    dir_v /= nrm
                h_cone = 0.15 * col.r
                ctr = tip_pos + dir_v * (h_cone * 0.5)
                cone = pv.Cone(center=ctr, direction=dir_v,
                               radius=0.06 * col.r, height=h_cone,
                               resolution=_MESH_RES["cone"])
                cone_act.mapper.SetInputData(cone)
                cone_act.VisibilityOn()
            else:
                tube_act.VisibilityOff()
                cone_act.VisibilityOff()

    # ------------------------------------------------------------------
    def update_arrows(self, columnas: list[ColumnaGeofisica], t: float) -> None:
        idx_central = CONFIG["indice_destacado"]
        col_central = columnas[idx_central]
        intensidad = intensidad_compresion_central(col_central)
        c_low = np.array(hex_a_rgb_f("#8B0000"))
        c_high = np.array(hex_a_rgb_f("#FF0000"))
        color_rgb = c_low * (1.0 - intensidad) + c_high * intensidad
        color_hex = "#{:02X}{:02X}{:02X}".format(
            *(int(round(c * 255.0)) for c in np.clip(color_rgb, 0.0, 1.0)))

        opacidad = 0.85 + 0.15 * intensidad
        long_base = 0.6 + 0.6 * intensidad
        grosor_shaft = 0.08 + 0.04 * intensidad
        grosor_tip = 0.18 + 0.08 * intensidad

        z_mid = CONFIG["z_centro_comun"]
        radio_ref = np.sqrt(CONFIG["volumen_ref"] / (np.pi * CONFIG["h_media"]))
        arrow_idx = 0

        for col in columnas:
            if col.indice == idx_central:
                continue
            delta = col_central.posicion - col.posicion
            dist = np.linalg.norm(delta)
            if dist < 1e-6:
                continue
            dir_v = delta / dist

            long_real = radio_ref * long_base
            margen = 0.15 * radio_ref
            p_fin = col_central.posicion - dir_v * (col_central.r + margen)
            p_ini = p_fin - dir_v * long_real

            arrow = pv.Arrow(
                start=(float(p_ini[0]), float(p_ini[1]), float(z_mid)),
                direction=(float(dir_v[0]), float(dir_v[1]), 0.0),
                scale=float(long_real), tip_length=0.3,
                tip_radius=float(grosor_tip), shaft_radius=float(grosor_shaft),
                tip_resolution=_MESH_RES["arrow_tip"],
                shaft_resolution=_MESH_RES["arrow_shaft"])

            act = self.arrows[arrow_idx]
            act.mapper.SetInputData(arrow)
            act.GetProperty().SetColor(*hex_a_rgb_f(color_hex))
            act.GetProperty().SetOpacity(opacidad)
            act.VisibilityOn()
            arrow_idx += 1

        # Ocultar flechas sobrantes
        for j in range(arrow_idx, len(self.arrows)):
            self.arrows[j].VisibilityOff()

    # ------------------------------------------------------------------
    def clear_arrows(self) -> None:
        for a in self.arrows:
            a.VisibilityOff()

# =============================================================================
# PIPELINE Y RENDERS DE CAMARA
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
    return (float(col.posicion[0] - col.r), float(col.posicion[0] + col.r),
            float(col.posicion[1] - col.r), float(col.posicion[1] + col.r),
            float(z_center - 0.5 * col.h), float(z_center + 0.5 * col.h))

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

    left_reserved = float(hud_ancho_reservado if hud_ancho_reservado is not None else 0.36)
    left_reserved += float(margen_izquierdo_extra_escena if margen_izquierdo_extra_escena is not None else 0.0)
    right_reserved = float(margen_derecho_escena if margen_derecho_escena is not None else 0.06)
    top_reserved = float(margen_superior_escena if margen_superior_escena is not None else 0.08)
    bottom_reserved = float(margen_inferior_escena if margen_inferior_escena is not None else 0.10)

    usable_w = max(1e-6, 1.0 - left_reserved - right_reserved)
    usable_h = max(1e-6, 1.0 - top_reserved - bottom_reserved)

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
        hud_ancho_reservado=CONFIG.get("hud_ancho_reservado_grupal", 0.30),
        margen_derecho_escena=CONFIG.get("margen_derecho_escena", 0.05),
        margen_superior_escena=CONFIG.get("margen_superior_escena", 0.18),
        margen_inferior_escena=CONFIG.get("margen_inferior_escena", 0.05),
        margen_izquierdo_extra_escena=CONFIG.get("margen_izquierdo_extra_escena", 0.0),
        camera_view_angle_deg=CONFIG.get("angulo_desde_vertical_deg", 60.0),
        factor_acercamiento=CONFIG.get("factor_acercamiento_grupal", 1.25),
    )

# --- FUNCIONES DE RENDER Y PIPELINE FINAL ---

def agregar_firma(plotter: pv.Plotter) -> None:
    win_h = plotter.window_size[1]
    scale = win_h / 720.0
    pos_x_firma = int((CONFIG.get("hud_pos_x", 40) + CONFIG.get("hud_padding", 35)) * scale)
    plotter.add_text("Guillermo Alba Buitron", position=(pos_x_firma, int(40 * scale)),
                     viewport=False, font_size=int(14 * scale), color="#FFFFFF",
                     font_file=HUD_FONT_FILE, name="footer", shadow=True)

def render_animacion_columna_central() -> None:
    plotter = crear_plotter(window_size=CONFIG_ACTIVA.get("resolucion_video", CONFIG["resolucion"]))
    scene = ColumnScene(plotter, 1)
    sufijo = CONFIG_ACTIVA["sufijo_archivo"]
    salida_video = CONFIG["ruta_salida"] / f"animacion_columna_central{sufijo}.mp4"
    salida_png = CONFIG["ruta_salida"] / f"fotograma_columna_central{sufijo}.png"
    col = ColumnaGeofisica(indice=CONFIG["indice_destacado"], posicion=np.array([0.0, 0.0]),
                           fase=0.0, frecuencia=0.62,
                           amp_local=CONFIG["amp_h"] * 0.95,
                           q_cte=CONFIG["f_coriolis"] / CONFIG["h_media"])
    total_frames = CONFIG_ACTIVA["fps"] * CONFIG_ACTIVA["duracion_seg"]
    dt = 1.0 / CONFIG_ACTIVA["fps"]

    with imageio.get_writer(salida_video, fps=CONFIG_ACTIVA["fps"], codec="libx264",
                            quality=CONFIG_ACTIVA["calidad_video"],
                            macro_block_size=CONFIG_ACTIVA["macro_block_size"]) as writer:
        for i in range(total_frames):
            t = i * dt
            col.actualizar_estado(t)

            foco, dist = calcular_foco_y_distancia_con_margenes(
                calcular_bounds_columna(col),
                resolucion=CONFIG_ACTIVA.get("resolucion_video", CONFIG["resolucion"]),
                hud_ancho_reservado=CONFIG.get("hud_ancho_reservado_individual", 0.30),
                margen_derecho_escena=CONFIG.get("margen_derecho_escena", 0.05),
                margen_superior_escena=CONFIG.get("margen_superior_escena", 0.18),
                margen_inferior_escena=CONFIG.get("margen_inferior_escena", 0.05),
                margen_izquierdo_extra_escena=CONFIG.get("margen_izquierdo_extra_escena", 0.0),
                camera_view_angle_deg=CONFIG.get("angulo_desde_vertical_deg", 60.0),
                factor_acercamiento=CONFIG.get("factor_acercamiento_individual", 1.15),
            )
            configurar_camara(plotter, foco_xyz=foco, distancia=dist)

            # Actualizar actores in-situ (SIN clear_actors + add_mesh)
            scene.update_column(0, col, destacada=True, t=t, es_grupal=False)
            dibujar_hud_academico(plotter, preparar_datos_hud(col, t, titulo="PANEL FISICO",
                                  subtitulo="COLUMNA CENTRAL"), "hud_individual")
            agregar_firma(plotter)

            frame = plotter.screenshot(return_img=True)
            writer.append_data(frame)
            col.theta_visual += col.omega * dt * CONFIG.get("escala_giro_visual", 12000.0)
            if i == total_frames // 2:
                imageio.imwrite(salida_png, frame)
    plotter.close()

def render_animacion_grupal() -> None:
    plotter = crear_plotter(window_size=CONFIG_ACTIVA.get("resolucion_video", CONFIG["resolucion"]))
    scene = ColumnScene(plotter, CONFIG["n_columnas"])
    sufijo = CONFIG_ACTIVA["sufijo_archivo"]
    salida_video = CONFIG["ruta_salida"] / f"animacion_grupal_5_columnas{sufijo}.mp4"
    salida_png = CONFIG["ruta_salida"] / f"fotograma_grupal_5_columnas{sufijo}.png"
    columnas = inicializar_columnas()
    total_frames = CONFIG_ACTIVA["fps"] * CONFIG_ACTIVA["duracion_seg"]
    dt = 1.0 / CONFIG_ACTIVA["fps"]
    idx_dest = CONFIG["indice_destacado"]

    with imageio.get_writer(salida_video, fps=CONFIG_ACTIVA["fps"], codec="libx264",
                            quality=CONFIG_ACTIVA["calidad_video"],
                            macro_block_size=CONFIG_ACTIVA["macro_block_size"]) as writer:
        for i in range(total_frames):
            t = i * dt
            for col in columnas:
                col.actualizar_estado(t)
            resolver_contactos(columnas, dt)

            foco, dist = parametros_camara_grupal(columnas)
            configurar_camara(plotter, foco_xyz=foco, distancia=dist)

            # Actualizar actores in-situ
            for col in columnas:
                scene.update_column(col.indice, col,
                                    destacada=(col.indice == idx_dest),
                                    t=t, es_grupal=True)
            scene.update_arrows(columnas, t)
            dibujar_hud_academico(plotter, preparar_datos_hud(columnas[idx_dest], t,
                                  titulo="PANEL FISICO", subtitulo="SISTEMA MULTI-COLUMNA"),
                                  "hud_grupal")
            agregar_firma(plotter)

            frame = plotter.screenshot(return_img=True)
            writer.append_data(frame)
            if i == total_frames // 2:
                imageio.imwrite(salida_png, frame)
    plotter.close()

def render_preview_columna_central() -> None:
    plotter = crear_plotter(window_size=CONFIG_ACTIVA["resolucion_preview"])
    scene = ColumnScene(plotter, 1)
    sufijo = CONFIG_ACTIVA["sufijo_archivo"]
    salida_png = CONFIG["ruta_salida"] / f"preview_columna_central{sufijo}.png"
    col = ColumnaGeofisica(indice=CONFIG["indice_destacado"], posicion=np.array([0.0, 0.0]),
                           fase=0.0, frecuencia=0.62,
                           amp_local=CONFIG["amp_h"] * 0.95,
                           q_cte=CONFIG["f_coriolis"] / CONFIG["h_media"])
    col.actualizar_estado(0.0)

    foco, dist = calcular_foco_y_distancia_con_margenes(
        calcular_bounds_columna(col), resolucion=CONFIG_ACTIVA["resolucion_preview"],
        hud_ancho_reservado=CONFIG.get("hud_ancho_reservado_individual", 0.30),
        margen_derecho_escena=CONFIG.get("margen_derecho_escena", 0.05),
        margen_superior_escena=CONFIG.get("margen_superior_escena", 0.18),
        margen_inferior_escena=CONFIG.get("margen_inferior_escena", 0.05),
        margen_izquierdo_extra_escena=CONFIG.get("margen_izquierdo_extra_escena", 0.0),
        camera_view_angle_deg=CONFIG.get("angulo_desde_vertical_deg", 60.0),
        factor_acercamiento=CONFIG.get("factor_acercamiento_individual", 1.15),
    )
    configurar_camara(plotter, foco_xyz=foco, distancia=dist)
    scene.update_column(0, col, destacada=True, t=0.0, es_grupal=False)
    dibujar_hud_academico(plotter, preparar_datos_hud(col, 0.0, titulo="PANEL FISICO",
                          subtitulo="COLUMNA CENTRAL"), "hud_preview_ind")
    agregar_firma(plotter)
    plotter.screenshot(salida_png)
    plotter.close()

def render_preview_grupal() -> None:
    plotter = crear_plotter(window_size=CONFIG_ACTIVA["resolucion_preview"])
    scene = ColumnScene(plotter, CONFIG["n_columnas"])
    sufijo = CONFIG_ACTIVA["sufijo_archivo"]
    salida_png = CONFIG["ruta_salida"] / f"preview_grupal{sufijo}.png"
    columnas = inicializar_columnas()
    for col in columnas:
        col.actualizar_estado(0.0)

    foco, dist = calcular_foco_y_distancia_con_margenes(
        calcular_bounds_columnas(columnas), resolucion=CONFIG_ACTIVA["resolucion_preview"],
        hud_ancho_reservado=CONFIG.get("hud_ancho_reservado_grupal", 0.30),
        margen_derecho_escena=CONFIG.get("margen_derecho_escena", 0.05),
        margen_superior_escena=CONFIG.get("margen_superior_escena", 0.18),
        margen_inferior_escena=CONFIG.get("margen_inferior_escena", 0.05),
        margen_izquierdo_extra_escena=CONFIG.get("margen_izquierdo_extra_escena", 0.0),
        camera_view_angle_deg=CONFIG.get("angulo_desde_vertical_deg", 60.0),
        factor_acercamiento=CONFIG.get("factor_acercamiento_grupal", 1.25),
    )
    configurar_camara(plotter, foco_xyz=foco, distancia=dist)
    idx_dest = CONFIG["indice_destacado"]
    for col in columnas:
        scene.update_column(col.indice, col, destacada=(col.indice == idx_dest),
                            t=0.0, es_grupal=True)
    scene.update_arrows(columnas, 0.0)
    dibujar_hud_academico(plotter, preparar_datos_hud(columnas[idx_dest], 0.0,
                          titulo="PANEL FISICO", subtitulo="SISTEMA MULTI-COLUMNA"),
                          "hud_preview_grup")
    agregar_firma(plotter)
    plotter.screenshot(salida_png)
    plotter.close()

def mostrar_resumen_configuracion() -> None:
    print("\nRESUMEN DE CONFIGURACION ACTIVA")
    print("-" * 30)
    print(f"Modo:             {CONFIG_ACTIVA['nombre']}")
    print(f"Resolucion Video: {CONFIG_ACTIVA['resolucion_video']}")
    print(f"FPS:              {CONFIG_ACTIVA['fps']}")
    print(f"Duracion:         {CONFIG_ACTIVA['duracion_seg']} s")
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

    print("\nINICIALIZANDO PIPELINE DE RENDER - COMPOSICION CORREGIDA")
    print("=" * 70)

    print("\n[1/4] Generando preview de columna central...")
    render_preview_columna_central()
    sufijo = CONFIG_ACTIVA["sufijo_archivo"]
    print(f"      Preview columna central guardada: preview_columna_central{sufijo}.png")

    print("\n[2/4] Generando preview grupal...")
    render_preview_grupal()
    print(f"      Preview grupal guardada: preview_grupal{sufijo}.png")

    print("\n[3/4] Iniciando render de video individual...")
    render_animacion_columna_central()
    print(f"      Video de columna individual completado: animacion_columna_central{sufijo}.mp4")

    print("\n[4/4] Iniciando render de video grupal...")
    render_animacion_grupal()
    print(f"      Video grupal completado: animacion_grupal_5_columnas{sufijo}.mp4")

    print("\n" + "=" * 70)
    print("RENDER FINALIZADO CORRECTAMENTE")
    print("=" * 70)

if __name__ == "__main__":
    main()
