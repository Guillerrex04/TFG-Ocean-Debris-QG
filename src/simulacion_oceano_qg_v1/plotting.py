import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import animation
import numpy as np
import cupy as cp
import shutil
import importlib
from importlib.util import find_spec
from grid import Grid


def _get_plot_coords(grid, centered: bool = False):
    """Returns (X_plot, Y_plot, x_label, y_label) in km.
    
    Physical:  x in [0, Lx], y in [0, Ly], labels 'x [km]', 'y [km]'
    Centered:  xc in [-Lx/2, Lx/2], yc in [-Ly/2, Ly/2], labels 'xc [km]', 'yc [km]'
    """
    X = cp.asnumpy(grid.X) if isinstance(grid.X, cp.ndarray) else grid.X
    Y = cp.asnumpy(grid.Y) if isinstance(grid.Y, cp.ndarray) else grid.Y
    if centered:
        cx, cy = grid.lx / 2, grid.ly / 2
        return (X - cx) / 1e3, (Y - cy) / 1e3, 'xc [km]', 'yc [km]'
    return X / 1e3, Y / 1e3, 'x [km]', 'y [km]'


def _save_figure(fig, save_path, dpi=None):
    """Guarda figura en formato correcto (SVG por defecto para estáticos)."""
    if save_path:
        ext = save_path.lower()
        if ext.endswith('.png'):
            fmt = 'png'
        elif ext.endswith('.pdf'):
            fmt = 'pdf'
        elif ext.endswith('.eps'):
            fmt = 'eps'
        else:
            fmt = 'svg'
            if not save_path.lower().endswith('.svg'):
                save_path = save_path.rsplit('.', 1)[0] + '.svg'
        if dpi is not None:
            plt.savefig(save_path, format=fmt, dpi=dpi)
        else:
            plt.savefig(save_path, format=fmt)


def set_plot_style(profile: str = 'paper', use_seaborn: bool = False):
    """Configura estilos visuales predefinidos para memoria o presentación.
    
    profile:
        - 'paper': estilo sobrio para memoria/documento.
        - 'presentation': estilo más contrastado y legible en proyector.
    """
    if use_seaborn:
        if find_spec('seaborn') is not None:
            sns = importlib.import_module('seaborn')
            sns.set_theme(style='whitegrid', context='talk' if profile == 'presentation' else 'paper')
    
    common = {
        'figure.autolayout': True,
        'savefig.bbox': 'tight',
        'savefig.format': 'svg',
        'axes.grid': True,
        'grid.alpha': 0.18,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.2,
        'image.interpolation': 'bicubic',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'lines.antialiased': True,
        'patch.antialiased': True
    }
    
    paper = {
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.titleweight': 'normal',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2.0,
        'lines.markersize': 5,
        'savefig.dpi': 200,
        'axes.prop_cycle': cycler(color=['#1f77b4', '#FF6B35', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    }
    
    presentation = {
        'font.family': 'DejaVu Sans',
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'axes.titleweight': 'semibold',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#cccccc',
        'lines.linewidth': 3.0,
        'lines.markersize': 8,
        'savefig.dpi': 240,
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#fafbfc',
        'axes.edgecolor': '#b0b0b0',
        'axes.prop_cycle': cycler(color=['#0A84FF', '#FF6B35', '#00C896', '#FF3B30', '#9D5FFF', '#FF9500'])
    }
    
    if profile not in ('paper', 'presentation'):
        raise ValueError("profile debe ser 'paper' o 'presentation'")
    
    style = common.copy()
    style.update(presentation if profile == 'presentation' else paper)
    plt.rcParams.update(style)


def _configure_ffmpeg():
    """Configura matplotlib para usar el binario de ffmpeg incluido en imageio-ffmpeg."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        mpl.rcParams['animation.ffmpeg_path'] = get_ffmpeg_exe()
        return
    except ImportError:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path
            return
    raise RuntimeError(
        "No se encontró ffmpeg. Ejecuta el proyecto con el venv de TFG o instala imageio-ffmpeg/ffmpeg."
    )


def _render_fields(ax, grid: Grid, zeta, psi, title: str, centered: bool = False):
    """Dibuja en un eje la vorticidad y los contornos de psi."""
    Xp, Yp, xlab, ylab = _get_plot_coords(grid, centered)
    zeta_np = cp.asnumpy(zeta) if isinstance(zeta, cp.ndarray) else zeta
    psi_np = cp.asnumpy(psi) if isinstance(psi, cp.ndarray) else psi
    
    vmax = np.max(np.abs(zeta_np)) if np.any(zeta_np) else 1.0
    im = ax.pcolormesh(Xp, Yp, zeta_np,
                       cmap='twilight_shifted', vmin=-vmax, vmax=vmax, shading='auto')
    ax.contour(Xp, Yp, psi_np, colors='#2d2d2d', linewidths=1.0, alpha=0.5, levels=12)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')
    return im


def _render_complete(ax, grid: Grid, zeta, psi, px, py, t_days: float, title: str, centered: bool = False):
    """Dibuja en un eje los campos y la nube de partículas."""
    Xp, Yp, xlab, ylab = _get_plot_coords(grid, centered)
    zeta_np = cp.asnumpy(zeta) if isinstance(zeta, cp.ndarray) else zeta
    psi_np = cp.asnumpy(psi) if isinstance(psi, cp.ndarray) else psi
    px_np = cp.asnumpy(px) if isinstance(px, cp.ndarray) else px
    py_np = cp.asnumpy(py) if isinstance(py, cp.ndarray) else py
    
    if centered:
        cx, cy = grid.lx / 2, grid.ly / 2
        pxp, pyp = (px_np - cx) / 1e3, (py_np - cy) / 1e3
    else:
        pxp, pyp = px_np / 1e3, py_np / 1e3
    
    vmax = np.max(np.abs(zeta_np)) if np.any(zeta_np) else 1.0
    im = ax.pcolormesh(Xp, Yp, zeta_np,
                       cmap='twilight_shifted', vmin=-vmax, vmax=vmax, shading='auto')
    ax.contour(Xp, Yp, psi_np, colors='#333333', linewidths=0.8, alpha=0.4, levels=12)
    ax.scatter(pxp, pyp, c='#B50000', s=6, edgecolors='#5C0000', linewidths=0.3, alpha=0.8, label='Restos flotantes')
    ax.text(0.02, 0.95, f'Día: {t_days:.2f}', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc', boxstyle='round,pad=0.5'))
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')
    return im


def plot_fields(grid: Grid, zeta: np.ndarray, psi: np.ndarray, title: str = "Campos Eulerianos", save_path: str = None, centered: bool = False):
    """Vorticidad (twilight mejorado) y contornos de Función de Corriente."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = _render_fields(ax, grid, zeta, psi, title, centered=centered)
    cbar = plt.colorbar(im, ax=ax, label=r'Vorticidad $\zeta$ [s$^{-1}$]', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    _save_figure(fig, save_path)
    return fig, ax


def plot_streamfunction(grid: Grid, psi, title: str = "Función de Corriente", save_path: str = None, centered: bool = False):
    """Mapa dedicado de función de corriente psi con contornos rellenos."""
    Xp, Yp, xlab, ylab = _get_plot_coords(grid, centered)
    psi_np = cp.asnumpy(psi) if isinstance(psi, cp.ndarray) else psi
    
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    vmax = np.max(np.abs(psi_np)) if np.any(psi_np) else 1.0
    im = ax.pcolormesh(Xp, Yp, psi_np,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    ax.contour(Xp, Yp, psi_np, colors='#2d2d2d', linewidths=0.9, alpha=0.5, levels=14)
    cbar = plt.colorbar(im, ax=ax, label=r'$\psi$ [m$^2$/s]', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')
    _save_figure(fig, save_path)
    return fig, ax


def plot_velocity_field(grid: Grid, u, v, save_path: str = None, centered: bool = False):
    """Mapa de velocidad usando streamplot coloreado por módulo."""
    x = cp.asnumpy(grid.x) if isinstance(grid.x, cp.ndarray) else grid.x
    y = cp.asnumpy(grid.y) if isinstance(grid.y, cp.ndarray) else grid.y
    u_np = cp.asnumpy(u) if isinstance(u, cp.ndarray) else u
    v_np = cp.asnumpy(v) if isinstance(v, cp.ndarray) else v
    
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    speed = np.sqrt(u_np**2 + v_np**2)
    if centered:
        cx, cy = grid.lx / 2, grid.ly / 2
        xp, yp = (x - cx) / 1e3, (y - cy) / 1e3
        xlab, ylab = 'xc [km]', 'yc [km]'
    else:
        xp, yp = x / 1e3, y / 1e3
        xlab, ylab = 'x [km]', 'y [km]'
    strm = ax.streamplot(xp, yp, u_np.T, v_np.T, color=speed.T, 
                         cmap='plasma', linewidth=1.3, density=1.3, arrowsize=1.5)
    cbar = plt.colorbar(strm.lines, ax=ax, label='Velocidad [m/s]', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title("Campo de Velocidades Geostróficas", pad=12)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')
    _save_figure(fig, save_path)
    return fig, ax


def plot_snapshot_complete(grid: Grid, zeta, psi, px, py, t_days: float, save_path: str = None, centered: bool = False):
    """Composición final para la memoria: Campos + Nube de Partículas."""
    Xp, Yp, xlab, ylab = _get_plot_coords(grid, centered)
    zeta_np = cp.asnumpy(zeta) if isinstance(zeta, cp.ndarray) else zeta
    psi_np = cp.asnumpy(psi) if isinstance(psi, cp.ndarray) else psi
    px_np = cp.asnumpy(px) if isinstance(px, cp.ndarray) else px
    py_np = cp.asnumpy(py) if isinstance(py, cp.ndarray) else py
    
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = _render_complete(ax, grid, zeta_np, psi_np, px_np, py_np, t_days, "Transporte Lagrangiano en el flujo QG", centered=centered)
    cbar = plt.colorbar(im, ax=ax, label=r'$\zeta$ [s$^{-1}$]', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=11)
    _save_figure(fig, save_path)
    return fig, ax


def save_fields_animation(grid, zeta_frames, psi_frames, t_days, save_path: str, fps: int = 6, dpi: int = 150, centered: bool = False):
    """Guarda una animación MP4 de vorticidad y función de corriente."""
    _configure_ffmpeg()
    Xp, Yp, xlab, ylab = _get_plot_coords(grid, centered)

    all_zeta = np.concatenate([cp.asnumpy(z).ravel() if isinstance(z, cp.ndarray) else z.ravel() for z in zeta_frames])
    vmax_zeta = np.percentile(np.abs(all_zeta), 99.5) if np.any(all_zeta) else 1.0

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    im = ax.pcolormesh(Xp, Yp, cp.asnumpy(zeta_frames[0]) if isinstance(zeta_frames[0], cp.ndarray) else zeta_frames[0],
                       cmap='twilight_shifted', vmin=-vmax_zeta, vmax=vmax_zeta, shading='auto')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, label=r'Vorticidad $\zeta$ [s$^{-1}$]')
    levels_psi = 12
    cset = []
    title_artist = ax.set_title(f"Campos Eulerianos - Día {t_days[0]:.2f}", pad=12)

    def animate(idx):
        zeta = cp.asnumpy(zeta_frames[idx]) if isinstance(zeta_frames[idx], cp.ndarray) else zeta_frames[idx]
        psi = cp.asnumpy(psi_frames[idx]) if isinstance(psi_frames[idx], cp.ndarray) else psi_frames[idx]
        im.set_array(zeta.ravel())
        if cset:
            for c in cset:
                c.remove()
        cset_new = ax.contour(Xp, Yp, psi, colors='#2d2d2d', linewidths=1.0, alpha=0.5, levels=levels_psi)
        cset[:] = [cset_new]
        title_artist.set_text(f"Campos Eulerianos - Día {t_days[idx]:.2f}")

    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    anim = animation.FuncAnimation(fig, animate, frames=len(zeta_frames), blit=False)
    anim.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)


def save_streamfunction_animation(grid, psi_frames, t_days, save_path: str, fps: int = 6, dpi: int = 150, centered: bool = False):
    """Guarda una animación MP4 dedicada a la función de corriente psi."""
    _configure_ffmpeg()
    Xp, Yp, xlab, ylab = _get_plot_coords(grid, centered)

    all_psi = np.concatenate([cp.asnumpy(p).ravel() if isinstance(p, cp.ndarray) else p.ravel() for p in psi_frames])
    vmax_psi = np.percentile(np.abs(all_psi), 99.5) if np.any(all_psi) else 1.0

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    im = ax.pcolormesh(Xp, Yp, cp.asnumpy(psi_frames[0]) if isinstance(psi_frames[0], cp.ndarray) else psi_frames[0],
                       cmap='coolwarm', vmin=-vmax_psi, vmax=vmax_psi, shading='auto')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, label=r'$\psi$ [m$^2$/s]')
    levels_psi = 12
    cset = []
    title_artist = ax.set_title(f"Función de Corriente - Día {t_days[0]:.2f}")

    def animate(idx):
        psi = cp.asnumpy(psi_frames[idx]) if isinstance(psi_frames[idx], cp.ndarray) else psi_frames[idx]
        im.set_array(psi.ravel())
        if cset:
            for c in cset:
                c.remove()
        cset_new = ax.contour(Xp, Yp, psi, colors='k', linewidths=0.6, alpha=0.45, levels=levels_psi)
        cset[:] = [cset_new]
        title_artist.set_text(f"Función de Corriente - Día {t_days[idx]:.2f}")

    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    anim = animation.FuncAnimation(fig, animate, frames=len(psi_frames), blit=False)
    anim.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)


def save_complete_animation(grid, zeta_frames, psi_frames, px_frames, py_frames, t_days, save_path: str, fps: int = 6, dpi: int = 150, centered: bool = False):
    """Guarda una animación MP4 con campos y partículas."""
    _configure_ffmpeg()
    Xp, Yp, xlab, ylab = _get_plot_coords(grid, centered)

    all_zeta = np.concatenate([cp.asnumpy(z).ravel() if isinstance(z, cp.ndarray) else z.ravel() for z in zeta_frames])
    vmax_zeta = np.percentile(np.abs(all_zeta), 99.5) if np.any(all_zeta) else 1.0

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    im = ax.pcolormesh(Xp, Yp, cp.asnumpy(zeta_frames[0]) if isinstance(zeta_frames[0], cp.ndarray) else zeta_frames[0],
                       cmap='twilight_shifted', vmin=-vmax_zeta, vmax=vmax_zeta, shading='auto')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, label=r'$\zeta$ [s$^{-1}$]')
    levels_psi = 12

    px0 = cp.asnumpy(px_frames[0]) if isinstance(px_frames[0], cp.ndarray) else px_frames[0]
    py0 = cp.asnumpy(py_frames[0]) if isinstance(py_frames[0], cp.ndarray) else py_frames[0]
    if centered:
        cx, cy = grid.lx / 2, grid.ly / 2
        px0_plot, py0_plot = (px0 - cx) / 1e3, (py0 - cy) / 1e3
        _shift = (cx, cy)
    else:
        px0_plot, py0_plot = px0 / 1e3, py0 / 1e3
        _shift = (0.0, 0.0)

    scatter_artist = ax.scatter(px0_plot, py0_plot,
        c='#B50000', s=4, edgecolors='#5C0000', linewidths=0.2, alpha=0.8
    )
    cset = []
    title_artist = ax.set_title(f"Transporte Lagrangiano - Día {t_days[0]:.2f}", pad=12)

    def animate(idx):
        zeta = cp.asnumpy(zeta_frames[idx]) if isinstance(zeta_frames[idx], cp.ndarray) else zeta_frames[idx]
        psi = cp.asnumpy(psi_frames[idx]) if isinstance(psi_frames[idx], cp.ndarray) else psi_frames[idx]
        px = cp.asnumpy(px_frames[idx]) if isinstance(px_frames[idx], cp.ndarray) else px_frames[idx]
        py = cp.asnumpy(py_frames[idx]) if isinstance(py_frames[idx], cp.ndarray) else py_frames[idx]
        im.set_array(zeta.ravel())
        if cset:
            for c in cset:
                c.remove()
        cset_new = ax.contour(Xp, Yp, psi, colors='#333333', linewidths=0.8, alpha=0.4, levels=levels_psi)
        cset[:] = [cset_new]
        offsets = np.column_stack([(px - _shift[0]) / 1e3, (py - _shift[1]) / 1e3])
        scatter_artist.set_offsets(offsets)
        title_artist.set_text(f"Transporte Lagrangiano - Día {t_days[idx]:.2f}")

    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    anim = animation.FuncAnimation(fig, animate, frames=len(zeta_frames), blit=False)
    anim.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)


def save_velocity_animation(grid, u_frames, v_frames, t_days, save_path: str, fps: int = 6, dpi: int = 150, centered: bool = False):
    """Guarda una animación MP4 del campo de velocidades (streamplot coloreado)."""
    _configure_ffmpeg()
    x = cp.asnumpy(grid.x) if isinstance(grid.x, cp.ndarray) else grid.x
    y = cp.asnumpy(grid.y) if isinstance(grid.y, cp.ndarray) else grid.y
    if centered:
        cx, cy = grid.lx / 2, grid.ly / 2
        xp, yp = (x - cx) / 1e3, (y - cy) / 1e3
        xlab, ylab = 'xc [km]', 'yc [km]'
    else:
        xp, yp = x / 1e3, y / 1e3
        xlab, ylab = 'x [km]', 'y [km]'

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_aspect('equal')
    cbar_artist = fig.colorbar(ax.collections[0] if ax.collections else plt.Rectangle((0,0)), ax=ax, label='Velocidad [m/s]', pad=0.02)
    cbar_artist.ax.tick_params(labelsize=11)

    def animate(idx):
        u = cp.asnumpy(u_frames[idx]) if isinstance(u_frames[idx], cp.ndarray) else u_frames[idx]
        v = cp.asnumpy(v_frames[idx]) if isinstance(v_frames[idx], cp.ndarray) else v_frames[idx]
        ax.cla()
        speed = np.sqrt(u**2 + v**2)
        strm = ax.streamplot(xp, yp, u.T, v.T, color=speed.T,
                             cmap='plasma', linewidth=1.5, density=1.4, arrowsize=1.8)
        ax.set_title(f"Campo de Velocidades Geostróficas - Día {t_days[idx]:.2f}", fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel(xlab, fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.set_aspect('equal')

    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    anim = animation.FuncAnimation(fig, animate, frames=len(u_frames), blit=False)
    anim.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)


def plot_dispersion_metrics(t_days: np.ndarray, msd_vals: np.ndarray, r_rms_vals: np.ndarray, save_path: str = None):
    """Gráfica de evolución temporal de MSD y Radio RMS para la memoria."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(t_days, msd_vals, color='#0A84FF', lw=3, marker='o', markersize=4, markeredgecolor='#0052B4', markeredgewidth=1, alpha=0.85)
    ax1.fill_between(t_days, msd_vals, alpha=0.15, color='#0A84FF')
    ax1.set_ylabel(r'MSD [m$^2$]', fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_title("Evolución de métricas de dispersión", fontweight='bold', pad=12)
    ax2.plot(t_days, r_rms_vals, color='#FF6B35', lw=3, marker='s', markersize=4, markeredgecolor='#D62828', markeredgewidth=1, alpha=0.85)
    ax2.fill_between(t_days, r_rms_vals, alpha=0.15, color='#FF6B35')
    ax2.set_ylabel('Radio RMS [m]', fontweight='bold')
    ax2.set_xlabel('Tiempo [días]', fontweight='bold')
    ax2.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_energy_timeseries(t_days: np.ndarray, energy_vals: np.ndarray, enstrophy_vals: np.ndarray = None, save_path: str = None):
    """Evolución temporal de energía cinética y enstrofía."""
    if enstrophy_vals is not None:
        fig = plt.figure(figsize=(12, 8), dpi=200)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.plot(t_days, energy_vals, color='#2E86AB', lw=2.5, marker='', alpha=0.85)
        ax1.fill_between(t_days, energy_vals, alpha=0.15, color='#2E86AB')
        ax1.set_title('Evolución de energía y enstrofía', fontweight='bold', pad=12, fontsize=14)
        ax1.set_xlabel('Tiempo [días]', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Energía Cinética Total [J/kg]', fontweight='bold', fontsize=11)
        ax1.legend(['Energía'], loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3))

        ax2.plot(t_days, enstrophy_vals, color='#E94F37', lw=2.5, marker='', alpha=0.85)
        ax2.fill_between(t_days, enstrophy_vals, alpha=0.15, color='#E94F37')
        ax2.set_xlabel('Tiempo [días]', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Enstrofía Total [s⁻²]', fontweight='bold', fontsize=11)
        ax2.legend(['Enstrofía'], loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3))

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    else:
        fig = plt.figure(figsize=(12, 8), dpi=200)
        ax = fig.add_subplot(111)

        ax.plot(t_days, energy_vals, color='#2E86AB', lw=2.5, marker='', alpha=0.85)
        ax.fill_between(t_days, energy_vals, alpha=0.15, color='#2E86AB')
        ax.set_title('Evolución de energía y enstrofía', fontweight='bold', pad=12, fontsize=14)
        ax.set_xlabel('Tiempo [días]', fontweight='bold', fontsize=11)
        ax.set_ylabel('Energía Cinética Total [J/kg]', fontweight='bold', fontsize=11)
        ax.legend(['Energía'], loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3))

    AUTHOR_TEXT = "TFG: Simulación QG Barotrópica Oceánica | Guillermo Alba Buitrón"
    fig.text(0.12, 0.01, AUTHOR_TEXT, fontsize=9, color='gray',
             ha='left', alpha=0.7, transform=fig.transFigure)

    _save_figure(fig, save_path, dpi=200)
    return fig


def plot_wind_stress_profile(ly, tau0, save_path=None, centered=False, swap_axes=True, ny=257):
    """Perfil meridional del esfuerzo zonal del viento tau_x(y).
    
    Construcción analítica consistente con el curl usado en el modelo:
        tau_x(y) = tau0 * cos(2*pi*(y - Ly/2)/Ly)
    
    El curl del esfuerzo derivado es:
        curl(tau) = -d(tau_x)/dy = tau0*(2*pi/Ly)*sin(2*pi*(y - Ly/2)/Ly)
    que coincide exactamente con grid.Grid.get_wind_stress_curl().
    
    Así queda cero la componente tau_y y la convención curl = -d(tau_x)/dy.
    
    Parameters
    ----------
    ly : float
        Extensión meridional del dominio en metros.
    tau0 : float
        Amplitud máxima del esfuerzo del viento en N/m^2 (Pa).
    save_path : str or None
        Ruta de guardado. Si no termina en extensión se usa SVG.
    centered : bool
        Si True, el eje vertical muestra yc = y - Ly/2 en km.
        Si False (default), muestra y en [0, Ly] en km.
    swap_axes : bool
        Si True (default): horizontal = tau_x, vertical = y (perfil vertical).
        Si False: horizontal = y, vertical = tau_x (orientación matemática).
    ny : int
        Número de puntos para la malla vertical (default 257).
    """
    paper_fonts = {
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
    }
    with plt.rc_context(paper_fonts):
        y = np.linspace(0, ly, ny)
        tau_x = tau0 * np.cos(2.0 * np.pi * (y - ly / 2) / ly)
        
        if centered:
            y_plot = (y - ly / 2) / 1e3
            y_label = 'yc [km]'
        else:
            y_plot = y / 1e3
            y_label = 'y [km]'
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        if swap_axes:
            ax.plot(tau_x, y_plot, color='#0A84FF', lw=2.0)
            ax.axvline(0.0, color='gray', lw=0.8, ls='--', alpha=0.5)
            if not centered:
                ax.axhline(ly / 2 / 1e3, color='gray', lw=0.6, ls=':', alpha=0.4)
            ax.set_xlabel(r'$\tau_x$ [N m$^{-2}$]')
            ax.set_ylabel(y_label)
        else:
            ax.plot(y_plot, tau_x, color='#0A84FF', lw=2.0)
            ax.axhline(0.0, color='gray', lw=0.8, ls='--', alpha=0.5)
            if not centered:
                ax.axvline(ly / 2 / 1e3, color='gray', lw=0.6, ls=':', alpha=0.4)
            ax.set_xlabel(y_label)
            ax.set_ylabel(r'$\tau_x$ [N m$^{-2}$]')
        
        ax.set_title('Perfil meridional del esfuerzo zonal del viento', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        fig.text(0.12, 0.01, "TFG: Simulación QG Barotrópica Oceánica | Guillermo Alba Buitrón",
                 fontsize=9, color='gray', ha='left', alpha=0.7, transform=fig.transFigure)
        _save_figure(fig, save_path)
        return fig


def plot_wind_curl_profile(ly, tau0, save_path=None, centered=False, swap_axes=True, ny=257):
    """Perfil meridional del curl del esfuerzo del viento.
    
    Construcción analítica:
        curl_tau(y) = tau0*(2*pi/Ly)*sin(2*pi*(y - Ly/2)/Ly)
    que coincide con grid.Grid.get_wind_stress_curl().
    
    Parámetros: idénticos a plot_wind_stress_profile.
    """
    paper_fonts = {
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
    }
    with plt.rc_context(paper_fonts):
        y = np.linspace(0, ly, ny)
        curl_tau = tau0 * (2.0 * np.pi / ly) * np.sin(2.0 * np.pi * (y - ly / 2) / ly)
        
        if centered:
            y_plot = (y - ly / 2) / 1e3
            y_label = 'yc [km]'
        else:
            y_plot = y / 1e3
            y_label = 'y [km]'
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        if swap_axes:
            ax.plot(curl_tau, y_plot, color='#E94F37', lw=2.0)
            ax.axvline(0.0, color='gray', lw=0.8, ls='--', alpha=0.5)
            if not centered:
                ax.axhline(ly / 2 / 1e3, color='gray', lw=0.6, ls=':', alpha=0.4)
            ax.set_xlabel(r'$(\nabla \times \boldsymbol{\tau})_{z}$ [N m$^{-3}$]')
            ax.set_ylabel(y_label)
        else:
            ax.plot(y_plot, curl_tau, color='#E94F37', lw=2.0)
            ax.axhline(0.0, color='gray', lw=0.8, ls='--', alpha=0.5)
            if not centered:
                ax.axvline(ly / 2 / 1e3, color='gray', lw=0.6, ls=':', alpha=0.4)
            ax.set_xlabel(y_label)
            ax.set_ylabel(r'$(\nabla \times \boldsymbol{\tau})_{z}$ [N m$^{-3}$]')
        
        ax.set_title(
            'Perfil meridional de la componente vertical\n'
            'del rotacional del esfuerzo del viento',
            fontweight='bold',
            pad=10
        )
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.text(0.12, 0.01, "TFG: Simulación QG Barotrópica Oceánica | Guillermo Alba Buitrón",
                 fontsize=9, color='gray', ha='left', alpha=0.7, transform=fig.transFigure)
        _save_figure(fig, save_path)
        return fig