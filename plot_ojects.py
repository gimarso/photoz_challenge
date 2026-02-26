import os


import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

# --- 2. CONFIGURACIÓN DE ESTILOS Y CONSTANTES ---

# Aplicar estilo oscuro y preferencias del usuario (fuentes grandes, líneas anchas, sin grids)
plt.style.use('dark_background')
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'lines.linewidth': 3,
    'axes.grid': False
})

JPAS_BANDS = [
    'J0378','J0390','J0400','J0410','J0420','J0430','J0440','J0450',
    'J0460','J0470','J0480','J0490','J0500','J0510','J0520','J0530',
    'J0540','J0550','J0560','J0570','J0580','J0590','J0600','J0610',
    'J0620','J0630','J0640','J0650','J0660','J0670','J0680','J0690',
    'J0700','J0710','J0720','J0730','J0740','J0750','J0760','J0770',
    'J0780','J0790','J0800','J0810','J0820','J0830','J0840','J0850',
    'J0860','J0870','J0880','J0890','J0900','J0910'
]
EFF_WAVE_JPAS = {band: float(band[1:]) for band in JPAS_BANDS}

MAG_BANDS = [
    'MAG_NUV','MAG_FUV','MAG_G','MAG_R','MAG_i','MAG_Z',"MAG_J_2MASS",
    "MAG_H_2MASS","MAG_Ks_2MASS",'MAG_W1','MAG_W2','MAG_W3','MAG_W4'
]
EFF_WAVE_MAG = {
    'MAG_FUV': 151.6, 'MAG_NUV': 226.7, 'MAG_G': 477.0, 'MAG_R': 623.1, 
    'MAG_i': 762.5, 'MAG_Z': 913.4, 'MAG_J_2MASS': 1235.0, 'MAG_H_2MASS': 1662.0, 
    'MAG_Ks_2MASS': 2159.0, 'MAG_W1': 3352.0, 'MAG_W2': 4602.0, 'MAG_W3': 11560.0, 'MAG_W4': 22080.0
}

# Líneas de emisión de referencia (en nm)
REF_EMISSION_LINES = {
    r'Ly$\alpha$': 121.567, r'CIV': 154.906, r'CIII]': 190.873,
    r'MgII': 279.875, r'[OII]': 372.709, r'[OIII]': 500.684, r'H$\alpha$': 656.280
}

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- 3. FUNCIÓN DE DIBUJO ---

def plot_panel_sed(ax, row, bands_list, wave_dict, is_mag=False, is_left=False, is_bottom=False, ylim=None):
    obj_type = row.get('TYPE', 'UNKNOWN')
    z = row.get('Z', 0.0)
    mag_i = row.get('MAG_i', np.nan)

    waves, vals, errs = [], [], []
    for b in bands_list:
        val = row.get(b, np.nan)
        if np.isfinite(val):
            vals.append(val)
            waves.append(wave_dict[b])
            err_col = f"{b}_ERR"
            errs.append(row.get(err_col, 0.0))

    waves, vals, errs = np.array(waves), np.array(vals), np.array(errs)
    
    # Colormap (Rainbow)
    norm = mcolors.LogNorm(vmin=100, vmax=25000) if is_mag else mcolors.Normalize(vmin=350, vmax=950)
    cmap = plt.get_cmap('turbo')
    colors = cmap(norm(waves))

    # 1. Puntos de datos (Colores arcoíris, borde blanco)
    for i in range(len(waves)):
        ax.errorbar(waves[i], vals[i], yerr=errs[i], fmt='o', ecolor=colors[i], 
                    markerfacecolor=colors[i], markeredgecolor='white', markeredgewidth=0.5,
                    markersize=12, elinewidth=2, zorder=10)

    # 2. LÍNEAS DE EMISIÓN (Solo para J-PAS y restringidas al rango visual)
    if not is_mag and obj_type in ['GALAXY', 'QSO']:
        ymin_p, ymax_p = ylim if ylim else (np.min(vals), np.max(vals))
        for name, l_rest in REF_EMISSION_LINES.items():
            l_obs = l_rest * (1 + z)
            # Solo pintar si cae dentro del rango de 350nm a 920nm
            if 350 < l_obs < 920:
                ax.axvline(x=l_obs, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
                # Etiqueta de la línea
                y_text = ymin_p + 0.88 * (ymax_p - ymin_p)
                ax.text(l_obs, y_text, name, rotation=90, color='white', fontsize=12,
                        va='center', ha='center', backgroundcolor='black', alpha=0.7)

    # 3. Anotaciones de Texto (Tipo, Redshift y Magnitud)
    ax.text(0.05, 0.92, f"{obj_type}", transform=ax.transAxes, color='white', fontweight='bold', fontsize=20)
    ax.text(0.05, 0.82, f"z = {z:.3f}", transform=ax.transAxes, color='white', fontsize=18)
    ax.text(0.55, 0.92, f"$i = {mag_i:.2f}$", transform=ax.transAxes, color='white', fontsize=18)

    # 4. Formato de Ejes y Rango X
    if is_bottom: ax.set_xlabel(r'$\lambda$ [nm]')
    if is_left: ax.set_ylabel('Magnitude' if is_mag else 'Normalized Flux')
    
    if is_mag:
        ax.set_xscale('log')
        ax.invert_yaxis()
    
    if not is_left: ax.tick_params(labelleft=False)
    
    # RESTRICCIÓN DEL EJE X: 9200 Amstrong = 920 nm
    if not is_mag:
        ax.set_xlim(350, 920)
    else:
        ax.set_xlim(np.min(waves)*0.8, np.max(waves)*1.2)
        
    if ylim: ax.set_ylim(ylim)

def main():
    cfg = load_config()
    df_val = pd.read_hdf(cfg['data']['train_path'], key='data')
    if 'Z' not in df_val.columns:
        df_val['Z'] = np.where(df_val['SPECTYPE'] == 2.0, df_val.get('Z_QSO', 0), df_val.get('Z_GAL', 0))

    # SELECCIÓN ALEATORIA: 3 Galaxias y 3 QSOs
    sample_gal = df_val[df_val['TYPE'] == 'GALAXY'].sample(3)
    sample_qso = df_val[df_val['TYPE'] == 'QSO'].sample(3)
    plot_df = pd.concat([sample_gal, sample_qso]) # Gals arriba (fila 0), QSOs abajo (fila 1)

    output_dir = './pdf/'
    os.makedirs(output_dir, exist_ok=True)

    # --- PDF 1: J-PAS SED (Flujo con Líneas de Emisión y eje X hasta 920nm) ---
    pdf1_path = os.path.join(output_dir, "train_jpas_flux_samples.pdf")
    with PdfPages(pdf1_path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(24, 16), sharex=True)
        plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.08, right=0.97, bottom=0.1, top=0.92)
        fig.suptitle("J-PAS Photo-spectra", fontsize=30, weight='bold')

        for i, (idx, row) in enumerate(plot_df.iterrows()):
            ax = axes[i // 3, i % 3]
            # Límites Y dinámicos por fila
            row_data = plot_df.iloc[(i//3)*3 : (i//3)*3 + 3][JPAS_BANDS].values.flatten()
            row_data = row_data[np.isfinite(row_data)]
            ylim = (np.min(row_data)*0.9, np.max(row_data)*1.25) if len(row_data)>0 else None
            
            plot_panel_sed(ax, row, JPAS_BANDS, EFF_WAVE_JPAS, is_mag=False, 
                           is_left=(i%3==0), is_bottom=(i//3==1), ylim=ylim)
        pdf.savefig(fig)
        plt.close()

    # --- PDF 2: Broadband SED (Magnitudes) ---
    pdf2_path = os.path.join(output_dir, "train_broadband_mag_samples.pdf")
    with PdfPages(pdf2_path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(24, 16), sharex='row')
        plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.08, right=0.97, bottom=0.1, top=0.92)
        fig.suptitle("Broad-band Magnitude SED", fontsize=30, weight='bold')

        for i, (idx, row) in enumerate(plot_df.iterrows()):
            ax = axes[i // 3, i % 3]
            row_data = plot_df.iloc[(i//3)*3 : (i//3)*3 + 3][MAG_BANDS].values.flatten()
            row_data = row_data[np.isfinite(row_data)]
            ylim = (np.max(row_data)+1, np.min(row_data)-1) if len(row_data)>0 else None
            
            plot_panel_sed(ax, row, MAG_BANDS, EFF_WAVE_MAG, is_mag=True, 
                           is_left=(i%3==0), is_bottom=(i//3==1), ylim=ylim)
        pdf.savefig(fig)
        plt.close()

    print(f"Informes generados exitosamente en {output_dir}")

if __name__ == "__main__":
    main()