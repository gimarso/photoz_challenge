import os


import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 2. PLOTTING STYLE SETTINGS ---
plt.style.use('dark_background')

# Personalización basada en instrucciones previas
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 25,
    'lines.linewidth': 3,
    'axes.grid': False,      # Eliminar grids
    'legend.loc': 'lower left' # Leyendas abajo a la izquierda
})

# Definición de colores para los tipos de objetos
COLOR_MAP = {
    'GALAXY_ID': 'green',
    'QSO': 'tab:red',
    'GALAXY_OOD1': 'yellow',
    'GALAXY_OOD2': 'gold',
    'GALAXY_OOD': 'yellow'
}

def get_smart_limits(data, padding=0.05):
    """Calcula límites de los ejes basados en percentiles para filtrar outliers."""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return 0, 1
    
    low = np.percentile(valid_data, 1)
    high = np.percentile(valid_data, 99)
    
    span = high - low
    return low - span * padding, high + span * padding

def get_color(type_name):
    """Devuelve el color específico para un tipo de objeto."""
    return COLOR_MAP.get(type_name, 'white')

def plot_dataset(file_path):
    """Lee el archivo HDF5 y genera un PDF con 4 gráficos de diagnóstico actualizados."""
    print(f"Reading file: {file_path} ...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Carga de datos
    with pd.HDFStore(file_path, 'r') as store:
        df = store['data']

    # --- Setup Output Directory ---
    output_dir = os.path.join(os.getcwd(), './pdf')
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(file_path).replace('.h5', '')
    output_pdf = os.path.join(output_dir, f"{base_name}_plots.pdf")
    
    print(f"Generating plots in: {output_pdf}")

    fig, axs = plt.subplots(2, 2, figsize=(22, 20))

    unique_types = np.sort(df['TYPE'].unique())

    # --- PLOT 1: Z Distribution (Histograma completo) ---
    ax1 = axs[0, 0]
    if 'Z' in df.columns:
        z_data_all = df['Z'].dropna()
        # Usamos todo el rango disponible sin cortes arbitrarios
        bins = np.linspace(z_data_all.min(), z_data_all.max(), 60)

        for t in unique_types:
            subset = df[df['TYPE'] == t]
            data_z = subset['Z'].dropna()
            ax1.hist(data_z, bins=bins, density=True, histtype='step', 
                     label=f"{t} ($N$={len(data_z)})", color=get_color(t))
    
    ax1.set_xlabel(r'Redshift ($z$)')
    ax1.set_ylabel(r'Density')
    ax1.set_title(r'Redshift Distribution')
    ax1.legend(loc='upper right')
    
    # --- PLOT 2: Color-Magnitude Diagram (Óptico: i vs g-r) ---
    ax2 = axs[0, 1]
    if all(col in df.columns for col in ['MAG_G', 'MAG_R', 'MAG_i']):
        x_val = df['MAG_i']
        y_val = df['MAG_G'] - df['MAG_R']
        
        for t in unique_types:
            mask = df['TYPE'] == t
            ax2.scatter(x_val[mask], y_val[mask], s=15, alpha=0.5, 
                        label=t, color=get_color(t))
            
        ax2.set_xlabel(r'$m_i$')
        ax2.set_ylabel(r'$m_g - m_r$')
        ax2.set_title('Color-Magnitude: Optical')
        
        xlims = get_smart_limits(x_val)
        ylims = get_smart_limits(y_val)
        ax2.set_ylim(ylims)
        ax2.set_xlim(xlims[1], xlims[0]) # Invertir magnitud (brillante a la izquierda)
        ax2.legend(loc='upper left')
        
    else:
        ax2.text(0.5, 0.5, "Missing Optical Columns", ha='center')

    # --- PLOT 3: Color-Magnitude Diagram (GALEX: NUV vs FUV-NUV) ---
    ax3 = axs[1, 0]
    if all(col in df.columns for col in ['MAG_FUV', 'MAG_NUV']):
        x_val = df['MAG_NUV']
        y_val = df['MAG_FUV'] - df['MAG_NUV']
        
        for t in unique_types:
            mask = df['TYPE'] == t
            ax3.scatter(x_val[mask], y_val[mask], s=15, alpha=0.5, 
                        label=t, color=get_color(t))
        
        ax3.set_xlabel(r'$m_{NUV}$')
        ax3.set_ylabel(r'$m_{FUV} - m_{NUV}$')
        ax3.set_title('Color-Magnitude: GALEX (UV)')
        
        xlims = get_smart_limits(x_val)
        ylims = get_smart_limits(y_val)
        ax3.set_ylim(ylims)
        ax3.set_xlim(xlims[1], xlims[0])
        ax3.legend(loc='upper left')
    else:
        ax3.text(0.5, 0.5, "Missing GALEX Columns", ha='center')

    # --- PLOT 4: Color-Magnitude Diagram (WISE: W1 vs W1-W2) ---
    ax4 = axs[1, 1]
    if all(col in df.columns for col in ['MAG_W1', 'MAG_W2']):
        x_val = df['MAG_W1']
        y_val = df['MAG_W1'] - df['MAG_W2']
        
        for t in unique_types:
            mask = df['TYPE'] == t
            ax4.scatter(x_val[mask], y_val[mask], s=15, alpha=0.5, 
                         label=t, color=get_color(t))
        
        ax4.set_xlabel(r'$m_{W1}$')
        ax4.set_ylabel(r'$m_{W1} - m_{W2}$')
        ax4.set_title('Color-Magnitude: WISE (IR)')
        
        xlims = get_smart_limits(x_val)
        ylims = get_smart_limits(y_val)
        ax4.set_ylim(ylims)
        ax4.set_xlim(xlims[1], xlims[0])
        ax4.legend(loc='upper right')
    else:
        ax4.text(0.5, 0.5, "Missing WISE Columns", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_pdf)
    plt.close()
    print("Done.")

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Photo-Z Datasets")
    default_path = './data/validation_set.h5'
    
    parser.add_argument('--file', type=str, default=default_path,
                        help='Path to the .h5 file to visualize')

    args = parser.parse_args()
    
    plot_dataset(args.file)