import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. ENVIRONMENT CONFIGURATION ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- 2. PLOTTING STYLE SETTINGS ---
plt.style.use('dark_background')

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'lines.linewidth': 2
})

# Define custom colors
COLOR_MAP = {
    'GALAXY_ID': 'green',      # Green
    'QSO': 'tab:red',             # Red
    'GALAXY_OOD1': 'yellow',      # Requested
    'GALAXY_OOD2': 'gold',        # Variant of yellow
    'GALAXY_OOD': 'yellow'        # Fallback
}

def get_smart_limits(data, padding=0.05):
    """
    Calculates axis limits based on data percentiles to filter outliers.
    """
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return 0, 1
    
    low = np.percentile(valid_data, 1)
    high = np.percentile(valid_data, 99)
    
    span = high - low
    return low - span * padding, high + span * padding

def get_color(type_name):
    """Returns the specific color for a type, or white if unknown."""
    return COLOR_MAP.get(type_name, 'white')

def plot_dataset(file_path):
    """
    Reads the HDF5 file and generates a PDF with 4 diagnostic plots.
    """
    print(f"Reading file: {file_path} ...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Load data
    with pd.HDFStore(file_path, 'r') as store:
        df = store['data']

    # --- Setup Output Directory ---
    # Create ../pdf/ relative to current script execution dir
    output_dir = os.path.join(os.getcwd(), '..', 'pdf')
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(file_path).replace('.h5', '')
    output_pdf = os.path.join(output_dir, f"{base_name}_plots.pdf")
    
    print(f"Generating plots in: {output_pdf}")

    # Prepare figure
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(f'Diagnostic Plots: {base_name}', fontsize=22, weight='bold')

    unique_types = np.sort(df['TYPE'].unique())

    # --- PLOT 1: Z Distribution (Histogram) ---
    # Constraint: Plot only up to Z=1.7
    ax1 = axs[0, 0]
    z_max_limit = 1.8
    bins = np.linspace(0, z_max_limit, 50) # Fixed bins up to 1.7

    for t in unique_types:
        subset = df[df['TYPE'] == t]
        if 'Z' in subset.columns:
            data_z = subset['Z'].dropna()
            # Filter for the text count, but hist handles range via bins/range arg
            count = len(data_z)
            ax1.hist(data_z, bins=bins, density=True, histtype='step', 
                     label=f"{t} (N={count})", linewidth=2.5, color=get_color(t),
                     range=(0, z_max_limit))
    
    ax1.set_xlim(0, z_max_limit)
    ax1.set_xlabel('Redshift (Z)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Redshift Distribution (Z < {z_max_limit})')
    ax1.legend(loc='upper right')
    
    # --- PLOT 2: Color-Magnitude Diagram ---
    # X axis: Magnitude (MAG_i)
    # Y axis: Color (MAG_G - MAG_R)
    ax2 = axs[0, 1]
    
    if all(col in df.columns for col in ['MAG_G', 'MAG_R', 'MAG_i']):
        x_data = df['MAG_i']
        y_data = df['MAG_G'] - df['MAG_R']
        
        for t in unique_types:
            mask = df['TYPE'] == t
            ax2.scatter(x_data[mask], y_data[mask], s=10, alpha=0.6, 
                        label=t, color=get_color(t))
            
        ax2.set_xlabel('Magnitude (MAG_i)')
        ax2.set_ylabel('Color (MAG_G - MAG_R)')
        ax2.set_title('Color-Magnitude Diagram')
        
        # Smart limits
        xlims = get_smart_limits(x_data)
        ylims = get_smart_limits(y_data)
        
        ax2.set_ylim(ylims)
        # Invert X axis for Magnitude (Astronomical convention: brighter is lower)
        ax2.set_xlim(xlims[1], xlims[0]) 
        
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Missing Magnitude Columns", ha='center')

    # --- PLOT 3: LOGSFR vs LOGM (Colored by TYPE) ---
    ax3 = axs[1, 0]
    if all(col in df.columns for col in ['LOGM', 'LOGSFR']):
        for t in unique_types:
            mask = df['TYPE'] == t
            subset = df[mask]
            ax3.scatter(subset['LOGM'], subset['LOGSFR'], s=10, alpha=0.6, 
                        label=t, color=get_color(t))
        
        ax3.set_xlabel('Log Mass (LOGM)')
        ax3.set_ylabel('Log SFR (LOGSFR)')
        ax3.set_title('SFR vs Mass (by TYPE)')
        
        ax3.set_xlim(get_smart_limits(df['LOGM']))
        ax3.set_ylim(get_smart_limits(df['LOGSFR']))
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Missing LOGM/LOGSFR Columns", ha='center')

    # --- PLOT 4: LOGSFR vs LOGM (Colored by Z) ---
    # Constraint: Colorbar max at 1.7
    ax4 = axs[1, 1]
    if all(col in df.columns for col in ['LOGM', 'LOGSFR', 'Z']):
        
        # Plot all objects
        # vmin=0, vmax=1.7 ensures the color scale is fixed to that range
        sc = ax4.scatter(df['LOGM'], df['LOGSFR'], c=df['Z'], s=5, alpha=0.6, 
                         cmap='plasma', vmin=0, vmax=z_max_limit)
        
        ax4.set_xlabel('Log Mass (LOGM)')
        ax4.set_ylabel('Log SFR (LOGSFR)')
        ax4.set_title(f'SFR vs Mass (Colored by Z < {z_max_limit})')
        
        ax4.set_xlim(get_smart_limits(df['LOGM']))
        ax4.set_ylim(get_smart_limits(df['LOGSFR']))
        
        cbar = plt.colorbar(sc, ax=ax4)
        cbar.set_label('Redshift (Z)')
    else:
        ax4.text(0.5, 0.5, "Missing LOGM/LOGSFR/Z Columns", ha='center')

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()
    print("Done.")

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Photo-Z Datasets")
    default_path = '/home/users/dae/gimarso/AI_course_photoz/data/validation_set.h5'
    
    parser.add_argument('--file', type=str, default=default_path,
                        help='Path to the .h5 file to visualize')

    args = parser.parse_args()
    
    plot_dataset(args.file)
