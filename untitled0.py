import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_superimposed_comparison(voltages_python, voltages_matlab, results_dir):
    """
    Create superimposed plot comparing Python and MATLAB voltage profiles
    ENSURES IMAGE FILES ARE CREATED AND SAVED TO DISK
    """
    print("\n" + "="*70)
    print("GENERATING SUPERIMPOSED COMPARISON PLOTS...")
    print("="*70)
    
    bus_numbers = np.arange(1, len(voltages_python) + 1)
    Vmag_python = np.abs(voltages_python)
    Vmag_matlab = np.abs(voltages_matlab)
    
    # ===============================
    # SUPERIMPOSED VOLTAGE MAGNITUDE PLOT
    # ===============================
    print("\n[1/3] Creating superimposed voltage profile plot...")
    
    fig = plt.figure(figsize=(12, 7))
    
    plt.plot(bus_numbers, Vmag_python, marker='o', linewidth=2.5, 
             label='Python Load Flow', color='#2E86AB', markersize=6)
    plt.plot(bus_numbers, Vmag_matlab, marker='s', linewidth=2.5, 
             label='MATLAB Load Flow', color='#A23B72', markersize=6, linestyle='--')
    
    plt.xlabel("Bus Number", fontsize=12, fontweight='bold')
    plt.ylabel("Voltage Magnitude (p.u.)", fontsize=12, fontweight='bold')
    plt.title("Superimposed Voltage Magnitude Profile - Python vs MATLAB (33-Bus System)", 
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    
    ymin = min(Vmag_python.min(), Vmag_matlab.min()) - 0.02
    ymax = max(Vmag_python.max(), Vmag_matlab.max()) + 0.02
    plt.ylim([ymin, ymax])
    plt.xticks(np.linspace(1, len(bus_numbers), 33, dtype=int), rotation=45)
    
    plt.tight_layout()
    
    # SAVE WITH ERROR CHECKING
    superimposed_path = os.path.join(results_dir, "voltage_magnitude_superimposed.png")
    try:
        plt.savefig(superimposed_path, dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)
        
        # Verify file exists and has content
        if os.path.exists(superimposed_path):
            file_size = os.path.getsize(superimposed_path)
            if file_size > 0:
                print(f"   ✓ SUCCESS: Saved to {superimposed_path}")
                print(f"             File size: {file_size / 1024:.2f} KB")
            else:
                print(f"   ✗ ERROR: File created but is empty!")
        else:
            print(f"   ✗ ERROR: File was not created!")
    except Exception as e:
        print(f"   ✗ ERROR saving superimposed plot: {str(e)}")
        plt.close(fig)
    
    # ===============================
    # VOLTAGE MAGNITUDE DIFFERENCE PLOT
    # ===============================
    print("[2/3] Creating voltage difference bar chart...")
    
    voltage_difference = np.abs(Vmag_python - Vmag_matlab)
    
    fig2 = plt.figure(figsize=(12, 7))
    plt.bar(bus_numbers, voltage_difference, color='#F18F01', alpha=0.7, 
            edgecolor='black', linewidth=1)
    
    plt.xlabel("Bus Number", fontsize=12, fontweight='bold')
    plt.ylabel("Voltage Difference (p.u.)", fontsize=12, fontweight='bold')
    plt.title("Voltage Magnitude Difference: |Python - MATLAB| (33-Bus System)", 
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    mean_diff = voltage_difference.mean()
    plt.axhline(y=mean_diff, color='red', linestyle='--', linewidth=2, 
                label=f'Mean Difference: {mean_diff:.8f}')
    plt.legend(fontsize=11)
    
    plt.xticks(np.linspace(1, len(bus_numbers), 33, dtype=int), rotation=45)
    plt.tight_layout()
    
    # SAVE WITH ERROR CHECKING
    error_plot_path = os.path.join(results_dir, "voltage_magnitude_difference.png")
    try:
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight', format='png')
        plt.close(fig2)
        
        # Verify file exists and has content
        if os.path.exists(error_plot_path):
            file_size = os.path.getsize(error_plot_path)
            if file_size > 0:
                print(f"SUCCESS: Saved to {error_plot_path}")
                print(f"File size: {file_size / 1024:.2f} KB")
            else:
                print(f"ERROR: File created but is empty!")
        else:
            print(f"ERROR: File was not created!")
    except Exception as e:
        print(f"ERROR saving difference plot: {str(e)}")
        plt.close(fig2)
    
    # ===============================
    # PRINT STATISTICS
    # ===============================
    print("[3/3] Generating comparison statistics...")
    
    max_error = voltage_difference.max()
    mean_error = voltage_difference.mean()
    rmse = np.sqrt(np.mean(voltage_difference ** 2))
    
    print("\n" + "="*70)
    print("VOLTAGE PROFILE COMPARISON STATISTICS")
    print("="*70)
    print(f"Maximum Voltage Difference: {max_error:.10f} p.u.")
    print(f"Mean Voltage Difference:    {mean_error:.10f} p.u.")
    print(f"RMSE (Root Mean Square):    {rmse:.10f} p.u.")
    print(f"Total Buses Compared:       {len(bus_numbers)}")
    print("="*70)
    
    # ===============================
    # SAVE COMPARISON CSV
    # ===============================
    print("\nSaving detailed comparison data to CSV...")
    
    comparison_df = pd.DataFrame({
        "Bus": np.arange(1, len(bus_numbers) + 1),
        "Python_Voltage_Mag": Vmag_python,
        "MATLAB_Voltage_Mag": Vmag_matlab,
        "Difference": voltage_difference
    })
    
    comparison_csv_path = os.path.join(results_dir, "voltage_comparison.csv")
    try:
        comparison_df.to_csv(comparison_csv_path, index=False)
        if os.path.exists(comparison_csv_path):
            print(f"SUCCESS: Saved to {comparison_csv_path}")
        else:
            print(f"ERROR: CSV file was not created!")
    except Exception as e:
        print(f"   ✗ ERROR saving CSV: {str(e)}")
    
    print("\n" + "="*70)
    print("PLOT GENERATION COMPLETE")
    print("="*70 + "\n")
