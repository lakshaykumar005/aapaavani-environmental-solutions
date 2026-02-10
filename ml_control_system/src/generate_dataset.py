import numpy as np
import pandas as pd

# Set random seed for reproducibility (but logic remains non-deterministic per requirements)
np.random.seed(42)

def generate_dataset(n_rows=240):
    # --- 1. Generate Inputs (External Conditions) ---
    # Scenario: Industrial Wastewater Treatment Plant
    
    # A: Inflow Rate (L/min) - Range: 50–500
    A = np.zeros(n_rows)
    A[0] = np.random.uniform(50, 500)
    for i in range(1, n_rows):
        A[i] = np.clip(A[i-1] + np.random.normal(0, 10), 50, 500)

    # B: pH Level (x100 for integer scale) or Chemical Demand - Range: 200–800
    B = np.zeros(n_rows)
    B[0] = np.random.uniform(200, 800)
    for i in range(1, n_rows):
        B[i] = np.clip(B[i-1] + np.random.normal(0, 15), 200, 800)

    # C: Contaminant Load / TSS (mg/L) - Range: 10–300
    C = np.random.uniform(10, 300, n_rows) # More random, less trend

    # D: Conductivity (µS/cm) - Range: 50–400
    D = np.zeros(n_rows)
    D[0] = np.random.uniform(50, 400)
    for i in range(1, n_rows):
        D[i] = np.clip(D[i-1] + np.random.normal(0, 8), 50, 400)

    # E: Rain Volume (mm) - Range: 15–35
    E = np.random.uniform(15, 35, n_rows)

    # --- 2. Generate Control Settings (Adjustable) ---
    # F: Chemical Dosage Rate (L/hr) - Range: 1.0–4.0
    F = np.random.uniform(1.0, 4.0, n_rows) 
    
    # G: Aeration Blower Speed (Hz) - Range: 30–80
    G = np.random.uniform(30.0, 80.0, n_rows) 
    
    # H: Mixer RPM (x100) - Range: 0.5–2.0
    H = np.random.uniform(0.5, 2.0, n_rows) 

    # --- 3. System Characteristics (Fixed) ---
    # I: Plant Capacity (m3) - Range 100-1000
    I_val = 600 
    # J: Biological Baseline - Range 2000-5000
    J_val = 3500 
    
    I = np.full(n_rows, I_val)
    J = np.full(n_rows, J_val)

    # --- 4. Calculate Outputs (Targets) ---
    # K: Toxicity Level (ppm) - Target < 10 (Lower is Better)
    # L: Turbidity (NTU) - Target Low (Lower is Better)
    # M: Operational Cost ($/hr) - Target Low (Lower is Better)
    
    # Design logic: 
    # - High Inputs (Inflow, Contaminants) -> Higher Stress -> Higher Toxicity/Turbidity
    # - Controls have "Optimal" points (e.g. ideal chemical dosage). Deviation -> Higher Toxicity.
    
    # Optimal Control Points (Secret ground truth):
    # F_opt = 2.5 (Ideal Chemical Dosage)
    # G_opt = 50.0 (Ideal Airflow)
    # H_opt = 1.2 (Ideal Mixing)
    
    # Normalize inputs to roughly 0-1 scale for formula weighting
    A_norm = (A - 50) / 450
    B_norm = (B - 200) / 600
    C_norm = (C - 10) / 290
    D_norm = (D - 50) / 350
    E_norm = (E - 15) / 20
    
    # Stress factor from inputs (0 to ~3-4)
    # Stress factor from inputs (0 to ~3-4)
    # Different outputs are sensitive to different inputs based on plant physics
    
    # K (Toxicity): Loads (A), pH (B), Conductivity (D)
    # Rationale: High flow reduces treatment time. Extreme pH/Conductivity kills bacteria.
    stress_K = 0.8*A_norm + 0.6*np.abs(B_norm - 0.5) + 0.4*D_norm
    
    # L (Turbidity): Flow (A), Solids (C), Rain (E)
    # Rationale: High flow/Rain scours sludge. High Solids input = High Solids output.
    stress_L = 0.5*A_norm + 1.0*C_norm + 0.5*E_norm
    
    # M (Cost): Flow (A) mostly (Pumping cost)
    # Rationale: More water = more electricity for pumps. controls F,G,H add to this later.
    stress_M = 1.0*A_norm
    
    # Penalty from suboptimal controls (always positive, 0 is best)
    # K (Toxicity) is sensitive to F (Chemical) and G (Air)
    ctrl_penalty_K = 15 * np.abs(F - 2.5) + 0.5 * np.abs(G - 50.0)
    # L (Turbidity) is sensitive to G (Air) and H (Mixing)
    ctrl_penalty_L = 0.4 * np.abs(G - 60.0) + 10 * np.abs(H - 1.2)
    # M (Cost) is sensitive to F (Chemical Cost) and H (Energy Cost)
    ctrl_penalty_M = 8 * (F - 3.0)**2 + 15 * (H - 0.8)**2

    # Noise (Non-deterministic part)
    noise_K = np.random.normal(0, 2.5, n_rows)
    noise_L = np.random.normal(0, 1.5, n_rows)
    noise_M = np.random.normal(0, 2.0, n_rows)

    # Base values derived from System Params (I, J)
    # I/20 -> 30, J/200 -> 17.5. 
    base_K = (I / 20.0) + 10 
    base_L = (J / 200.0) - 10
    base_M = (I / 40.0) + (J / 400.0)

    # Final Output Calculation
    # K: Toxicity
    K = base_K + (stress_K * 15) + ctrl_penalty_K + noise_K
    
    # L: Turbidity
    L = base_L + (stress_L * 8) + ctrl_penalty_L + noise_L
    
    # M: Cost
    M = base_M + (stress_M * 10) + ctrl_penalty_M + noise_M

    # Clipping to ensure realistic bounds
    K = np.clip(K, 10, 100)
    L = np.clip(L, 1, 40)
    M = np.clip(M, 5, 80) # Allowed to go a bit higher for "bad" states

    # --- 5. Generate Status Flag (N) ---
    # 1 = Compliant Discharge, 0 = Violation/Pollution Event
    
    N = np.ones(n_rows, dtype=int)
    
    # Vectorized condition
    # If ANY metric is too high, status becomes 0 (Bad)
    bad_condition = (K > 75) | (L > 28) | (M > 55)
    
    # Random flip: Sometimes it's bad even if metrics are OK (rare equipment failure)
    # Sometimes it's good even if metrics are slightly high (operator override)
    
    # First set strictly based on thresholds
    N[bad_condition] = 0
    
    # Add some randomness (Non-deterministic classification)
    # Flip 5% of labels to simulate edge cases and make it harder for the model
    mask_flip = np.random.choice([False, True], size=n_rows, p=[0.95, 0.05])
    N[mask_flip] = 1 - N[mask_flip] 

    # Create DataFrame
    df = pd.DataFrame({
        'Hour': np.arange(n_rows),
        'A': A, 'B': B, 'C': C, 'D': D, 'E': E,
        'F': F, 'G': G, 'H': H,
        'I': I, 'J': J,
        'K': K, 'L': L, 'M': M,
        'N': N
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_dataset()
    
    import os
    # Ensure data directory exists relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    csv_file = os.path.join(data_dir, "synthetic_control_data.csv")
    df.to_csv(csv_file, index=False)
    
    # Save to Excel
    excel_file = os.path.join(data_dir, "synthetic_control_data.xlsx")
    df.to_excel(excel_file, index=False)
    
    print(f"Dataset generated with {len(df)} rows.")
    print(f"Saved to {csv_file} and {excel_file}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Validate Ranges
    print("\nRange Validation (Min/Max):")
    cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M']
    print(df[cols].agg(['min', 'max']))
    
    print("\nStatus Distribution:")
    print(df['N'].value_counts())
