import os

# --- VARIABLE MAPPING (Business Context) ---
# Scenario: Industrial Wastewater Treatment Plant Control
VAR_MAP = {
    'A': 'A (Inflow Rate)',
    'B': 'B (pH Level)',
    'C': 'C (Contaminant Load/TSS)',
    'D': 'D (Conductivity µS/cm)',
    'E': 'E (Rain Volume)',
    'F': 'F (Chemical Dosage)',
    'G': 'G (Aeration Speed)',
    'H': 'H (Mixer RPM)',
    'I': 'I (Plant Capacity)',
    'J': 'J (Biological Baseline)',
    'K': 'K (Toxicity Level)',
    'L': 'L (Turbidity)',
    'M': 'M (Op Cost $/hr)',
    'N': 'N (Compliance Status)'
}

# Features: Inputs (A-E), Controls (F-H), Systems (I-J)
FEATURE_COLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
# Targets: Outputs to minimize
TARGET_COLS_REG = ['K', 'L', 'M']
# Target: Status to classify
TARGET_COL_CLS = 'N'

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'images')

def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
