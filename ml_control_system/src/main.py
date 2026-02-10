import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import ensure_dirs
from data_loader import load_data
from models import train_models
from optimization import demonstrate_optimization
from visualization import generate_visualizations

def main():
    print("="*60)
    print("ML Control System - Industrial Wastewater Treatment Optimization")
    print("="*60)
    
    np.random.seed(42)
    ensure_dirs()
    
    # 1. Load Data
    df = load_data()
    if df is None: return

    # 2. Train Models
    reg_models, clf_model, data_split, model_results, cls_results, conf_mx, best_cls_name = train_models(df)
    
    X_test, y_reg_test, y_cls_test = data_split[1], data_split[3], data_split[5]

    # 3. Control Optimization & Demo
    trajectories, last_example = demonstrate_optimization(X_test, y_reg_test, y_cls_test, reg_models, clf_model)

    # 4. Visualization
    generate_visualizations(data_split, model_results, cls_results, conf_mx, best_cls_name, reg_models, trajectories, last_example)

if __name__ == "__main__":
    main()
