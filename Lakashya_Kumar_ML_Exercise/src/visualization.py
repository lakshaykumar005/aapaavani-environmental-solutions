import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import IMAGES_DIR, VAR_MAP, FEATURE_COLS, TARGET_COLS_REG

def save_plot(filename):
    save_path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(save_path)
    print(f"  > Saved {filename}")

def generate_visualizations(data_split, model_results, cls_results, conf_mx, best_cls_name, reg_models, trajectories, last_example_vals):
    print("\n[PART 4] Generating Outstanding Visualizations...")
    sns.set(style="whitegrid")
    
    X_train, X_test, y_reg_train, _, _, _ = data_split
    
    # 0. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    full_data = X_train.copy()
    for t in TARGET_COLS_REG:
        full_data[t] = y_reg_train[t]
    full_data.rename(columns=VAR_MAP, inplace=True)
    
    corr = full_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f')
    plt.title("Feature Connection Heatmap (Inputs vs Outputs)\n(Low Linear Correlation = High Non-Linearity)")
    plt.tight_layout()
    save_plot("correlation_heatmap.png")
    
    # Justification Print
    print("\n[JUSTIFICATION] Why Negative R2 only for K?")
    print("  1. K (Toxicity) is a 'Sweet Spot' problem (V-shaped).")
    print("  2. L/M are 'Volume' problems (Linear).")
    
    # 1. Model Comparison Plot
    if model_results:
        df_plot = pd.DataFrame(model_results)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.barplot(data=df_plot, x='Target', y='R2', hue='Model', palette='magma', ax=axes[0])
        axes[0].set_title("Model Accuracy (R2 Score)")
        axes[0].set_ylim(bottom=-1.0, top=1.05)
        
        sns.barplot(data=df_plot, x='Target', y='RMSE', hue='Model', palette='magma', ax=axes[1])
        axes[1].set_title("Prediction Error (RMSE)")
        
        plt.tight_layout()
        save_plot("model_comparison.png")

    # 1b. Classification Comparison
    if cls_results:
        df_cls = pd.DataFrame(cls_results)
        df_cls_melt = df_cls.melt(id_vars='Model', value_vars=['Accuracy', 'F1 Score'], var_name='Metric', value_name='Score')
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_cls_melt, x='Model', y='Score', hue='Metric', palette='viridis')
        plt.title(f"Classification Model Performance (Best: {best_cls_name})")
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right')
        plt.tight_layout()
        save_plot("classification_comparison.png")

    # 1c. Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mx, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Violation (0)', 'Compliant (1)'],
                yticklabels=['Violation (0)', 'Compliant (1)'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({best_cls_name})")
    plt.tight_layout()
    save_plot("confusion_matrix.png")

    # 2. Feature Importance for K
    if 'K' in reg_models and hasattr(reg_models['K'], 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        imps = reg_models['K'].feature_importances_
        idxs = np.argsort(imps)[::-1]
        sns.barplot(x=[imps[i] for i in idxs], y=[VAR_MAP[FEATURE_COLS[i]] for i in idxs], palette="viridis")
        plt.title(f"Key Drivers of Toxicity (K)")
        plt.tight_layout()
        save_plot("feature_importance_toxicity.png")

    # 3. Control Landscape
    plt.figure(figsize=(10, 8))
    f_range = np.linspace(1.0, 4.0, 50)
    g_range = np.linspace(30.0, 80.0, 50)
    F_grid, G_grid = np.meshgrid(f_range, g_range)
    
    base_input = X_train.mean().values
    Z = np.zeros_like(F_grid)
    for i in range(F_grid.shape[0]):
        for j in range(F_grid.shape[1]):
            input_row = base_input.copy()
            input_row[5] = F_grid[i, j]
            input_row[6] = G_grid[i, j]
            input_df_heat = pd.DataFrame([input_row], columns=FEATURE_COLS)
            Z[i, j] = reg_models['K'].predict(input_df_heat)[0]
            
    contour = plt.contourf(F_grid, G_grid, Z, levels=20, cmap="RdYlGn_r")
    plt.colorbar(contour, label="Predicted Toxicity (K)")
    plt.xlabel("Chemical Dosage (F)")
    plt.ylabel("Aeration Speed (G)")
    plt.title("Control Strategy Map: Optimization Trajectory")
    
    if trajectories:
        for traj in trajectories:
            plt.arrow(traj['start_F'], traj['start_G'], 
                      traj['end_F'] - traj['start_F'], traj['end_G'] - traj['start_G'], 
                      head_width=0.1, head_length=2, fc='blue', ec='blue', width=0.03)
            plt.scatter([traj['start_F']], [traj['start_G']], color='red', s=100)
            plt.scatter([traj['end_F']], [traj['end_G']], color='green', marker='*', s=200)
            
    plt.tight_layout()
    save_plot("control_landscape_heatmap.png")

    # 4. Optimization Impact
    if last_example_vals:
        labels = ['Toxicity (K)', 'Turbidity (L)', 'Op Cost (M)']
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, last_example_vals['before'], width, label='Before (Violation)', color='#e74c3c')
        plt.bar(x + width/2, last_example_vals['after'], width, label='After (Compliant)', color='#2ecc71')
        plt.xticks(x, labels)
        plt.legend()
        plt.tight_layout()
        save_plot("optimization_impact.png")
