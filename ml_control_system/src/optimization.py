import numpy as np
import pandas as pd
from config import FEATURE_COLS, TARGET_COLS_REG, VAR_MAP

def predict_system_state(inputs_dict, reg_models, clf_model):
    """Helper to predict K, L, M, N for a given input state using trained models."""
    input_df = pd.DataFrame([inputs_dict])
    input_df = input_df[FEATURE_COLS]
    
    preds = {}
    for t in TARGET_COLS_REG:
        preds[t] = reg_models[t].predict(input_df)[0]
    
    preds['N_prob'] = clf_model.predict_proba(input_df)[0][1]
    preds['N'] = int(preds['N_prob'] > 0.5)
    return preds

def optimize_controls(current_state, reg_models, clf_model):
    """Grid Search Algorithm to find best F, G, H."""
    fixed_inputs = {k: current_state[k] for k in ['A', 'B', 'C', 'D', 'E', 'I', 'J']}
    
    f_steps = np.linspace(1.0, 4.0, 5) 
    g_steps = np.linspace(30.0, 80.0, 5) 
    h_steps = np.linspace(0.5, 2.0, 5)   
    
    best_score = float('inf')
    best_controls = None
    best_predictions = None
    
    for f in f_steps:
        for g in g_steps:
            for h in h_steps:
                candidate = fixed_inputs.copy()
                candidate.update({'F': f, 'G': g, 'H': h})
                
                preds = predict_system_state(candidate, reg_models, clf_model)
                
                cost = preds['K'] + preds['L'] + preds['M']
                if preds['N'] == 0:
                    cost += 5000 
                    
                if cost < best_score:
                    best_score = cost
                    best_controls = {'F': f, 'G': g, 'H': h}
                    best_predictions = preds
                    
    return best_controls, best_predictions

def demonstrate_optimization(X_test, y_reg_test, y_cls_test, reg_models, clf_model):
    """Finds bad examples and demonstrates the fix."""
    print("\n[PART 3] SIMPLE CONTROL ADJUSTMENT LOGIC")
    bad_indices = y_cls_test[y_cls_test == 0].index
    trajectories = []
    
    if len(bad_indices) > 0:
        num_examples = min(3, len(bad_indices))
        print(f"\nFOUND {len(bad_indices)} 'BAD' EXAMPLES. DEMONSTRATING FIX FOR {num_examples}:")
        
        total_hourly_savings = 0
        
        for i in range(num_examples):
            example_idx = bad_indices[i] 
            current_row = X_test.loc[example_idx]
            current_targets = {
                'K': y_reg_test.loc[example_idx, 'K'],
                'L': y_reg_test.loc[example_idx, 'L'],
                'M': y_reg_test.loc[example_idx, 'M'],
                'N': y_cls_test.loc[example_idx]
            }
            
            print(f"\nExample #{i+1} (Row {example_idx})")
            print("  [BEFORE] Unacceptable State:")
            print(f"    Controls: {VAR_MAP['F']}={current_row['F']:.2f}, {VAR_MAP['G']}={current_row['G']:.1f}, {VAR_MAP['H']}={current_row['H']:.2f}")
            print(f"    Status:   {VAR_MAP['N']}={current_targets['N']} (VIOLATION)")
            
            input_state = current_row.to_dict()
            best_ctrl, best_pred = optimize_controls(input_state, reg_models, clf_model)
            
            trajectories.append({
                'start_F': current_row['F'], 'start_G': current_row['G'],
                'end_F': best_ctrl['F'], 'end_G': best_ctrl['G']
            })
            
            savings_hr = current_targets['M'] - best_pred['M']
            if savings_hr > 0: total_hourly_savings += savings_hr
            
            print("  [AFTER] Optimized Controls:")
            print(f"    Controls: {VAR_MAP['F']}={best_ctrl['F']:.2f}, {VAR_MAP['G']}={best_ctrl['G']:.1f}, {VAR_MAP['H']}={best_ctrl['H']:.2f}")
            print(f"    Status:   {VAR_MAP['N']}={best_pred['N']} (COMPLIANT)")
            print(f"    Hourly Savings: ${savings_hr:.2f}")
            print("-" * 40)
            
        avg_savings = total_hourly_savings / num_examples
        annual_savings = avg_savings * 24 * 365
        print(f"\n[BUSINESS VALUE] Projected Annual Savings: ${annual_savings:,.2f}")
        
        # Determine last example for plotting
        last_example = {
            'before': [current_targets['K'], current_targets['L'], current_targets['M']],
            'after': [best_pred['K'], best_pred['L'], best_pred['M']]
        }
        return trajectories, last_example
        
    else:
        print("No 'Bad' examples found in test set to optimize.")
        return [], None
