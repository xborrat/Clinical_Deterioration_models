import json
import os
import numpy as np

def export_xgboost_json(model, features_df, threshold, output_dir, model_name='xgboost_xb'):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    # 1. Save Model JSON (Native XGBoost)
    model.save_model(os.path.join(output_dir, f"{model_name}.json"))
    # 2. Save Metadata JSON
    metadata = {
        "threshold": float(threshold),
        "feature_names": features_df.columns.tolist()
    }
    with open(os.path.join(output_dir, f"{model_name}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model saved to {output_dir}")

def export_kprototypes_json(model, features_df, output_dir, model_name='kproto_model'):
    """
    Exports K-Prototypes centroids/params to JSON.
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # Check if centroids are numpy array, then convert
    centroids = model.cluster_centroids_
    if hasattr(centroids, 'tolist'):
        centroids = centroids.astype(float).tolist()
    else:
        # Fallback if it's already a list of lists but contains numpy types
        centroids = [[float(x) for x in c] for c in centroids]

    model_data = {
        "n_clusters": int(model.n_clusters),
        "gamma": float(model.gamma) if hasattr(model, 'gamma') and model.gamma else None,
        "max_iter": int(model.max_iter),
        "cost": float(model.cost_),
        "n_iter": int(model.n_iter_),
        "cluster_centroids": centroids
    }
    
    with open(os.path.join(output_dir, f"{model_name}.json"), 'w') as f:
        json.dump(model_data, f, indent=2)

    metadata = {
        "feature_names": features_df.columns.tolist(),
        "info": "Centroids in model.json map to feature_names order."
    }
    with open(os.path.join(output_dir, f"{model_name}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Model and Metadata saved to: {output_dir}")