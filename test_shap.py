import shap.explainers._tree
import xgboost as xgb
import numpy as np
import json

original_decode = shap.explainers._tree.decode_ubjson_buffer
def patched_decode(*args, **kwargs):
    jmodel = original_decode(*args, **kwargs)
    print("Patched called")
    params = jmodel["learner"]["learner_model_param"]
    base_score = params.get("base_score")
    print(f"Old base_score: {base_score}")
    if isinstance(base_score, str) and base_score.startswith("["):
        params["base_score"] = str(json.loads(base_score)[0])
        print(f"New base_score: {params['base_score']}")
    return jmodel

shap.explainers._tree.decode_ubjson_buffer = patched_decode

model=xgb.XGBRegressor().fit(np.array([[1]]), np.array([1]))
loader = shap.explainers._tree.XGBTreeModelLoader(model.get_booster())
