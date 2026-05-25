""""
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment = client.get_experiment_by_name("Titanic-Auto-Tuning")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

best_run_id = runs[0].info.run_id

print("Using BEST RUN:", best_run_id)

model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

"""

#---------------------- 

#using joblib

import joblib

model = joblib.load("model/model.pkl")