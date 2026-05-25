import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/titanic.csv")

X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Columns
num_cols = ["Age", "Fare"]
cat_cols = ["Pclass", "Sex"]

# 🔥 FULL preprocessing inside pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

mlflow.set_experiment("Titanic-Auto-Tuning")

best_acc = 0
best_run_id = None

# 🔥 MULTIPLE EXPERIMENTS
for n in [10, 50, 100, 200]:

    with mlflow.start_run() as run:

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=n, random_state=42))
        ])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log
        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"n={n}, acc={acc}")

        # Track best
        if acc > best_acc:
            best_acc = acc
            best_run_id = run.info.run_id

print("BEST RUN:", best_run_id)

from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "TitanicBestModel"

# Create model if not exists
try:
    client.create_registered_model(model_name)
except:
    pass

# Register best run model
model_uri = f"runs:/{best_run_id}/model"

client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=best_run_id
)

print("✅ Model registered successfully")

mlflow.sklearn.log_model(
    pipeline,
    "model",
    registered_model_name="TitanicBestModel"   # 🔥 THIS LINE IS KEY
)