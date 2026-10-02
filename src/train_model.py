"""Train and save a software defect classification pipeline."""

from pathlib import Path
import re

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cm1.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "defect_model.pkl"
TARGET_COLUMN = "defects"
RANDOM_STATE = 42


def load_dataset(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
	"""Load labeled data, remove exact duplicates, and exclude identifier columns."""
	data = pd.read_csv(data_path)
	if TARGET_COLUMN not in data.columns:
		raise KeyError(f"Required target column '{TARGET_COLUMN}' was not found.")

	original_rows = len(data)
	data = data.drop_duplicates().copy()
	duplicate_rows = original_rows - len(data)
	data = data.dropna(subset=[TARGET_COLUMN])

	identifier_columns = [
		column
		for column in data.columns
		if column != TARGET_COLUMN
		and (
			re.fullmatch(r"id", str(column).strip(), flags=re.IGNORECASE)
			or re.search(r"(?:^|[_\s])(?:id|identifier)(?:$|[_\s])", str(column), flags=re.IGNORECASE)
			or str(column).strip().lower().startswith("unnamed:")
		)
	]
	features = data.drop(columns=[TARGET_COLUMN, *identifier_columns])
	target = data[TARGET_COLUMN]
	return features, target, {
		"identifier_columns": identifier_columns,
		"duplicate_rows_removed": duplicate_rows,
	}


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
	"""Build train-fitted imputing, scaling, and categorical encoding steps."""
	numeric_columns = features.select_dtypes(include="number").columns.tolist()
	categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", OneHotEncoder(handle_unknown="ignore")),
		]
	)
	return ColumnTransformer(
		transformers=[
			("numeric", numeric_pipeline, numeric_columns),
			("categorical", categorical_pipeline, categorical_columns),
		],
		remainder="drop",
	)


def build_model_pipelines(features: pd.DataFrame, training_target: pd.Series) -> dict[str, Pipeline]:
	"""Return the four requested models, each with its own preprocessing pipeline."""
	negative_count = int((training_target == 0).sum())
	positive_count = int((training_target == 1).sum())
	scale_pos_weight = negative_count / positive_count if positive_count else 1.0

	models = {
		"Logistic Regression": LogisticRegression(
			class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
		),
		"Decision Tree": DecisionTreeClassifier(
			class_weight="balanced", random_state=RANDOM_STATE
		),
		"Random Forest": RandomForestClassifier(
			n_estimators=300,
			class_weight="balanced_subsample",
			random_state=RANDOM_STATE,
			n_jobs=1,
		),
		"XGBoost": XGBClassifier(
			n_estimators=300,
			max_depth=3,
			learning_rate=0.05,
			subsample=0.9,
			colsample_bytree=0.9,
			scale_pos_weight=scale_pos_weight,
			eval_metric="logloss",
			random_state=RANDOM_STATE,
			n_jobs=1,
		),
	}
	return {
		name: Pipeline(
			steps=[("preprocessing", build_preprocessor(features)), ("model", estimator)]
		)
		for name, estimator in models.items()
	}


def select_model_by_training_f1(
	features: pd.DataFrame, target: pd.Series
) -> tuple[str, Pipeline, pd.DataFrame]:
	"""Select a model using stratified cross-validation, never using test data."""
	candidates = build_model_pipelines(features, target)
	folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
	rows = []
	for name, pipeline in candidates.items():
		scores = cross_val_score(
			pipeline, features, target, cv=folds, scoring="f1", n_jobs=1
		)
		rows.append({"Model": name, "Training CV F1": scores.mean(), "CV F1 Std": scores.std()})

	results = pd.DataFrame(rows).sort_values("Training CV F1", ascending=False).reset_index(drop=True)
	selected_name = results.loc[0, "Model"]
	return selected_name, candidates[selected_name], results


def main() -> None:
	if not DATA_PATH.is_file():
		raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

	features, target, data_summary = load_dataset()
	if target.nunique() != 2:
		raise ValueError("This training script requires exactly two classes in the defects target.")

	x_train, _, y_train, _ = train_test_split(
		features,
		target,
		test_size=0.2,
		stratify=target,
		random_state=RANDOM_STATE,
	)
	selected_name, selected_pipeline, comparison = select_model_by_training_f1(x_train, y_train)

	# Keep the stratified holdout out of both preprocessing fitting and training.
	selected_pipeline.fit(x_train, y_train)
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(selected_pipeline, MODEL_PATH)

	print(f"Exact duplicate rows removed: {data_summary['duplicate_rows_removed']}")
	identifier_columns = data_summary["identifier_columns"]
	if identifier_columns:
		print(f"Identifier columns removed: {', '.join(identifier_columns)}")
	print("Training-only cross-validation F1 scores:")
	print(comparison.to_string(index=False))
	print(f"Selected model: {selected_name}")
	print(f"Saved complete preprocessing-and-model pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
	main()
