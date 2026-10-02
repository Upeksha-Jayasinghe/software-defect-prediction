"""FastAPI application for software defect risk predictions."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import ModuleFeatures, PredictionResponse


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "defect_model.pkl"

if not MODEL_PATH.is_file():
	raise FileNotFoundError(
		f"Trained model not found at {MODEL_PATH}. Run the model training script first."
	)

model = joblib.load(MODEL_PATH)
expected_features = list(model.feature_names_in_)

app = FastAPI(
	title="Software Defect Prediction API",
	description="Predict defect risk from software module metrics.",
	version="1.0.0",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
	allow_credentials=False,
	allow_methods=["GET", "POST"],
	allow_headers=["Content-Type"],
)


def is_defective_class(class_value: object) -> bool:
	"""Recognize the defective positive class without relying on class order."""
	normalized = str(class_value).strip().lower()
	return class_value is True or normalized in {"true", "1", "yes", "defective"}


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ModuleFeatures) -> PredictionResponse:
	try:
		feature_values = features.model_dump(by_alias=True)
		missing_features = set(expected_features) - set(feature_values)
		if missing_features:
			raise HTTPException(
				status_code=500,
				detail=f"API schema does not include model features: {sorted(missing_features)}",
			)

		feature_frame = pd.DataFrame(
			[[feature_values[name] for name in expected_features]],
			columns=expected_features,
		)
		predicted_class = model.predict(feature_frame)[0]
		prediction = is_defective_class(predicted_class)

		defect_probability = None
		if hasattr(model, "predict_proba"):
			probabilities = model.predict_proba(feature_frame)[0]
			for index, class_value in enumerate(model.classes_):
				if is_defective_class(class_value):
					defect_probability = float(probabilities[index])
					break

		if defect_probability is None:
			risk_level = "Unavailable"
		elif defect_probability < 0.30:
			risk_level = "Low"
		elif defect_probability <= 0.60:
			risk_level = "Medium"
		else:
			risk_level = "High"

		return PredictionResponse(
			prediction=prediction,
			defect_probability=defect_probability,
			risk_level=risk_level,
		)
	except HTTPException:
		raise
	except Exception as error:
		logger.exception("Prediction failed")
		raise HTTPException(
			status_code=500,
			detail="Prediction failed. Verify the supplied software metrics and try again.",
		) from error
