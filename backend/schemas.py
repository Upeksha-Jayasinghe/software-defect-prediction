"""Request and response schemas for defect-risk predictions."""

from pydantic import BaseModel, ConfigDict, Field


class ModuleFeatures(BaseModel):
	"""Numeric features expected by the saved CM1 model pipeline."""

	model_config = ConfigDict(extra="forbid")

	loc: float = Field(allow_inf_nan=False)
	v_g: float = Field(alias="v(g)", allow_inf_nan=False)
	ev_g: float = Field(alias="ev(g)", allow_inf_nan=False)
	iv_g: float = Field(alias="iv(g)", allow_inf_nan=False)
	n: float = Field(allow_inf_nan=False)
	v: float = Field(allow_inf_nan=False)
	l: float = Field(allow_inf_nan=False)
	d: float = Field(allow_inf_nan=False)
	i: float = Field(allow_inf_nan=False)
	e: float = Field(allow_inf_nan=False)
	b: float = Field(allow_inf_nan=False)
	t: float = Field(allow_inf_nan=False)
	lOCode: float = Field(allow_inf_nan=False)
	lOComment: float = Field(allow_inf_nan=False)
	lOBlank: float = Field(allow_inf_nan=False)
	locCodeAndComment: float = Field(allow_inf_nan=False)
	uniq_Op: float = Field(allow_inf_nan=False)
	uniq_Opnd: float = Field(allow_inf_nan=False)
	total_Op: float = Field(allow_inf_nan=False)
	total_Opnd: float = Field(allow_inf_nan=False)
	branchCount: float = Field(allow_inf_nan=False)


class PredictionResponse(BaseModel):
	prediction: bool
	defect_probability: float | None
	risk_level: str
