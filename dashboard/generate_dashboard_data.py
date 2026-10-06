"""Generate Power BI CSVs from the CM1 dataset and saved notebook results."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "cm1.csv"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_model_evaluation.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "dashboard" / "data"

CM1_COLUMNS = [
	"loc",
	"v(g)",
	"ev(g)",
	"iv(g)",
	"n",
	"v",
	"l",
	"d",
	"i",
	"e",
	"b",
	"t",
	"lOCode",
	"lOComment",
	"lOBlank",
	"locCodeAndComment",
	"uniq_Op",
	"uniq_Opnd",
	"total_Op",
	"total_Opnd",
	"branchCount",
	"defects",
]
EXPECTED_MODELS = {
	"Logistic Regression",
	"Decision Tree",
	"Random Forest",
	"XGBoost",
}
MODEL_METRIC_COLUMNS = {
	"model": "Model",
	"accuracy": "Accuracy",
	"precision": "Precision",
	"recall": "Recall",
	"f1-score": "F1_Score",
	"roc-auc": "ROC_AUC",
}


class _TableParser(HTMLParser):
	"""Read simple notebook-rendered HTML tables without an HTML dependency."""

	def __init__(self) -> None:
		super().__init__()
		self.tables: list[list[list[tuple[str, str]]]] = []
		self._table_depth = 0
		self._table: list[list[tuple[str, str]]] = []
		self._row: list[tuple[str, str]] | None = None
		self._cell_tag: str | None = None
		self._cell_text: list[str] = []

	def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
		if tag == "table":
			if self._table_depth == 0:
				self._table = []
				self._table_depth = 1
			return
		if self._table_depth and tag == "tr":
			self._row = []
		elif self._row is not None and tag in {"th", "td"}:
			self._cell_tag = tag
			self._cell_text = []

	def handle_data(self, data: str) -> None:
		if self._cell_tag is not None:
			self._cell_text.append(data)

	def handle_endtag(self, tag: str) -> None:
		if self._cell_tag == tag and tag in {"th", "td"}:
			value = " ".join("".join(self._cell_text).split())
			if self._row is not None:
				self._row.append((tag, value))
			self._cell_tag = None
		elif tag == "tr" and self._row is not None:
			self._table.append(self._row)
			self._row = None
		elif tag == "table" and self._table_depth:
			self._table_depth = 0
			self.tables.append(self._table)


def _html_tables(html: str) -> list[list[list[tuple[str, str]]]]:
	parser = _TableParser()
	parser.feed(html)
	return parser.tables


def _mime_text(value: object) -> str:
	if isinstance(value, list):
		return "".join(str(part) for part in value)
	return str(value)


def _cell_values(row: list[tuple[str, str]]) -> list[str]:
	return [value for _, value in row]


def _find_header(
	table: list[list[tuple[str, str]]], required: set[str]
) -> tuple[int, dict[str, int]] | None:
	for row_index, row in enumerate(table):
		header = {value.strip().lower(): index for index, value in enumerate(_cell_values(row))}
		if required.issubset(header):
			return row_index, header
	return None


def _evaluation_tables(
	notebook: dict[str, object],
) -> tuple[
	list[list[list[tuple[str, str]]]],
	list[list[list[tuple[str, str]]]],
	list[list[list[tuple[str, str]]]],
]:
	metric_tables = []
	cv_tables = []
	importance_tables = []
	for cell in notebook.get("cells", []):
		source = "".join(cell.get("source", []))
		outputs = cell.get("outputs", [])
		for output in outputs:
			html = output.get("data", {}).get("text/html")
			if not html:
				continue
			for table in _html_tables(_mime_text(html)):
				if _find_header(table, set(MODEL_METRIC_COLUMNS)):
					metric_tables.append(table)
				elif _find_header(table, {"model", "mean cv f1"}):
					cv_tables.append(table)
				elif "feature_importances_" in source and _find_header(table, {"importance"}):
					importance_tables.append(table)
	return metric_tables, cv_tables, importance_tables


def _read_model_results(notebook_path: Path) -> tuple[pd.DataFrame, str]:
	if not notebook_path.is_file():
		raise FileNotFoundError(f"Evaluation notebook not found: {notebook_path}")
	with notebook_path.open(encoding="utf-8") as notebook_file:
		notebook = json.load(notebook_file)
	metric_tables, cv_tables, _ = _evaluation_tables(notebook)
	if len(metric_tables) != 1:
		raise ValueError("Expected one stored four-model evaluation table in the executed notebook.")

	table = metric_tables[0]
	header_index, header = _find_header(table, set(MODEL_METRIC_COLUMNS)) or (0, {})
	rows = []
	for row in table[header_index + 1 :]:
		values = _cell_values(row)
		if len(values) <= max(header.values()):
			continue
		model_name = values[header["model"]]
		if model_name not in EXPECTED_MODELS:
			continue
		result = {"Model": model_name}
		for source_column, output_column in MODEL_METRIC_COLUMNS.items():
			if source_column != "model":
				result[output_column] = float(values[header[source_column]])
		rows.append(result)

	comparison = pd.DataFrame(
		rows,
		columns=["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
	)
	if set(comparison["Model"]) != EXPECTED_MODELS or comparison["Model"].duplicated().any():
		raise ValueError("Stored notebook evaluation does not contain exactly one row for each expected model.")
	if comparison.isna().any().any():
		raise ValueError("Stored notebook evaluation contains missing metric values.")

	if len(cv_tables) != 1:
		raise ValueError("Expected one stored training cross-validation table to identify the selected model.")
	cv_table = cv_tables[0]
	cv_header_index, cv_header = _find_header(cv_table, {"model", "mean cv f1"}) or (0, {})
	cv_rows = []
	for row in cv_table[cv_header_index + 1 :]:
		values = _cell_values(row)
		if len(values) > max(cv_header.values()) and values[cv_header["model"]] in EXPECTED_MODELS:
			cv_rows.append((values[cv_header["model"]], float(values[cv_header["mean cv f1"]])))
	if {name for name, _ in cv_rows} != EXPECTED_MODELS:
		raise ValueError("Stored cross-validation table does not contain all four candidate models.")
	selected_model = max(cv_rows, key=lambda row: row[1])[0]
	comparison["Selected"] = comparison["Model"].map(
		lambda name: "Yes" if name == selected_model else "No"
	)
	return comparison, selected_model


def _read_feature_importance(notebook_path: Path) -> pd.DataFrame:
	with notebook_path.open(encoding="utf-8") as notebook_file:
		notebook = json.load(notebook_file)
	_, _, tables = _evaluation_tables(notebook)
	if len(tables) != 2:
		raise ValueError("Expected stored feature-importance tables for Random Forest and XGBoost.")

	rows = []
	for model_name, table in zip(("Random Forest", "XGBoost"), tables, strict=True):
		header_index, header = _find_header(table, {"importance"}) or (0, {})
		importance_column = header["importance"]
		for row in table[header_index + 1 :]:
			if not row:
				continue
			values = _cell_values(row)
			if len(values) <= importance_column:
				continue
			feature = values[0].removeprefix("numeric__")
			try:
				importance = float(values[importance_column])
			except ValueError:
				continue
			rows.append({"Feature": feature, "Importance": importance, "Model": model_name})

	result = pd.DataFrame(rows)
	if result.empty or result[["Feature", "Importance", "Model"]].isna().any().any():
		raise ValueError("Stored notebook feature-importance tables could not be read.")
	result = result.sort_values(["Model", "Importance", "Feature"], ascending=[True, False, True])
	result["Rank"] = result.groupby("Model").cumcount() + 1
	return result[["Feature", "Importance", "Model", "Rank"]].reset_index(drop=True)


def _prepare_dashboard_dataset(source_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
	if not source_path.is_file():
		raise FileNotFoundError(f"CM1 dataset not found: {source_path}")
	data = pd.read_csv(source_path)
	missing_columns = [column for column in CM1_COLUMNS if column not in data.columns]
	if missing_columns:
		raise ValueError(f"CM1 dataset is missing required columns: {missing_columns}")
	data = data[CM1_COLUMNS].copy()
	target = data["defects"].astype("string").str.strip().str.lower()
	status_map = {
		"true": "Defective",
		"1": "Defective",
		"false": "Non-Defective",
		"0": "Non-Defective",
	}
	status = target.map(status_map)
	if status.isna().any():
		raise ValueError("The defects target contains values other than True/False or 1/0.")
	data["Defect_Status"] = status
	return data, pd.DataFrame(
		[
			{"Metric": "Total Modules", "Value": len(data)},
			{"Metric": "Defective Modules", "Value": int((status == "Defective").sum())},
			{"Metric": "Non-Defective Modules", "Value": int((status == "Non-Defective").sum())},
			{"Metric": "Defect Rate", "Value": float((status == "Defective").mean())},
			{"Metric": "Duplicate Rows Removed", "Value": int(data[CM1_COLUMNS].duplicated().sum())},
			{"Metric": "Number of Features", "Value": len(CM1_COLUMNS) - 1},
		]
	)


def main() -> None:
	dashboard_data, summary = _prepare_dashboard_dataset(SOURCE_PATH)
	model_comparison, selected_model = _read_model_results(NOTEBOOK_PATH)
	feature_importance = _read_feature_importance(NOTEBOOK_PATH)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	files = {
		"cm1_dashboard_data.csv": dashboard_data,
		"dataset_summary.csv": summary,
		"model_comparison.csv": model_comparison,
		"feature_importance.csv": feature_importance,
	}
	for filename, frame in files.items():
		frame.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8")
		print(f"Wrote {filename}: {len(frame)} rows")
	print(f"Selected model from stored training cross-validation: {selected_model}")


if __name__ == "__main__":
	main()