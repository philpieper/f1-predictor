simple_top10_binary_classification:
	uv run python -m f1_predictor.simple_top10_binary_classification
last3_quali_binary_classification:
	uv run python -m f1_predictor.last3_quali_binary_classification
weighted_ensemble_binary_classification:
	uv run python -m f1_predictor.weighted_ensemble_binary_classification
trained_weights_weighted_ensemble_binary_classification:
	uv run python -m f1_predictor.trained_weights_weighted_ensemble_binary_classification
data_loader:
	uv run python -m f1_predictor.data_loader
features:
	uv run python -m f1_predictor.features
ranking_metrics:
	uv run python -m f1_predictor.ranking_metrics
compare_pipeline:
	uv run python -m f1_predictor.compare_pipeline
