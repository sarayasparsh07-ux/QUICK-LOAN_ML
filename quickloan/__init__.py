"""QuickLoan credit-risk intelligence package.

Modules
-------
config      Constants, paths, feature schema, colour tokens.
data        Loading, validation, data-quality audit, the common train/test split.
features    Leakage-free feature engineering transformer (used inside pipelines).
finance     The official money-saved metric plus breakdowns and baselines.
models      Pipeline + hyperparameter-grid definitions for the ten models.
evaluation  Classification metrics, curves, thresholds and segment diagnostics.
artifacts   The artifact contract: save / load / validate.
"""

__version__ = "1.0.0"
