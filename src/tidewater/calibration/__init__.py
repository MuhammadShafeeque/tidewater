"""Calibration of the calving constant k against observed frontal ablation.

``targets`` (observation tables and units), ``static`` (k on the inverted
front, fill rules, ocean scaling), ``covariates`` (models of k),
``permutation`` (significance of a model search), ``dynamic`` (correction of k
through the host's dynamic model).
"""


def column_units(flux_unit):
    """Unit of every column of the calibration tables, for a flux unit."""
    per_k = f"{flux_unit} per yr-1"
    return {
        "glacier_id": "id", "calving_k": "yr-1", "calving_k_unc": "yr-1",
        "k_static": "yr-1", "k_prev": "yr-1", "k_ref": "yr-1",
        "flux": flux_unit, "flux_err": flux_unit, "q_target": flux_unit,
        "q_err": flux_unit, "q_dyn": flux_unit, "shape": per_k,
        "ratio": "1", "passes": "count", "converged": "boolean",
        "within_error": "boolean", "scaled": "boolean", "reason": "text",
        "rule": "text", "rules": "text", "source": "text", "error": "text",
        "history": f"pairs of yr-1 and {flux_unit}", "tf_mean": "degC",
        "tf_ref": "degC", "start_date": "as given", "end_date": "as given",
    }
