import numpy as np
import itertools

from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation



def randomized_search_cv(model,dataset: Dataset, hyperparameter_grid: dict, scoring, cv:int, n_iter: int) -> dict:
    for param in hyperparameter_grid:
        if not hasattr(model, param):
            raise ValueError(f"Invalid hyperparameter: {param}")


    # 2. Generate all combinations
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    all_combinations = list(itertools.product(*param_values))

    # Randomly select n_iter combinations
    n_iter = min(n_iter, len(all_combinations))
    indices = np.random.choice(len(all_combinations), size=n_iter, replace=False)
    selected_combinations = [all_combinations[i] for i in indices]

    results_scores = []
    results_hyperparameters = []

    # 3–6. Evaluate each combination
    for combination in selected_combinations:
        current_params = {}

        for name, value in zip(param_names, combination):
            setattr(model, name, value)
            current_params[name] = value

        scores = k_fold_cross_validation(
            model=model,
            dataset=dataset,
            scoring=scoring,
            cv=cv
        )

        mean_score = np.mean(scores)

        results_scores.append(mean_score)
        results_hyperparameters.append(current_params)

    # 7. Best result
    best_idx = np.argmax(results_scores)

    # 8. Output dictionary
    return {
        "hyperparameters": results_hyperparameters,
        "scores": results_scores,
        "best_hyperparameters": results_hyperparameters[best_idx],
        "best_score": results_scores[best_idx]
    }

