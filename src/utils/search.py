import optuna
from optuna.samplers import TPESampler
import json
from typing import Dict, Any, Callable

def create_study(study_name: str, storage: str = None, seed: int = 42):
    """Create Optuna study for hyperparameter optimization."""
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        sampler=TPESampler(seed=seed),
        load_if_exists=True
    )

def default_search_space(trial, args) -> Dict[str, Any]:
    """Default hyperparameter search space for ATSP models."""
    search_params = {
        'lr_init': trial.suggest_float('lr_init', 1e-4, 1e-2, log=True),
        'lr_decay': trial.suggest_float('lr_decay', 0.95, 0.999),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
    }
    
    # Model-specific hyperparameters
    if hasattr(args, 'hidden_dim'):
        search_params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    if hasattr(args, 'num_layers'):
        search_params['num_layers'] = trial.suggest_int('num_layers', 2, 6)
    if hasattr(args, 'dropout'):
        search_params['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Relation types
    all_relations = ['ss', 'st', 'tt', 'pp']
    search_params['relation_types'] = trial.suggest_categorical('relation_types', [
        ['ss'], ['tt'], ['pp'], ['st'],  # single
        ['ss', 'tt'], ['ss', 'pp'], ['st', 'tt'],  # pairs
        ['ss', 'st', 'tt'], ['ss', 'tt', 'pp'],  # triplets
        ['ss', 'st', 'tt', 'pp']  # all
    ])
    
    return search_params

def run_hyperopt(objective_fn: Callable, n_trials: int, study_name: str, 
                storage: str = None, seed: int = 42) -> optuna.study.Study:
    """Run hyperparameter optimization."""
    study = create_study(study_name, storage, seed)
    study.optimize(objective_fn, n_trials=n_trials)
    return study

def save_best_params(study, output_path: str):
    """Save best hyperparameters to JSON file."""
    best_params = study.best_trial.params
    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=2)
