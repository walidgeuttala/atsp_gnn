"""
Refactored hyperparameter search for ATSP (Optuna + memory-safe size testing).

Key behavior:
 - Grid over relation subsets and agg choices ("sum"/"concat").
 - For each combination, run an Optuna study (objective uses average validation loss on atsp_size=50).
 - Test best configs on size 500 with memory clearing between tests
 - Only save models that work on target sizes without OOM
 - Integrate size testing during the search process
 - Log all details including OOM failures
"""
import itertools
import importlib
import logging
import math
import os
import sys
import traceback
import gc
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import optuna
import torch


# ensure relative imports work when executed as a script
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from .args import parse_args
from ..utils import fix_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("search")

def get_slurm_job_id() -> str:
    """
    Return the SLURM job ID as a string if available, otherwise 'no_slurm_job'.
    """
    return os.environ.get("SLURM_JOB_ID", "no_slurm_job")

def get_output_base() -> str:
    """
    Create a base output directory inside ./search/<SLURM_JOB_ID>/.
    """
    slurm_id = get_slurm_job_id()
    base_dir = os.path.join("search", slurm_id)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def patch_tb_dir(args):
    """
    Force args.tb_dir to ./search/<SLURM_JOB_ID>
    """
    slurm_id = get_slurm_job_id()
    base = os.path.join("./search", slurm_id)
    os.makedirs(base, exist_ok=True)
    args.tb_dir = base
    return args

class OOMError(Exception):
    pass

def clear_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def locate(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Cannot import {module_name}: {e}")

def all_nonempty_subsets(seq: List[str]) -> Iterable[Tuple[str, ...]]:
    for r in range(1, len(seq)+1):
        for comb in itertools.combinations(seq, r):
            yield tuple(sorted(comb))

def safe_step(fn, *args, **kwargs):
    """
    Wrapper to call training/testing functions and convert CUDA OOM to OOMError for upstream handling.
    """
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if 'out of memory' in msg or isinstance(e, torch.cuda.OutOfMemoryError):
            logger.warning("Caught CUDA OOM: %s", e)
            clear_memory()
            raise OOMError(str(e))
        raise

def test_single_instance_on_size(args: Any, config: Dict[str, Any], size: int, framework: str) -> bool:
    """
    Test if a configuration works on a single instance of given size without OOM.
    Returns True if successful, False if OOM or other error.
    """
    
    test_args = deepcopy(args)
    test_args.relation_types = list(config["relations"])
    test_args.agg = config["agg"]
    # set hyperparams
    for k, v in config["best_params"].items():
        setattr(test_args, k, v)
    test_args.atsp_size = size
    fix_seed(getattr(test_args, "seed", 0))

    try:
        # Clear memory before test
        clear_memory()
        
        # Import required modules
        trainer_mod = locate(f"src.engine.train_{framework}")
        TrainerClass = getattr(trainer_mod, "ATSPTrainerDGL", None) or getattr(trainer_mod, "ATSPTrainerPyG", None)
        if TrainerClass is None:
            raise ImportError(f"Trainer class not found")

        models_mod = locate(f"src.models.models_{framework}")
        get_model = getattr(models_mod, f"get_{framework}_model", None)
        if get_model is None:
            raise ImportError("Model factory not found")

        # Create tester instead of trainer for single instance test
        tester_mod = locate(f"src.engine.test_{framework}")
        TesterClass = getattr(tester_mod, "ATSPTesterDGL", None) or getattr(tester_mod, "ATSPTesterPyG", None)
        if TesterClass is None:
            raise ImportError(f"Tester class not found")
        
        # Create model and tester
        model = get_model(test_args)
        tester = TesterClass(test_args)
                
        # Create test dataset
        test_dataset = tester.create_test_dataset()
        
        
        # Test only first instance to check for OOM
        if len(test_dataset) > 0:
            result = safe_step(tester.test_instance, model, test_dataset, 0)
            logger.info(f"Size {size} test successful - Gap: {result.get('final_gap', 'N/A'):.2f}%")
            
        # Clean up
        del model, tester, test_dataset
        clear_memory()

        return True
        
    except OOMError:
        logger.info(f"Config failed on size {size} due to OOM")
        clear_memory()
        return False
    except Exception as e:
        logger.warning(f"Config failed on size {size} due to error: {str(e)[:100]}...")
        clear_memory()
        return False

def objective_factory(args: Any, relation_subset: Tuple[str, ...], agg_choice: str, framework: str, n_epochs_optuna: int):
    """
    Optuna objective -- sample hyperparams, train on atsp_size=50, return validation loss.
    """
    def objective(trial: optuna.trial.Trial) -> float:
        clear_memory()  # Clear memory before each trial
        
        # sample hyperparams
        hidden_dim = int(trial.suggest_categorical("hidden_dim", [1,2,4,8,16,32]))
        valid_heads = [d for d in [1,2,4,8,16] if hidden_dim % d == 0 and d <= hidden_dim]
        num_heads = int(trial.suggest_categorical("num_heads", valid_heads))
        num_gnn_layers = int(trial.suggest_categorical("num_gnn_layers", [1,2,3]))
        lr_init = float(trial.suggest_float("lr_init", 1e-5, 1e-2, log=True))


        run_args = deepcopy(args)
        run_args.relation_types = list(relation_subset)
        run_args.agg = agg_choice
        run_args.hidden_dim = hidden_dim
        run_args.num_heads = num_heads
        run_args.num_gnn_layers = num_gnn_layers
        run_args.lr_init = lr_init
        run_args.atsp_size = 50
        fix_seed(getattr(run_args, "seed", 0))

        trainer_module = f"src.engine.train_{framework}"
        try:
            trainer_mod = locate(trainer_module)
            TrainerClass = getattr(trainer_mod, "ATSPTrainerDGL", None) or getattr(trainer_mod, "ATSPTrainerPyG", None)
            if TrainerClass is None:
                raise ImportError(f"Trainer class not found in {trainer_module}")

            models_mod = locate(f"src.models.models_{framework}")
            get_model = getattr(models_mod, f"get_{framework}_model", None)
            if get_model is None:
                raise ImportError("Model factory not found")

            model = get_model(run_args)
            trainer = TrainerClass(run_args)

            try:
                results = safe_step(trainer.train, model, trial_id=trial.number)
            except OOMError:
                logger.warning("OOM during training; pruning this Optuna trial.")
                raise optuna.exceptions.TrialPruned()

            avg_val = results['best_val_loss']
            trial.report(avg_val, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Clean up after successful training
            del model, trainer
            clear_memory()
            
            return avg_val

        except OOMError:
            raise optuna.exceptions.TrialPruned()
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.exception("Unexpected error in objective: %s", traceback.format_exc())
            clear_memory()
            return float('inf')

    return objective

def build_save_folder(base_dir: str, tag: str, rels: Tuple[str, ...], agg: str, params: Dict[str, Any], size_info: str) -> str:
    """Create informative folder name."""
    rel_part = "-".join(rels) if rels else "none"
    params_part = []
    for k in ["hidden_dim", "num_heads", "lr_init"]:
        if k in params:
            v = params[k]
            if isinstance(v, float):
                params_part.append(f"{k}{v:.0e}")
            else:
                params_part.append(f"{k}{v}")
    params_str = "_".join(params_part) if params_part else "default"
    foldername = f"best_{size_info}_agg{agg}_rels{rel_part}_{tag}_{params_str}"
    full = os.path.join(base_dir, foldername)
    os.makedirs(full, exist_ok=True)
    return full

def save_model_checkpoint(trainer, model, final_args: Any, save_folder: str, filename: str):
    """Save model checkpoint with error handling."""
    ckpt_path = os.path.join(save_folder, filename)
    
    try:
        # Try to locate trainer's checkpoint if it saved one
        candidate_paths = []
        if hasattr(trainer, "log_dir"):
            for root, dirs, files in os.walk(trainer.log_dir):
                for fname in files:
                    if fname.endswith(".pt"):
                        candidate_paths.append(os.path.join(root, fname))
        
        best_src = None
        if candidate_paths:
            best_candidates = [p for p in candidate_paths if 'best' in os.path.basename(p).lower()]
            best_src = best_candidates[0] if best_candidates else candidate_paths[0]
        
        if best_src:
            logger.info("Copying trainer checkpoint %s to %s", best_src, ckpt_path)
            import shutil
            shutil.copy2(best_src, ckpt_path)
        else:
            logger.info("Saving model state_dict to %s", ckpt_path)
            state = {'model_state_dict': model.state_dict(), 'args': vars(final_args)}
            torch.save(state, ckpt_path)
        
        return ckpt_path
    except Exception:
        logger.exception("Failed to save checkpoint; falling back to saving state_dict.")
        state = {'model_state_dict': model.state_dict(), 'args': vars(final_args)}
        torch.save(state, ckpt_path)
        return ckpt_path

def train_and_save_model(args: Any, config: Dict[str, Any], size: int, framework: str, save_folder: str, filename: str) -> bool:
    """Train and save a model for specific size."""
    clear_memory()
    
    final_args = deepcopy(args)
    final_args.relation_types = list(config["relations"])
    final_args.agg = config["agg"]
    for k, v in config["best_params"].items():
        setattr(final_args, k, v)
    final_args.atsp_size = size
    fix_seed(getattr(final_args, "seed", 0))

    try:
        trainer_mod = locate(f"src.engine.train_{framework}")
        TrainerClass = getattr(trainer_mod, "ATSPTrainerDGL", None) or getattr(trainer_mod, "ATSPTrainerPyG", None)
        models_mod = locate(f"src.models.models_{framework}")
        get_model = getattr(models_mod, f"get_{framework}_model")

        model = get_model(final_args)
        trainer = TrainerClass(final_args)
        results = safe_step(trainer.train, model, 0)

        save_model_checkpoint(trainer, model, final_args, save_folder, filename)
        logger.info(f"Successfully saved model for size {size}: {filename}")
        
        # Clean up
        del model, trainer
        clear_memory()
        return True

    except Exception as e:
        logger.exception(f"Failed to train/save size {size} model: %s", e)
        clear_memory()
        return False

def run_search(args: Any, n_optuna_trials: int = 20, n_jobs: int = 1, n_epochs_optuna: int = 2):
    args = patch_tb_dir(args)
    framework = args.framework.lower()
    relation_space = list(all_nonempty_subsets(list(args.relation_types)))
    agg_space = ["sum", "concat"]

    logger.info("Starting grid over %d relation subsets x %d aggs = %d combinations",
                len(relation_space), len(agg_space), len(relation_space) * len(agg_space))

    # Store results for different size categories
    all_results = {
        "size_50": [],      # All configs that work on size 50
        "size_500": [],     # Configs that work on both 50 and 500
    }
    
    search_summary = []
    out_base = getattr(args, "tb_dir", get_output_base())


    # Phase 1: Optuna search for each relation/agg combination
    for rel_idx, rel_subset in enumerate(relation_space):
        for agg_idx, agg_choice in enumerate(agg_space):
            combo_name = f"rel{rel_idx+1}_agg_{agg_choice}"
            logger.info("=== Searching %s: relations=%s agg=%s ===", combo_name, rel_subset, agg_choice)
            
            clear_memory()  # Clear before each combination
            
            study = optuna.create_study(direction="minimize")
            objective = objective_factory(args, rel_subset, agg_choice, framework, n_epochs_optuna)

            try:
                study.optimize(objective, n_trials=n_optuna_trials, n_jobs=n_jobs)
            except Exception as e:
                logger.warning("Optuna run failed for %s: %s", combo_name, e)
                continue

            # Get best trial for this combination
            if not study.trials:
                logger.warning("No trials completed for %s", combo_name)
                continue
                
            successful_trials = [t for t in study.trials 
                               if t.state == optuna.trial.TrialState.COMPLETE and t.value != float('inf')]
            
            if not successful_trials:
                logger.warning("No successful trials for %s", combo_name)
                continue
                
            best_trial = min(successful_trials, key=lambda t: t.value)
            
            config = {
                "relations": rel_subset,
                "agg": agg_choice,
                "best_params": best_trial.params,
                "best_value": best_trial.value,
                "combo_name": combo_name
            }
            
            logger.info("Best for %s: val_loss=%.6f params=%s", combo_name, best_trial.value, best_trial.params)
            
            # Add to size_50 results
            all_results["size_50"].append(config.copy())
            
            # Test on size 500
            logger.info("Testing %s on size 500...", combo_name)
            works_500 = test_single_instance_on_size(args, config, 500, framework)
            config["works_on_500"] = works_500
            
            if works_500:
                logger.info("%s works on size 500!", combo_name)
                all_results["size_500"].append(config.copy())
            else:
                logger.info("%s failed on size 500", combo_name)
            
            # Add to search summary
            search_summary.append(config)
            
            logger.info("=== Completed %s ===\n", combo_name)

    # Phase 2: Save best models for each size category
    best_models = {}
    
    # Best for size 50
    if all_results["size_50"]:
        all_results["size_50"].sort(key=lambda x: x["best_value"])
        best_50 = all_results["size_50"][0]
        best_models["size_50"] = best_50
        logger.info("=== BEST SIZE 50 MODEL ===")
        logger.info("Config: %s, Val Loss: %.6f", best_50["combo_name"], best_50["best_value"])
        
        save_folder_50 = build_save_folder(out_base, "model", best_50["relations"], 
                                         best_50["agg"], best_50["best_params"], "size50")
        train_and_save_model(args, best_50, 50, framework, save_folder_50, "best_model_atsp50.pt")

    # Best for size 500 (among those that work on both 50 and 500)
    if all_results["size_500"]:
        all_results["size_500"].sort(key=lambda x: x["best_value"])
        best_500 = all_results["size_500"][0]
        best_models["size_500"] = best_500
        logger.info("=== BEST SIZE 500 MODEL (works on 50+500) ===")
        logger.info("Config: %s, Val Loss: %.6f", best_500["combo_name"], best_500["best_value"])
        
        save_folder_500 = build_save_folder(out_base, "model", best_500["relations"],
                                           best_500["agg"], best_500["best_params"], "size50_500")
        train_and_save_model(args, best_500, 500, framework, save_folder_500, "best_model_atsp500.pt")

    # Phase 3: Save all working models for size 500 (if possible)
    logger.info("=== SAVING ADDITIONAL WORKING MODELS ===")
    
    # Save top working models for size 500 (up to 5)
    for i, config in enumerate(all_results["size_500"][:5]):
        if i == 0:  # Skip the best one (already saved)
            continue
        save_folder = build_save_folder(out_base, f"model_rank{i+1}", config["relations"],
                                       config["agg"], config["best_params"], "size500_working")
        success = train_and_save_model(args, config, 500, framework, save_folder, f"model_atsp500_rank{i+1}.pt")
        if success:
            logger.info("Saved rank %d model for size 500: %s", i+1, config["combo_name"])

    # Phase 4: Write detailed summary
    summary_file = os.path.join(out_base, "hyperparam_search_detailed_summary.yaml")
    try:
        import yaml
        summary_data = {
            "search_results": search_summary,
            "best_models": best_models,
            "statistics": {
                "total_combinations_tested": len(search_summary),
                "combinations_work_size_50": len(all_results["size_50"]),
                "combinations_work_size_500": len(all_results["size_500"]),
            },
            "working_configs_by_size": {
                "size_50": all_results["size_50"],
                "size_500": all_results["size_500"], 
            }
        }
        with open(summary_file, "w") as f:
            yaml.safe_dump(summary_data, f, default_flow_style=False)
        logger.info("Wrote detailed search summary to %s", summary_file)
    except Exception:
        logger.exception("Failed to write summary file")

    # Final report
    logger.info("\n" + "="*80)
    logger.info("SEARCH COMPLETED!")
    logger.info("SLURM Job ID: %s", get_slurm_job_id())
    logger.info("Output Directory: %s", out_base)
    logger.info("Total combinations tested: %d", len(search_summary))
    logger.info("Working on size 50: %d", len(all_results["size_50"]))
    logger.info("Working on size 500: %d", len(all_results["size_500"]))  
    logger.info("="*80)

    return search_summary, best_models

if __name__ == "__main__":
    parser = parse_args()
    N_OPTUNA_TRIALS = getattr(parser, "n_trials", 2) or 2
    N_JOBS = 1

    try:
        results, best = run_search(parser, n_optuna_trials=N_OPTUNA_TRIALS, n_jobs=N_JOBS)
        logger.info("Search completed. Best configs: %s", best)
    except Exception:
        logger.exception("Search failed: %s", traceback.format_exc())