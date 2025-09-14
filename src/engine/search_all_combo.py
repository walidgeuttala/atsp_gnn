"""
Optuna hyperparameter search for ATSP with size 500 pre-screening.
Key features:
- For each Optuna trial: FIRST test config on size 500, then train on size 50 if no OOM
- If config fails size 500 test, return high loss immediately (no training)
- Save best working model during Optuna process (overwrites previous best)
- Memory management and OOM handling
"""
import itertools
import importlib
import logging
import os
import sys
import gc
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple
import optuna
import torch

# Setup paths and logging
# Ensure Python can import both 'src.engine.*' (for relative imports inside modules)
# and 'data.*'/'engine.*' (direct top-level convenience). Add project root and src/.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROJ_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
for _p in (_PROJ_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from .args import parse_args
from ..utils import fix_seed
import json
import multiprocessing as mp
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("search")

class OOMError(Exception):
    pass

def clear_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_output_dir() -> str:
    """Create output directory for search results."""
    slurm_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    base_dir = os.path.join("search", slurm_id)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def locate_module(module_name: str):
    """Import module with error handling."""
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Cannot import {module_name}: {e}")

def all_nonempty_subsets(seq: List[str]) -> Iterable[Tuple[str, ...]]:
    """Generate all non-empty subsets of a sequence."""
    for r in range(1, len(seq) + 1):
        for comb in itertools.combinations(seq, r):
            yield tuple(sorted(comb))

def safe_execute(fn, *args, **kwargs):
    """Execute function with OOM error handling."""
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            clear_memory()
            raise OOMError(str(e))
        raise

def test_size_500_compatibility(args: Any, relations: Tuple[str, ...], agg: str, params: Dict, framework: str) -> bool:
    """
    Quick test if configuration works on size 500 using single instance.
    Returns True if no OOM, False if OOM.
    """
    test_args = deepcopy(args)
    test_args.relation_types = list(relations)
    test_args.agg = agg
    test_args.atsp_size = 500
    # Use absolute path to saved_dataset/ATSP_30x500 relative to project root
    root_dir = Path(__file__).resolve().parents[2]
    test_args.data_path = str(root_dir / 'saved_dataset' / 'ATSP_30x500')
    
    # Apply hyperparameters
    for k, v in params.items():
        setattr(test_args, k, v)
    
    fix_seed(getattr(test_args, "seed", 0))
    
    try:
        clear_memory()
        
        # Import modules as 'src.engine.*' and 'src.models.*' so that
        # relative imports inside those modules (e.g., '..data.*') resolve.
        tester_mod = locate_module(f"src.engine.test_{framework}")
        models_mod = locate_module(f"src.models.models_{framework}")
        
        TesterClass = (getattr(tester_mod, "ATSPTesterDGL", None) or 
                      getattr(tester_mod, "ATSPTesterPyG", None))
        get_model = getattr(models_mod, f"get_{framework}_model", None)
        
        if not TesterClass or not get_model:
            raise ImportError("Required classes not found")
        
        # Create model and test single instance
        model = get_model(test_args)
        tester = TesterClass(test_args)
        test_dataset = tester.create_test_dataset()
        
        if len(test_dataset) > 0:
            # Just test one instance - we don't care about the result, only if it OOMs
            _ = safe_execute(tester.test_instance_fast, model, test_dataset, 0)
        
        # Cleanup
        del model, tester, test_dataset
        clear_memory()
        return True
        
    except OOMError:
        clear_memory()
        return False
    except Exception as e:
        logger.warning(f"Size 500 test error: {str(e)[:100]}")
        clear_memory()
        return False

def save_current_best_model(args: Any, relations: Tuple[str, ...], agg: str, params: Dict, 
                           val_loss: float, framework: str, save_dir: str) -> str:
    """Save the current best model (overwrites previous best for this combination)."""
    rel_str = "_".join(relations)
    model_name = f"best_model_rel_{rel_str}_{agg}.pt"
    model_path = os.path.join(save_dir, model_name)
    # Setup training arguments for final model
    final_args = deepcopy(args)
    final_args.relation_types = list(relations)
    final_args.agg = agg
    for k, v in params.items():
        setattr(final_args, k, v)
    final_args.atsp_size = 50  # Train final model on target size
    fix_seed(getattr(final_args, "seed", 0))
    
    try:
        clear_memory()
        
        # Import modules using 'src.' to satisfy relative imports inside trainers
        trainer_mod = locate_module(f"src.engine.train_{framework}")
        models_mod = locate_module(f"src.models.models_{framework}")
        
        TrainerClass = (getattr(trainer_mod, "ATSPTrainerDGL", None) or 
                       getattr(trainer_mod, "ATSPTrainerPyG", None))
        get_model = getattr(models_mod, f"get_{framework}_model")
        
        # Train final model
        model = get_model(final_args)
        trainer = TrainerClass(final_args, save_model=False)
        results = safe_execute(trainer.train, model, 0)
        
        # Save model checkpoint (overwrites previous)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'relations': relations,
            'agg': agg,
            'hyperparameters': params,
            'validation_loss': val_loss,
            'args': vars(final_args),
            'training_results': results
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Saved new best model (val_loss={val_loss:.6f}): {model_path}")
        
        # Cleanup
        del model, trainer
        clear_memory()
        return model_path
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        clear_memory()
        return None

def create_objective_with_size_screening(args: Any, rel_subset: Tuple[str, ...], agg_choice: str, 
                                       framework: str, save_dir: str):
    """
    Create Optuna objective with size 500 pre-screening.
    Flow: Sample params -> Test size 500 -> If OK, train on size 50 -> Save if best so far
    """
    best_loss_so_far = [float('inf')]  # Mutable container to track best
    
    def objective(trial: optuna.trial.Trial) -> float:
        clear_memory()
        
        # Sample hyperparameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64])
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        num_gnn_layers = trial.suggest_categorical("num_gnn_layers", [1, 2, 3])
        lr_init = trial.suggest_float("lr_init", 1e-5, 1e-2, log=True)
        
        params = {
            "hidden_dim": hidden_dim,
            "num_heads": num_heads, 
            "num_gnn_layers": num_gnn_layers,
            "lr_init": lr_init
        }
        
        logger.info(f"Trial {trial.number}: Testing size 500 compatibility first...")
        
        # STEP 1: Test size 500 compatibility BEFORE training
        if not test_size_500_compatibility(args, rel_subset, agg_choice, params, framework):
            logger.info(f"Trial {trial.number}: FAILED size 500 test - returning high loss")
            return float('inf')  # High loss for configs that can't handle size 500
        
        logger.info(f"Trial {trial.number}: PASSED size 500 test - proceeding to training on size 50")
        
        # STEP 2: If size 500 works, train on size 50 for speed
        train_args = deepcopy(args)
        train_args.relation_types = list(rel_subset)
        train_args.agg = agg_choice
        train_args.atsp_size = 50  # Train on smaller size for speed
        for k, v in params.items():
            setattr(train_args, k, v)
        
        fix_seed(getattr(train_args, "seed", 0))
        
        try:
            # Import training modules using 'src.' to satisfy relative imports inside trainers
            trainer_mod = locate_module(f"src.engine.train_{framework}")
            models_mod = locate_module(f"src.models.models_{framework}")
            
            TrainerClass = (getattr(trainer_mod, "ATSPTrainerDGL", None) or 
                           getattr(trainer_mod, "ATSPTrainerPyG", None))
            get_model = getattr(models_mod, f"get_{framework}_model", None)
            
            if not TrainerClass or not get_model:
                raise ImportError("Required classes not found")
            
            # Ensure a valid training dataset directory
            if not getattr(train_args, 'data_dir', None):
                train_args.data_dir = str(Path(__file__).resolve().parents[2] / 'saved_dataset' / 'ATSP_3000x50')
            else:
                # If provided path looks invalid, fallback to default training set
                data_dir_path = Path(train_args.data_dir)
                if not data_dir_path.exists() or not (data_dir_path / 'train.txt').exists():
                    train_args.data_dir = str(Path(__file__).resolve().parents[2] / 'saved_dataset' / 'ATSP_3000x50')

            # Train model on size 50
            model = get_model(train_args)
            trainer = TrainerClass(train_args, save_model=False)
            
            results = safe_execute(trainer.train, model, trial_id=trial.number)
            val_loss = results['best_val_loss']
            
            logger.info(f"Trial {trial.number}: Training completed - val_loss={val_loss:.6f}")
            
            # STEP 3: If this is the best so far, save the model
            if val_loss < best_loss_so_far[0]:
                logger.info(f"Trial {trial.number}: NEW BEST! Saving model...")
                model_path = save_current_best_model(args, rel_subset, agg_choice, params, 
                                                   val_loss, framework, save_dir)
                if model_path:
                    best_loss_so_far[0] = val_loss
                    logger.info(f"Trial {trial.number}: Best model saved (loss={val_loss:.6f})")
            
            # Cleanup training
            del model, trainer
            clear_memory()
            
            return val_loss
            
        except OOMError:
            logger.warning(f"Trial {trial.number}: OOM during training on size 50")
            raise optuna.exceptions.TrialPruned()
        except Exception as e:
            logger.warning(f"Trial {trial.number}: Training failed: {e}")
            clear_memory()
            return float('inf')
    
    return objective

def _run_one_subset_worker(args, rel_subset, agg_method, n_trials, framework, output_dir):
    """
    Runs a single Optuna study for one (rel_subset, agg_method) in an isolated process.
    Writes result JSON to output_dir and returns nothing.
    """
    # per-process logging to file
    combo_name = f"rel_{'_'.join(rel_subset)}_{agg_method}"
    log_path = os.path.join(output_dir, f"{combo_name}.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger_proc = logging.getLogger(f"search.{combo_name}")
    logger_proc.setLevel(logging.INFO)
    logger_proc.handlers = []
    logger_proc.addHandler(fh)

    try:
        # Limit native threads per worker to reduce risk of crashes in MKL/OpenMP libs
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        import torch
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        clear_memory()
        study = optuna.create_study(direction="minimize")

        objective = create_objective_with_size_screening(
            args, rel_subset, agg_method, framework, output_dir
        )

        # very important: no thread-based parallelism in the worker
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        successful = [t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE and t.value != float('inf')]

        if successful:
            best_trial = min(successful, key=lambda t: t.value)
            result = {
                "combo_name": combo_name,
                "relations": rel_subset,
                "agg": agg_method,
                "best_params": best_trial.params,
                "best_val_loss": best_trial.value,
                "total_trials": len(study.trials),
                "successful_trials": len(successful)
            }
        else:
            result = {
                "combo_name": combo_name,
                "relations": rel_subset,
                "agg": agg_method,
                "best_params": None,
                "best_val_loss": float("inf"),
                "total_trials": len(study.trials),
                "successful_trials": 0
            }

        # write per-subset result
        result_path = os.path.join(output_dir, f"result_{combo_name}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        logger_proc.info(f"Finished {combo_name} with best={result['best_val_loss']:.6f}")
    except Exception as e:
        # write a failure marker so parent can proceed
        fail_path = os.path.join(output_dir, f"result_{combo_name}.json")
        with open(fail_path, "w") as f:
            json.dump({"combo_name": combo_name, "error": str(e)}, f, indent=2)
        logger_proc.exception(f"Subset {combo_name} failed: {e}")
    finally:
        try:
            clear_memory()
        except Exception:
            pass


def run_optuna_search(args: Any, n_trials: int = 20, n_jobs: int = 1):
    """
    Run a separate process for each relation-subset study.
    Keeps your size-500 screening and per-subset best-model saving logic.
    """
    output_dir = get_output_dir()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    framework = args.framework.lower()

    relation_subsets = list(all_nonempty_subsets(list(args.relation_types)))
    agg_methods = ["sum"]  # you said only one agg type

    logger.info(f"Starting search: {len(relation_subsets)} relation subsets × {len(agg_methods)} agg methods")
    logger.info("Each trial: Size 500 test -> Train on size 50 if passed -> Save if best")

    # ensure spawn start method
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    results = []
    saved_models = []  # your save_current_best_model still runs inside the worker

    # run each subset in a fresh process
    for rel_subset in relation_subsets:
        for agg_method in agg_methods:
            combo_name = f"rel_{'_'.join(rel_subset)}_{agg_method}"
            logger.info(f"\n=== Optimizing {combo_name}: {rel_subset} + {agg_method} ===")

            proc = mp.Process(
                target=_run_one_subset_worker,
                args=(args, rel_subset, agg_method, n_trials, framework, output_dir),
                daemon=False
            )
            proc.start()
            proc.join()  # sequential for maximum stability

            # read back result JSON
            result_path = os.path.join(output_dir, f"result_{combo_name}.json")
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    r = json.load(f)
                if "error" in r:
                    logger.warning(f"Subset {combo_name} failed: {r['error']}")
                else:
                    results.append(r)
                    logger.info(f"Best {combo_name}: loss={r['best_val_loss']:.6f} "
                                f"({r['successful_trials']}/{r['total_trials']} trials passed size 500)")
            else:
                logger.warning(f"No result file for {combo_name}")

            # light cleanup after each child
            clear_memory()

    # write summary
    summary_file = os.path.join(output_dir, "search_summary.txt")
    with open(summary_file, "w") as f:
        f.write("ATSP Hyperparameter Search Results\n")
        f.write("=" * 50 + "\n")
        f.write("Search strategy: Size 500 pre-screening + Size 50 training\n")
        f.write(f"Total combinations: {len(relation_subsets) * len(agg_methods)}\n")
        f.write(f"Successful combinations: {sum(1 for r in results if r['successful_trials'] > 0)}\n\n")
        f.write("Results by combination:\n")
        for r in results:
            f.write(f"  {r['combo_name']}: loss={r['best_val_loss']:.6f}\n")
            f.write(f"    Relations: {tuple(r['relations'])}\n")
            f.write(f"    Agg: {r['agg']}\n")
            f.write(f"    Best params: {r['best_params']}\n")
            f.write(f"    Trials: {r['successful_trials']}/{r['total_trials']} passed size 500\n\n")

    logger.info("\nSEARCH COMPLETED!")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Working combinations: {sum(1 for r in results if r['successful_trials'] > 0)}")

    return results, saved_models

if __name__ == "__main__":
    args = parse_args()
    n_trials = getattr(args, "n_trials", 1)
    
    try:
        results, models = run_optuna_search(args, n_trials=n_trials)
        logger.info(f"Search successful! Found {len(results)} working combinations.")
    except Exception as e:
        logger.exception(f"Search failed: {e}")
