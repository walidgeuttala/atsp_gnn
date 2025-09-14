"""
Optuna hyperparameter search for ATSP with size 500 pre-screening.
Key features:
- For each Optuna trial: FIRST test config on size 500, then train on size 50 if no OOM
- If config fails size 500 test, return high loss immediately (no training)
- Save best working model during Optuna process (overwrites previous best)
- Memory management and OOM handling
"""
import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TQDM_DISABLE_MONITOR", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")  # optional
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from .args import parse_args
from ..utils import fix_seed

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
    test_args.data_path = '../saved_dataset/ATSP_30x500'
    
    # Apply hyperparameters
    for k, v in params.items():
        setattr(test_args, k, v)
    
    fix_seed(getattr(test_args, "seed", 0))
    
    try:
        clear_memory()
        
        # Import modules
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
        
        # Import modules
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
            # Import training modules
            trainer_mod = locate_module(f"src.engine.train_{framework}")
            models_mod = locate_module(f"src.models.models_{framework}")
            
            TrainerClass = (getattr(trainer_mod, "ATSPTrainerDGL", None) or 
                           getattr(trainer_mod, "ATSPTrainerPyG", None))
            get_model = getattr(models_mod, f"get_{framework}_model", None)
            
            if not TrainerClass or not get_model:
                raise ImportError("Required classes not found")
            
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

def run_optuna_search(args: Any, n_trials: int = 20, n_jobs: int = 1):
    """Main search function with size pre-screening."""
    output_dir = get_output_dir()
    framework = args.framework.lower()
    
    # Generate search space
    relation_subsets = list(all_nonempty_subsets(list(args.relation_types)))
    agg_methods = ["attn"]
    
    logger.info(f"Starting search: {len(relation_subsets)} relation subsets × {len(agg_methods)} agg methods")
    logger.info(f"Each trial: Size 500 test -> Train on size 50 (if passed) -> Save if best")
    
    search_results = []
    saved_models = []
    
    # Search each combination
    for rel_idx, rel_subset in enumerate(relation_subsets):
        for agg_method in agg_methods:
            combo_name = f"rel{rel_idx+1}_{agg_method}"
            logger.info(f"\n=== Optimizing {combo_name}: {rel_subset} + {agg_method} ===")
            
            clear_memory()
            
            # Run Optuna optimization with size pre-screening
            study = optuna.create_study(direction="minimize")
            objective = create_objective_with_size_screening(args, rel_subset, agg_method, 
                                                           framework, output_dir)
            
            try:
                study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
            except Exception as e:
                logger.warning(f"Optuna failed for {combo_name}: {e}")
                continue
            
            # Get best trial (only from trials that passed size 500 test)
            successful_trials = [t for t in study.trials 
                               if t.state == optuna.trial.TrialState.COMPLETE and t.value != float('inf')]
            
            if successful_trials:
                best_trial = min(successful_trials, key=lambda t: t.value)
                
                result = {
                    "combo_name": combo_name,
                    "relations": rel_subset,
                    "agg": agg_method,
                    "best_params": best_trial.params,
                    "best_val_loss": best_trial.value,
                    "total_trials": len(study.trials),
                    "successful_trials": len(successful_trials)
                }
                
                search_results.append(result)
                logger.info(f"Best {combo_name}: loss={best_trial.value:.6f} "
                          f"({len(successful_trials)}/{len(study.trials)} trials passed size 500)")
                
                # Model should already be saved during optimization
                model_name = f"best_model_rel_{hash(rel_subset)}_{agg_method}.pt"
                model_path = os.path.join(output_dir, model_name)
                if os.path.exists(model_path):
                    saved_models.append(model_path)
            else:
                logger.warning(f"No successful trials for {combo_name} (all failed size 500 test)")
    
    # Write summary
    summary_file = os.path.join(output_dir, "search_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"ATSP Hyperparameter Search Results\n")
        f.write(f"="*50 + "\n")
        f.write(f"Search strategy: Size 500 pre-screening + Size 50 training\n")
        f.write(f"Total combinations: {len(relation_subsets) * len(agg_methods)}\n")
        f.write(f"Successful combinations: {len(search_results)}\n")
        f.write(f"Models saved: {len(saved_models)}\n\n")
        
        f.write("Results by combination:\n")
        for result in search_results:
            f.write(f"  {result['combo_name']}: loss={result['best_val_loss']:.6f}\n")
            f.write(f"    Relations: {result['relations']}\n")
            f.write(f"    Agg: {result['agg']}\n")
            f.write(f"    Best params: {result['best_params']}\n")
            f.write(f"    Trials: {result['successful_trials']}/{result['total_trials']} passed size 500\n\n")
    
    logger.info(f"\nSEARCH COMPLETED!")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Working combinations: {len(search_results)}")
    logger.info(f"Saved models: {len(saved_models)}")
    
    return search_results, saved_models

if __name__ == "__main__":
    args = parse_args()
    n_trials = getattr(args, "n_trials", 1)
    
    try:
        results, models = run_optuna_search(args, n_trials=n_trials)
        logger.info(f"Search successful! Found {len(results)} working combinations.")
    except Exception as e:
        logger.exception(f"Search failed: {e}")