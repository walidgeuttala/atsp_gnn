import itertools
import importlib
import logging
import math
import os
import sys
import traceback
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import optuna
import torch

# make relative imports work when this file is executed as a script
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from .args import parse_args
from ..utils import fix_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("search")


def locate(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Cannot import {module_name}: {e}")


def all_nonempty_subsets(seq: List[str]) -> Iterable[Tuple[str, ...]]:
    """Generate all non-empty subsets (as tuples) of a list in deterministic order."""
    for r in range(1, len(seq) + 1):
        for comb in itertools.combinations(seq, r):
            yield tuple(sorted(comb))


def divisors(n: int) -> List[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _extract_val_scores(results: Any) -> Tuple[Optional[float], Optional[float]]:
    """Try to extract validation scores for sizes 50 and 100 from trainer results.

    Returns (score50, score100) where score may be None if not found.
    We interpret lower-is-better (e.g. loss). If your trainer reports metrics where higher is better,
    invert them before returning (or change objective accordingly).
    """
    # results might be a dict-like object
    candidates = {}
    if results is None:
        return None, None
    if isinstance(results, dict):
        for k, v in results.items():
            candidates[k.lower()] = v
    # try many key names commonly used
    possible_keys_50 = [
        "val_loss_50",
        "val_50",
        "val_metric_50",
        "val_score_50",
        "val_loss",
        "val_metric",
        "val_score",
    ]
    possible_keys_100 = [
        "val_loss_100",
        "val_100",
        "val_metric_100",
        "val_score_100",
    ]

    def extract_from_candidates(keys):
        for k in keys:
            if k in candidates:
                try:
                    return float(candidates[k])
                except Exception:
                    pass
        return None

    s50 = extract_from_candidates([k.lower() for k in possible_keys_50])
    s100 = extract_from_candidates([k.lower() for k in possible_keys_100])

    return s50, s100


class OOMError(Exception):
    pass


def safe_step(func, *args, **kwargs):
    """Run func(*args, **kwargs) and catch CUDA OOMs. Re-raises other exceptions."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg and "out of memory" in msg:
            # clear cache if torch is available
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            raise OOMError("CUDA OOM during run")
        else:
            raise


def objective_factory(args: Any, relation_subset: Tuple[str, ...], agg_choice: str, framework: str, n_epochs_optuna: int):
    """Return an Optuna objective function that will:
    - instantiate model with sampled params
    - train on atsp_size=50
    - validate on atsp_size=50 and 100
    - return average validation score (lower is better)
    """

    def objective(trial: optuna.trial.Trial) -> float:
        # sample hyperparams
        hidden_dim = int(trial.suggest_categorical("hidden_dim", [1, 2, 4, 8, 16, 32]))
        # choose num_heads as a divisor of hidden_dim (except 0)
        divs = divisors(hidden_dim)
        num_heads = int(trial.suggest_categorical("num_heads", [1, 2, 4, 8]))
        lr_init = float(trial.suggest_float("lr_init", 1e-5, 1e-2, log=True))
        num_gnn_layers = int(trial.suggest_int("num_gnn_layers", 1, 3))
        # prepare args copy for this run
        run_args = deepcopy(args)
        run_args.relation_types = list(relation_subset)
        run_args.agg = agg_choice
        run_args.hidden_dim = hidden_dim
        run_args.num_heads = num_heads
        run_args.lr_init = lr_init
        run_args.atsp_size = 50  # training size as requested
        run_args.num_gnn_layers = num_gnn_layers
        run_args.n_epochs = n_epochs_optuna  # short training for search
        # seed for reproducibility
        fix_seed(run_args.seed)

        if hidden_dim < num_heads:
            raise optuna.TrialPruned()  # skip this trial
        # instantiate modules
        trainer_module = f"src.engine.train_{framework}"
        tester_module = f"src.engine.test_{framework}"
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

            # run training -- wrapped to catch OOM
            try:
                results = safe_step(trainer.train, model, 0)
            except OOMError:
                logger.warning("OOM during training; dropping this trial")
                raise optuna.TrialPruned()

            # try to extract validation scores
            s50, s100 = _extract_val_scores(results)

            # If trainer didn't return both, attempt to run the tester (best-effort)
            if s50 is None or s100 is None:
                try:
                    # if trainer saved model to args.model_path, tester can evaluate it.
                    tester_mod = locate(tester_module)
                    TesterClass = getattr(tester_mod, "ATSPTesterDGL", None) or getattr(tester_mod, "ATSPTesterPyG", None)
                    tester = TesterClass(run_args)

                    # eval at 50
                    run_args.atsp_size = 50
                    try:
                        res50 = safe_step(tester.run_test, model)
                    except OOMError:
                        logger.warning("OOM during validation at size 50; marking trial as pruned")
                        raise optuna.TrialPruned()

                    s50_new, _ = _extract_val_scores(res50)
                    if s50 is None:
                        s50 = s50_new

                    # eval at 100
                    run_args.atsp_size = 100
                    try:
                        res100 = safe_step(tester.run_test, model)
                    except OOMError:
                        logger.warning("OOM during validation at size 100; marking trial as pruned")
                        raise optuna.TrialPruned()

                    _, s100_new = _extract_val_scores(res100)
                    if s100 is None:
                        s100 = s100_new
                except OOMError:
                    raise optuna.TrialPruned()
                except Exception:
                    # best-effort: if tester isn't available or fails, continue with whatever we have
                    logger.debug("Tester evaluation failed: %s", traceback.format_exc())

            # If still missing any scores, penalize this trial
            if s50 is None and s100 is None:
                logger.warning("No validation scores found; penalizing trial")
                return float("inf")
            # replace missing one with the other (conservative)
            if s50 is None:
                s50 = s100
            if s100 is None:
                s100 = s50

            avg_val = (s50 + s100) / 2.0
            # report intermediate value
            trial.report(avg_val, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return avg_val

        except OOMError:
            # mark as pruned when OOM occurs
            raise optuna.TrialPruned()
        except Exception:
            logger.exception("Unexpected error in objective: %s", traceback.format_exc())
            # return a very bad score so this config is not selected
            return float("inf")

    return objective


def run_search(args: Any, n_optuna_trials: int = 20, n_jobs: int = 1, n_epochs_optuna: int = 1):
    framework = args.framework.lower()

    relation_space = list(all_nonempty_subsets(list(args.relation_types)))
    agg_space = ["sum", "concat"]

    logger.info("Starting grid over %d relation subsets x %d aggs = %d combinations",
                len(relation_space), len(agg_space), len(relation_space) * len(agg_space))

    best_overall = {"score": float("inf"), "config": None}
    results_summary = []

    for rel_subset in relation_space:
        for agg_choice in agg_space:
            logger.info("Searching for relations=%s agg=%s", rel_subset, agg_choice)

            study = optuna.create_study(direction="minimize")
            objective = objective_factory(args, rel_subset, agg_choice, framework, n_epochs_optuna)

            try:
                study.optimize(objective, n_trials=n_optuna_trials, n_jobs=n_jobs)
            except Exception as e:
                logger.warning("Optuna run failed: %s", e)

            best = getattr(study, "best_trial", None)
            if best is None or best.value is None:
                logger.info("No successful trials for rel=%s agg=%s", rel_subset, agg_choice)
                best_value = float("inf")
                best_params = {}
            else:
                best_value = best.value
                best_params = best.params
            logger.info("Best trial for rel=%s agg=%s -> value=%.6f params=%s",
                        rel_subset, agg_choice, best_value, best_params)
            config = {
                "relations": rel_subset,
                "agg": agg_choice,
                "best_params": best.params if best else {},
                "best_value": best.value if best else float('inf')
            }
            results_summary.append(config)

            if best and best.value < best_overall["score"]:
                best_overall["score"] = best.value
                best_overall["config"] = deepcopy(config)

    logger.info("Search finished. Best overall: %s", best_overall)

    # final full test for the best configuration on size 500
    if best_overall["config"] is not None:
        best_conf = best_overall["config"]
        logger.info("Running final training & test for best config: %s", best_conf)

        final_args = deepcopy(args)
        final_args.relation_types = list(best_conf["relations"])
        final_args.agg = best_conf["agg"]
        for k, v in best_conf["best_params"].items():
            setattr(final_args, k, v)
        final_args.atsp_size = 50

        fix_seed(final_args.seed)
        try:
            trainer_mod = locate(f"src.engine.train_{final_args.framework}")
            TrainerClass = getattr(trainer_mod, "ATSPTrainerDGL", None) or getattr(trainer_mod, "ATSPTrainerPyG", None)
            models_mod = locate(f"src.models.models_{final_args.framework}")
            get_model = getattr(models_mod, f"get_{final_args.framework}_model")

            model = get_model(final_args)
            trainer = TrainerClass(final_args)

            try:
                safe_step(trainer.train, model, 0)
            except OOMError:
                logger.warning("OOM during final training; aborting final test")
                return results_summary, best_overall

            # Save the best model with descriptive name (no timestamp)
            rel_name = "-".join(final_args.relation_types)
            model_filename = f"{final_args.model}_{final_args.agg}_{rel_name}_best.pt"
            save_path = os.path.join(getattr(args, "tb_dir", "./runs"), model_filename)
            torch.save(model.state_dict(), save_path)
            logger.info("Saved best model to %s", save_path)

            # run tester on 500
            final_args.atsp_size = 500
            tester_mod = locate(f"src.engine.test_{final_args.framework}")
            TesterClass = getattr(tester_mod, "ATSPTesterDGL", None) or getattr(tester_mod, "ATSPTesterPyG", None)
            tester = TesterClass(final_args)
            try:
                test_res = safe_step(tester.run_test, model)
                logger.info("Final test results: %s", test_res)
            except OOMError:
                logger.warning("OOM during final test on size 500; dropping final result")
        except Exception:
            logger.exception("Failed during final test stage: %s", traceback.format_exc())

    # save summary to disk
    out_path = getattr(args, "tb_dir", "./runs")
    os.makedirs(out_path, exist_ok=True)
    summary_file = os.path.join(out_path, "hyperparam_search_summary.yaml")
    try:
        import yaml
        with open(summary_file, "w") as f:
            yaml.safe_dump({"results": results_summary, "best": best_overall}, f)
        logger.info("Wrote search summary to %s", summary_file)
    except Exception:
        logger.exception("Failed to write summary file")

    return results_summary, best_overall


if __name__ == "__main__":
    parser = parse_args()
    parser.n_trials = 1
    N_OPTUNA_TRIALS = 2
    N_JOBS = 1

    try:
        results, best = run_search(parser, n_optuna_trials=N_OPTUNA_TRIALS, n_jobs=N_JOBS)
        logger.info("Search completed. Best: %s", best)
    except Exception:
        logger.exception("Search failed: %s", traceback.format_exc())
