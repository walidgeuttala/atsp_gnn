import importlib
import os
import sys
from .args import parse_args, smart_instantiate
from ..utils import fix_seed

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


def locate(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f'Cannot import {module_name}: {e}')


def main():
    args = parse_args()
    fix_seed(args.seed)

    framework = args.framework.lower()
    mode = args.mode.lower()

    # Default modules
    if framework == 'dgl':
        trainer_module = 'src.engine.train_dgl'
        tester_module = 'src.engine.test_dgl'
        large_tester_module = 'src.engine.test_dgl_large'
    elif framework == 'pyg':
        trainer_module = 'src.engine.train_pyg'
        tester_module = 'src.engine.test_pyg'
        large_tester_module = None  # not implemented
    else:
        raise ValueError(f'Unknown framework: {framework}')

    if mode == 'train':
        mod = locate(trainer_module)
        TrainerClass = getattr(mod, 'ATSPTrainerDGL', None) or getattr(mod, 'ATSPTrainerPyG', None)
        if TrainerClass is None:
            raise AttributeError(f'{trainer_module} has no Trainer class (ATSPTrainerDGL/ATSPTrainerPyG)')
        trainer = TrainerClass(args)

        models_mod = locate(f'src.models.models_{framework}')
        get_model = getattr(models_mod, f'get_{framework}_model', None)
        if get_model is None:
            raise AttributeError('Model factory not found in src.models (expected get_dgl_model/get_pyg_model)')
        model = get_model(args)

        for trial in range(args.n_trials):
            print(f'=== Trial {trial} ===')
            trainer.train(model, trial_id=trial)

    elif mode == 'test':
        # Decide whether to use large tester
        use_large = hasattr(args, 'sub_size') and args.sub_size is not None and args.sub_size < args.atsp_size
        if use_large:
            mod = locate(large_tester_module)
            TesterClass = getattr(mod, 'ATSPTesterDGLLarge', None)
        else:
            mod = locate(tester_module)
            TesterClass = getattr(mod, 'ATSPTesterDGL', None) or getattr(mod, 'ATSPTesterPyG', None)

        if TesterClass is None:
            raise AttributeError(f'No tester class found in {tester_module}')

        tester = TesterClass(args)

        models_mod = locate(f'src.models.models_{framework}')
        get_model = getattr(models_mod, f'get_{framework}_model', None)
        if get_model is None:
            raise AttributeError('Model factory not found in src.models (expected get_dgl_model/get_pyg_model)')
        model = get_model(args)

        results = tester.run_test(model)
        # Derive results directory next to checkpoint unless overridden
        base_dir = getattr(args, 'results_dir', None)
        if not base_dir:
            base_dir = args.model_path
            if isinstance(base_dir, str) and base_dir.endswith('.pt'):
                base_dir = os.path.dirname(base_dir)
        print(f"Testing completed. Results saved to {base_dir}/test_atsp{args.atsp_size}/results.json")

    else:
        raise ValueError(f'Unknown mode: {mode}')


if __name__ == '__main__':
    main()
