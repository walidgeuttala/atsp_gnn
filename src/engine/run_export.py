import importlib
import os
import sys

from .args import parse_args
from ..utils import fix_seed

# Ensure src/ is importable for absolute imports inside modules
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

    if framework == 'dgl':
        trainer_module = 'src.engine.train_dgl'
        # Use the export tester instead of the standard one
        tester_module = 'src.engine.test_dgl_export'
    elif framework == 'pyg':
        trainer_module = 'src.engine.train_pyg'
        tester_module = 'src.engine.test_pyg'
    else:
        raise ValueError(f'Unknown framework: {framework}')

    if mode == 'train':
        mod = locate(trainer_module)
        TrainerClass = getattr(mod, 'ATSPTrainerDGL', None) or getattr(mod, 'ATSPTrainerPyG', None)
        if TrainerClass is None:
            raise AttributeError(f'{trainer_module} has no Trainer class (ATSPTrainerDGL/ATSPTrainerPyG)')
        trainer = TrainerClass(args, save_model=True)

        models_mod = locate(f'src.models.models_{framework}')
        get_model = getattr(models_mod, f'get_{framework}_model', None)
        if get_model is None:
            raise AttributeError('Model factory not found in src.models (expected get_dgl_model/get_pyg_model)')
        model = get_model(args)

        for trial in range(args.n_trials):
            print(f'=== Trial {trial} ===')
            trainer.train(model, trial_id=trial)

    elif mode == 'test':
        mod = locate(tester_module)
        TesterClass = getattr(mod, 'ATSPTesterDGLExport', None) or getattr(mod, 'ATSPTesterPyG', None)
        if TesterClass is None:
            raise AttributeError(f'{tester_module} has no Tester class (ATSPTesterDGLExport/ATSPTesterPyG)')
        tester = TesterClass(args)

        models_mod = locate(f'src.models.models_{framework}')
        get_model = getattr(models_mod, f'get_{framework}_model', None)
        if get_model is None:
            raise AttributeError('Model factory not found in src.models (expected get_dgl_model/get_pyg_model)')
        model = get_model(args)

        tester.run_test(model)

        # Print the results path next to checkpoint if model_path is a file
        base_dir = args.model_path
        if isinstance(base_dir, str) and base_dir.endswith('.pt'):
            base_dir = os.path.dirname(base_dir)
        ckpt_stem = os.path.splitext(os.path.basename(args.model_path))[0] if args.model_path else 'model'
        print(f"Testing completed. Results saved to {base_dir}/{ckpt_stem}/test_atsp{args.atsp_size}/results.json")
    else:
        raise ValueError(f'Unknown mode: {mode}')


if __name__ == '__main__':
    main()

