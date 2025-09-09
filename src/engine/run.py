import importlib
import os
from .args import parse_args, smart_instantiate
from ..utils import fix_seed

import sys, os
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

    # map to module names in your repo
    if framework == 'dgl':
        trainer_module = 'src.engine.train_dgl'
        tester_module = 'src.engine.test_dgl'
    elif framework == 'pyg':
        trainer_module = 'src.engine.train_pyg'
        tester_module = 'src.engine.test_pyg'
    else:
        raise ValueError(f'Unknown framework: {framework}')

    if mode == 'train':
        mod = locate(trainer_module)
        # expect the trainer to expose a class named ATSPTrainerDGL or ATSPTrainerPyG named consistently
        TrainerClass = getattr(mod, 'ATSPTrainerDGL', None) or getattr(mod, 'ATSPTrainerPyG', None)
        if TrainerClass is None:
            raise AttributeError(f'{trainer_module} has no Trainer class (ATSPTrainerDGL/ATSPTrainerPyG)')
        trainer = TrainerClass(args)

        # load model factory
        models_mod = locate(f'src.models.models_{framework}')
        get_model = getattr(models_mod, f'get_{framework}_model', None)
        if get_model is None:
            raise AttributeError('Model factory not found in src.models (expected get_dgl_model/get_pyg_model/get_model)')
        model = get_model(args)  # pass args so factory can build with correct dims

        # run trials
        for trial in range(args.n_trials):
            print(f'=== Trial {trial} ===')
            results = trainer.train(model, trial_id=trial)
            # trainer.train saves checkpoints and results

    elif mode == 'test':
        # Load tester module
        mod = locate(tester_module)
        TesterClass = getattr(mod, 'ATSPTesterDGL', None) or getattr(mod, 'ATSPTesterPyG', None)
        if TesterClass is None:
            raise AttributeError(f'{tester_module} has no Tester class (ATSPTesterDGL/ATSPTesterPyG)')
        tester = TesterClass(args)

        # Load model factory
        models_mod = locate(f'src.models.models_{framework}')
        get_model = getattr(models_mod, f'get_{framework}_model', None)
        if get_model is None:
            raise AttributeError('Model factory not found in src.models (expected get_dgl_model/get_pyg_model)')
        model = get_model(args)
        results = tester.run_test(model)

        print(f"Testing completed. Results saved to {checkpoint_path}/test_atsp{args.atsp_size}/results.json")
    else:
        raise ValueError(f'Unknown mode: {mode}')

if __name__ == '__main__':
    main()
