
import argparse
from pathlib import Path



from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.epsilon_value_estimator.iterative_epsilon_value_estimator import (
    IterativeEpsilonValueEstimator,
)
from ada_verona.verification_module.symphony_module import SymphonyModule
import numpy as np 

if __name__ == "__main__":
    # parse arguments from batch script.
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_verification_context", type=Path)
    parser.add_argument("--base_path_experiment_repository")
    parser.add_argument("--network_folder")
    parser.add_argument("--experiment_name")
    parser.add_argument("--epsilon_start", type=float)   
    parser.add_argument("--epsilon_end",type=float)
    args = parser.parse_args()

    # reload experiment repository
    baby_experiment_repository = ExperimentRepository(
        base_path=Path(args.base_path_experiment_repository), network_folder=Path(args.network_folder)
    )
    baby_experiment_repository.load_experiment(experiment_name=args.experiment_name)

 
    if args.epsilon_end == args.epsilon_start:
        quit()
    print(args.epsilon_start)
    eps_list = np.arange(0.001,args.epsilon_end, 0.002)

    verifier = SymphonyModule(timeout=3600, symphony_path="/home/annelot/coinbrew/SYMPHONY/SYMPHONY/Examples/milp")

    # create the binary search module
    epsilon_value_estimator = IterativeEpsilonValueEstimator(epsilon_value_list=eps_list.copy(), verifier=verifier)

    # This is the same verification context as in the main file and
    # its loaded from the yaml file we created in the main file
    verification_context = baby_experiment_repository.load_verification_context_from_yaml(
        Path(args.file_verification_context)
    )

    # get the epsilon value and save it in the reloaded repository at the correct place
    epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context, reverse_search=True)
    baby_experiment_repository.save_result(epsilon_value_result)
