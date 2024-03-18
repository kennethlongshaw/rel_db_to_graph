import time
import optuna
import subprocess
import polars as pl
from dvc import api


def objective(trial: optuna.Trial):
    params = {
        # Define the hyperparameters to tune
        'train.learning_rate': trial.suggest_float("learning_rate", high=0.01, low=0.000001, log=True),
        'train.num_neighbors': trial.suggest_categorical('num_neighbors', list(range(5, 55, 5))),
        'train.num_layers': trial.suggest_categorical('num_layers', [2, 3, 4, 5, 6]),
        'train.hidden_channels': trial.suggest_categorical('hidden_channels', list(range(10, 110, 10))),
        'train.dropout': trial.suggest_categorical('dropout', list([d/100 for d in range(5, 55, 5)]))
    }

    param_str = " ".join([f'-S "{param_key}={param_value}"' for param_key, param_value in params.items()])

    # Run the pipeline experiment
    experiment_name = f"{trial.study.study_name}_{trial.number}"

    # Run DVC experiment for the stage
    subprocess.run(
        f'dvc exp run -n {experiment_name} {param_str}',
        shell=True,
        check=True
    )

    time.sleep(2)

    # Retrieve the accuracy metric from the DVC experiment
    accuracy = pl.DataFrame(api.exp_show()).filter(pl.col('Experiment') == pl.lit(experiment_name)).select(
        'val.best_accuracy').row(0)[0]

    # Return the accuracy metric to Optuna
    return float(accuracy)


def main():
    storage = 'sqlite:///gnn_studies.db'
    study_name = 'music_gnn_v2'

    # Create an Optuna study
    study = optuna.create_study(storage=storage,
                                study_name=study_name,
                                load_if_exists=True,
                                direction="maximize")

    # Optimize the objective function
    study.optimize(objective, n_trials=1)

    # Print the best hyperparameters and score
    print("Best hyperparameters:", study.best_params)
    print("Best score:", study.best_value)


if __name__ == '__main__':
    main()
