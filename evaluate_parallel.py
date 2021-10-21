from __future__ import annotations

import logging; logging.basicConfig(level=logging.INFO)
import os
import pandas as pd
from collections import OrderedDict
from copy import deepcopy
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Pool, Queue
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from typing import Tuple, List

from train_parallel import get_training_environment
from train_parallel import TRADING_DAYS_IN_YEAR

N_PARALLEL_WORKERS = 4
CHECKPOINTS_BASE_DIR = "checkpoints/american_hedging"


def initialize_worker(logging_queue: Queue):
    queue_handler = QueueHandler(logging_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(queue_handler)


def get_hyperparams_dict_from_save_dir(model_dir: str) -> OrderedDict[str, float]:
    path_components: List[str] = model_dir.split('/')
    hyperparams_dict = OrderedDict()
    for component in path_components:
        try:
            key, val = component.split('=')
        except ValueError:
            continue

        hyperparams_dict[key] = float(val)

    return hyperparams_dict


def evaluate_out_of_sample(worker_input_tuple: Tuple[OrderedDict[str, float], int]):
    model_dir, num_years_out_of_sample = worker_input_tuple
    hyperparams_dict: OrderedDict[str, float] = get_hyperparams_dict_from_save_dir(model_dir)

    env = get_training_environment()
    check_env(deepcopy(env))
    # NOTE For out-of-sample evaluation, we do NOT want to reuse data across episodes.
    env.pricing_source.mkt_data_source.data_reuse_num_episodes = None

    model_save_path = os.path.join(model_dir, "fully_trained_model")
    try:
        model = PPO.load(model_save_path)
    except FileNotFoundError:
        logging.error(f"Could not find file for hyperparams: {hyperparams_dict} . Skipping ...")
        return

    logging.info(f"Evaluating hyperparams={hyperparams_dict} on {num_years_out_of_sample} years of data.")

    evaluation_histories = []
    reward_histories = []

    for i in range(num_years_out_of_sample):
        evaluation_records = []
        assert not env.pricing_source.mkt_data_source._should_read_from_cache
        assert not env.pricing_source.mkt_data_source._should_write_to_cache
        obs = env.reset()

        for day in range(TRADING_DAYS_IN_YEAR):
            prices = env.observation_array_to_dict(obs)
            # print(f"Observed prices: {prices}")
            action, _states = model.predict(obs)
            trade = env.action_array_to_dict(action)
            # print(f"Executed trade: {trade}")
            obs, rewards, dones, info = env.step(action)
            # print(f"Received reward: {rewards}")
            evaluation_records.append({"obs": prices, "action": trade, "reward": rewards})
            env.render()

        evaluation_history = pd.DataFrame.from_records(evaluation_records)
        reward_history = pd.DataFrame.from_records(env.reward_function._reward_records)

        evaluation_history["run_number"] = i
        reward_history["run_number"] = i

        evaluation_histories.append(evaluation_history)
        reward_histories.append(reward_history)

    evaluation_history = pd.concat(evaluation_histories, ignore_index=True)
    reward_history = pd.concat(reward_histories, ignore_index=True)

    evaluation_history.to_hdf(os.path.join(model_dir, "evaluation_history.h5"), key="df", mode="w")
    reward_history.to_hdf(os.path.join(model_dir, "reward_history.h5"), key="df", mode="w")


def main():
    # Evaluate each point in the hyperparam grid on 100 out-of-sample years
    NUM_YEARS_OUT_OF_SAMPLE = 100

    worker_input_grid = [
        (dirpath, NUM_YEARS_OUT_OF_SAMPLE)
        for dirpath, dirnames, filenames in os.walk(CHECKPOINTS_BASE_DIR)
        if filenames
    ]

    logging_queue = Queue()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    listener = QueueListener(logging_queue, handler)
    listener.start()

    with Pool(N_PARALLEL_WORKERS, initialize_worker, (logging_queue,)) as pool:
        pool.map(evaluate_out_of_sample, worker_input_grid, chunksize=8)

    listener.stop()


if __name__ == '__main__':
    main()
