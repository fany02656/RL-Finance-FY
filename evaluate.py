import logging; logging.basicConfig(level=logging.INFO)
import os
import pandas as pd
from numpy.random import default_rng
from stable_baselines3 import PPO
from tqdm import trange, tqdm

from train_parallel import get_training_environment, get_save_dir, hyperparams_tuple_to_dict
from train_parallel import HYPERPARAM_GRID, TRADING_DAYS_IN_YEAR

CHECKPOINTS_BASE_DIR = "checkpoints/euro_hedging_low_kappa"


def main():
    env = get_training_environment()
    # NOTE For out-of-sample evaluation, we do NOT want to reuse data across episodes.
    env.pricing_source.mkt_data_source.data_reuse_num_episodes = None

    # NOTE We shuffle HYPERPARAM_GRID just so that we get a good cross-section
    # of hyperparam results early on in evaluation, rather than having to wait
    # for a long time to see results for different values of all the hyperparams
    rng = default_rng()
    rng.shuffle(HYPERPARAM_GRID)

    for hyperparams_tuple in tqdm(HYPERPARAM_GRID):
        hyperparams_dict = hyperparams_tuple_to_dict(hyperparams_tuple)
        model_dir = get_save_dir(CHECKPOINTS_BASE_DIR, hyperparams_dict)

        model_save_path = os.path.join(model_dir, "fully_trained_model")
        try:
            model = PPO.load(model_save_path)
        except FileNotFoundError:
            logging.error(f"Could not find file for hyperparams: {hyperparams_dict} . Skipping ...")
            continue

        NUM_YEARS_OUT_OF_SAMPLE = 100
        tqdm.write(f"Hyperparams: {hyperparams_dict}")
        tqdm.write(f"Evaluating on {NUM_YEARS_OUT_OF_SAMPLE} years of data.")

        evaluation_histories = []
        reward_histories = []

        for i in trange(NUM_YEARS_OUT_OF_SAMPLE):
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


if __name__ == '__main__':
    main()
