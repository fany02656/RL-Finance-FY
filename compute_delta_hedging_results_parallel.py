from __future__ import annotations

import logging; logging.basicConfig(level=logging.INFO)
import os
import gc
import pandas as pd
import pickle

from collections import OrderedDict
from logging.handlers import QueueListener
from multiprocessing import Pool, Queue
from tqdm import tqdm

from golds.options_math import andersen, black_scholes
from golds.contracts import Option, OptionStyle, Stock
from train_parallel import TRADING_DAYS_IN_YEAR
from evaluate_parallel import CHECKPOINTS_BASE_DIR, initialize_worker, get_hyperparams_dict_from_save_dir


N_PARALLEL_WORKERS = 8
# TODO to make this proper, we should store these parameters (mu, sigma, r, etc.) at evaluation time
# Then, when we do this downstream analysis, we should read them from the evaluation phase's output.
NUM_SHARES_UNDERLYING = 100
SIGMA = 0.2
RISK_FREE_RATE = 0.
RECOMPUTE_EXISTING_RESULTS = False


def compute_delta(t: float, option: Option, s: float, change_in_s: float = 0.01) -> float:
    if option.style == OptionStyle.AMERICAN:
        p1 = andersen(s-(change_in_s/2), option.strike, option.expiry_time, t, SIGMA, option.flavor, RISK_FREE_RATE)
        p2 = andersen(s+(change_in_s/2), option.strike, option.expiry_time, t, SIGMA, option.flavor, RISK_FREE_RATE)
        return (p2-p1)/change_in_s
    else:
        assert option.style == OptionStyle.EUROPEAN
        tau = option.expiry_time - t
        price, delta, d1, d2 = black_scholes(tau, s, option.strike, SIGMA, option.flavor, RISK_FREE_RATE, return_aux_values=True)
        return delta


def process_hyperparams_results(rewards_df: pd.DataFrame):
    delta_per_run = []
    stock_holdings_per_run = []
    stock_prices_per_run = []

    # XXX get rid of tqdm if running in parallel!
    for episode_number in tqdm(sorted(rewards_df["run_number"].unique())):
        if episode_number % 100 == 0:
            gc.collect()

        delta = []
        stock_holdings = []
        stock_prices = []

        curr_run_rewards_df = rewards_df[rewards_df["run_number"] == episode_number].copy()
        curr_run_rewards_df.reset_index(inplace=True, drop=True)

        for i, row in curr_run_rewards_df.iterrows():
            option, = [asset for asset in row['curr_holdings'].keys() if isinstance(asset, Option)]

            prices = row['curr_prices']
            stock, = [asset for asset in row['curr_holdings'].keys() if isinstance(asset, Stock)]
            s = prices[stock]

            delta.append(compute_delta(row['idx_in_episode']/float(TRADING_DAYS_IN_YEAR), option, s))
            stock_holdings.append(row['curr_holdings'][stock])
            stock_prices.append(s)

        delta = pd.Series(delta)
        stock_holdings = pd.Series(stock_holdings)
        stock_prices = pd.Series(stock_prices)

        delta_per_run.append(delta)
        stock_holdings_per_run.append(stock_holdings)
        stock_prices_per_run.append(stock_prices)

    delta = pd.concat(delta_per_run, axis='columns')
    stock_holdings = pd.concat(stock_holdings_per_run, axis='columns')
    stock_prices = pd.concat(stock_prices_per_run, axis='columns')

    return delta, stock_holdings, stock_prices


def compute_delta_hedging_results_for_hyperparams(model_dir: str):
    hyperparams_dict: OrderedDict[str, float] = get_hyperparams_dict_from_save_dir(model_dir)

    try:
        rewards_df = pd.read_hdf(os.path.join(model_dir, "reward_history.h5"))
    except FileNotFoundError:
        logging.warning(f"reward_history.h5 not found for hyperparams={hyperparams_dict}. Perhaps not done evaluating?")
        return

    output_fpath = os.path.join(model_dir, "delta_hedging_analysis.h5")
    results_exist = False
    try:
        with pd.HDFStore(output_fpath, 'r') as store:
            results_exist = all(key in store for key in ('delta', 'stock_holdings', 'stock_prices', 'hyperparams'))
    except (FileNotFoundError, OSError):
        pass
    if results_exist and not RECOMPUTE_EXISTING_RESULTS:
        logging.info(f"Found existing results for hyperparams={hyperparams_dict}; not recomputing ...")
        return

    logging.info(f"Analyzing delta hedging results for hyperparams={hyperparams_dict} ...")
    delta, stock_holdings, stock_prices = process_hyperparams_results(rewards_df)

    try:
        os.remove(output_fpath)
    except FileNotFoundError:
        pass
    dfs_to_output = {'delta': delta, 'stock_holdings': stock_holdings, 'stock_prices': stock_prices}
    for df_name, df in dfs_to_output.items():
        df.to_hdf(output_fpath, key=df_name)
    pd.Series(hyperparams_dict).to_hdf(output_fpath, key='hyperparams')


def main():
    with open("dirs_sorted_by_goodness.pkl", 'rb') as f:
        dirs_sorted_by_goodness = pickle.load(f)

    for save_dir in tqdm(dirs_sorted_by_goodness):
        compute_delta_hedging_results_for_hyperparams(save_dir)

    return

    worker_input_grid = [dirpath for dirpath, dirnames, filenames in os.walk(CHECKPOINTS_BASE_DIR) if filenames]

    logging_queue = Queue()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    listener = QueueListener(logging_queue, handler)
    listener.start()

    with Pool(N_PARALLEL_WORKERS, initialize_worker, (logging_queue,)) as pool:
        pool.map(compute_delta_hedging_results_for_hyperparams, worker_input_grid, chunksize=8)

    listener.stop()


if __name__ == '__main__':
    main()
