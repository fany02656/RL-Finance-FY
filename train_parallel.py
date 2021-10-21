from __future__ import annotations

import os
import numpy as np
import sys
import pickle
from itertools import product
from collections import OrderedDict
from copy import deepcopy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from torch import nn
from typing import Tuple

from golds.callbacks import LoggerCallback
from golds.contracts import Currency, Stock, Option, OptionFlavor, OptionStyle, Holdings
from golds.env import AmericanOptionEnv
from golds.mkt_data import PricingSource, SingleStockGBMMarketDataSource
from golds.params import GBMParams
from golds.reward_functions import NaiveHedgingRewardFunction
from golds.tcost import NaiveTransactionCostModel

REWARD_KAPPA = 0.1
INITIAL_WEALTH = 1e7
SELF_FINANCING_LAMBDA = 1000
TRADING_DAYS_IN_YEAR = 252

SAVE_DIR_BASE = "/scratch/ars991/checkpoints/american_hedging_continuous_action"

MAX_JOB_ARRAY_SIZE = 2048

# XXX If you make any changes to the hyperparameter grid (changing the
# hyperparams we iterate over, or changing the order in which we iterate over
# them), make sure to also change the hyperparams_tuple_to_dict function below.
GAMMA_GRID = list(np.linspace(0.8, 0.99, 8))
GAE_LAMBDA_GRID = list(np.linspace(0.7, 0.99, 8))
ENT_COEF_GRID = list(np.linspace(0.15, 0.3, 8))
VF_COEF_GRID = [0.5]
MAX_GRAD_NORM_GRID = [0.5]
HYPERPARAM_GRID = list(product(GAMMA_GRID, GAE_LAMBDA_GRID, ENT_COEF_GRID, VF_COEF_GRID, MAX_GRAD_NORM_GRID))
assert len(HYPERPARAM_GRID) <= MAX_JOB_ARRAY_SIZE, f"hyperparam grid is too large for max job array size {MAX_JOB_ARRAY_SIZE}"

print(f"size of hyperparam grid: {len(HYPERPARAM_GRID)}")


def hyperparams_tuple_to_dict(hyperparams_tuple: Tuple) -> OrderedDict[str, float]:
    # XXX If you change the hyperparam grid to iterate over different
    # hyperparameters, make sure to change this function correspondingly!
    gamma, gae_lambda, ent_coef, vf_coef, max_grad_norm = hyperparams_tuple
    return OrderedDict([
        ('gamma', gamma),
        ('gae_lambda', gae_lambda),
        ('ent_coef', ent_coef),
        ('vf_coef', vf_coef),
        ('max_grad_norm', max_grad_norm)
    ])


def get_save_dir(base_dir: str, hyperparams_dict: OrderedDict[str, float]) -> str:
    dirnames = [base_dir] + list(f"{key}={val:.3f}" for key, val in hyperparams_dict.items())
    return os.path.join(*dirnames)


def get_training_environment():
    aapl = Stock(ticker="AAPL", is_tradable=True)
    warrant = Option(
        strike=100,
        expiry_time=1.,
        underlying=aapl,
        flavor=OptionFlavor.PUT,
        style=OptionStyle.AMERICAN,
        is_tradable=False
    )
    cash = Currency(code="USD", is_tradable=False)

    initial_holdings: Holdings = {
        aapl: -50.,
        warrant: 100.,
        cash: INITIAL_WEALTH,
    }

    universe = list(initial_holdings.keys())

    gbm_params = GBMParams(mu=0.5, sigma=0.07, risk_free_rate=0.)

    mkt_data_source = SingleStockGBMMarketDataSource(universe, gbm_params, data_reuse_num_episodes=5*3000)
    tcost_model = NaiveTransactionCostModel(universe)
    pricing_source = PricingSource(mkt_data_source, tcost_model)

    return AmericanOptionEnv(
        episode_length=50,
        pricing_source=pricing_source,
        reward_function=NaiveHedgingRewardFunction(kappa=REWARD_KAPPA, initial_holdings=initial_holdings, reward_clip_range=(-50, 20)),
        actions_config=list(range(-100, 101))
    )


def main(hyperparam_grid_idx: int):
    if hyperparam_grid_idx >= len(HYPERPARAM_GRID):
        return

    env = get_training_environment()
    env_copy: AmericanOptionEnv = deepcopy(env)
    # NOTE we make env_copy and call check_env on it, because check_env calls reset() which would mess up the original env
    check_env(env_copy)

    # TODO experiment with gamma, gae_lambda, ent_coef, vf_coef, max_grad_norm (kwargs to PPO.__init__)
    # TODO experiment with batch size (how to do this?)
    # TODO Lerrel says entropy related to exploration -- increase ent_coef if agent is not exploring enough
    # TODO experiment with different number of hidden nodes per layer in "net_arch" (64? 128? more?)
    # TODO reward clipping (*)
    # TODO use t-costs to ensure that the agent does not over-trade
    # TODO check that average reward converges
    # TODO reduce GBM variance such that the entire 50-period episode has vol equivalent to 10 trading days (*)
    # TODO maybe exercise at time of expiry for Euro options (or American without early exercise) and let agent get final reward
    # TODO try continuous action space
    # TODO try transaction costs (this is easily implemented in the RewardFunction.evaluate_reward method)

    hyperparams_tuple = HYPERPARAM_GRID[hyperparam_grid_idx]
    hyperparams_dict = hyperparams_tuple_to_dict(hyperparams_tuple)
    print("Hyperparameter settings:")
    for k, v in hyperparams_dict.items():
        print(f"\t{k} = {v}")
    save_dir = get_save_dir(SAVE_DIR_BASE, hyperparams_dict)
    os.makedirs(save_dir, exist_ok=True)

    policy_kwargs = {"activation_fn": nn.ReLU, "net_arch": [32]*5}
    model = PPO(MlpPolicy, env, verbose=1, learning_rate=1e-4, policy_kwargs=policy_kwargs, **hyperparams_dict)
    logger_callback = LoggerCallback(save_path=os.path.join(save_dir, "rl_logs.json"), save_freq=1000)
    checkpoint_callback = CheckpointCallback(save_freq=20_000, save_path=save_dir, name_prefix='model_checkpoint')
    # N_YEARS_TRAINING = 50_000
    # TOTAL_TRAINING_TIMESTEPS = N_YEARS_TRAINING*TRADING_DAYS_IN_YEAR
    TOTAL_TRAINING_TIMESTEPS = 12_500_000
    model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS, callback=[logger_callback, checkpoint_callback])

    # TODO should log the training output here somehow (loss over time). I guess we can recover it from the checkpoints?
    with open(os.path.join(save_dir, "training_env.pkl"), "w+b") as f:
        pickle.dump(env, f)
    model.save(os.path.join(save_dir, "fully_trained_model"))


if __name__ == '__main__':
    main(int(sys.argv[1]))
