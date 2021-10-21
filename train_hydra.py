from __future__ import annotations

import hydra
import itertools
import logging
import numpy as np
import os
import pickle
from copy import deepcopy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DDPG
from torch import nn
from typing import List
from omegaconf import DictConfig
from stable_baselines3.ddpg import MlpPolicy

from golds.callbacks import ActionFunctionCallback, LoggerCallback
from golds.contracts import Currency, Stock, Option, OptionFlavor, OptionStyle, Holdings, Valuation
from golds.env import AmericanOptionEnv
from golds.mkt_data import PricingSource, SingleStockGBMMarketDataSource
from golds.params import GBMParams
from golds.reward_functions import NaiveHedgingRewardFunction, RewardFunction
from golds.tcost import NaiveTransactionCostModel


def get_training_environment(cfg: DictConfig, save_dir: str):
    aapl = Stock(ticker="AAPL", is_tradable=True)
    # TODO we can also make the option parameters part of the config
    warrant = Option(
        strike=100,
        expiry_time=1.,
        underlying=aapl,
        flavor=OptionFlavor.CALL,
        style=OptionStyle.EUROPEAN,
        is_tradable=False
    )
    cash = Currency(code="USD", is_tradable=False)

    initial_holdings: Holdings = {
        aapl: cfg['init_stock_holdings'],
        warrant: cfg['init_option_holdings'],
        cash: cfg['init_wealth'],
    }

    universe = list(initial_holdings.keys())

    gbm_params = GBMParams(mu=cfg['gbm_mu'], sigma=cfg['gbm_sigma'], risk_free_rate=cfg['gbm_r'])

    mkt_data_source = SingleStockGBMMarketDataSource(universe, gbm_params, data_reuse_num_episodes=cfg['data_reuse_num_episodes'])
    tcost_model = NaiveTransactionCostModel(universe)
    pricing_source = PricingSource(mkt_data_source, tcost_model)

    reward_clip_range = (cfg['reward_clip_min'], cfg['reward_clip_max'])
    reward_records_save_path = os.path.join(save_dir, "reward_history.h5")

    valid_actions = list(range(cfg['action_min'], cfg['action_max']+1))

    reward_function: RewardFunction = NaiveHedgingRewardFunction(
        kappa=cfg['reward_kappa'],
        initial_holdings=initial_holdings,
        reward_clip_range=reward_clip_range,
        reward_records_save_path=reward_records_save_path
    )

    return AmericanOptionEnv(
        episode_length=50,
        pricing_source=pricing_source,
        reward_function=reward_function,
        actions_config=valid_actions
    )


def get_observation_grid(env: AmericanOptionEnv) -> List[Valuation]:
    universe = env.pricing_source.universe

    ASSET_PRICE_STEP_SIZE = 0.10
    STOCK_PRICE_MIN = 0.10
    STOCK_PRICE_MAX = 300.00
    STOCK_N_STEPS = int(1+(STOCK_PRICE_MAX-STOCK_PRICE_MIN)/ASSET_PRICE_STEP_SIZE)

    price_grids = []
    for asset in universe:
        if isinstance(asset, Stock):
            price_grids.append(np.linspace(STOCK_PRICE_MIN, STOCK_PRICE_MAX, num=STOCK_N_STEPS))
        elif isinstance(asset, Option):
            option_price_min = 0.
            option_price_max = STOCK_PRICE_MAX - asset.strike
            option_price_num_steps = int(1+(option_price_max-option_price_min)/ASSET_PRICE_STEP_SIZE)
            price_grids.append(np.linspace(option_price_min, option_price_max, num=option_price_num_steps))
        else:
            assert isinstance(asset, Currency)
            price_grids.append(np.array([1.]))

    return [dict(zip(universe, prices)) for prices in itertools.product(*price_grids)]


@hydra.main(config_path=".", config_name="hydra_config")
def main(cfg: DictConfig):
    save_dir = os.getcwd()
    env = get_training_environment(cfg, save_dir)
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

    logging.info("Hyperparameter settings:")
    for k, v in cfg.items():
        logging.info(f"\t{k} = {v}")

    policy_kwargs = {"activation_fn": nn.ReLU, "net_arch": [32]*5}
    #PPO_HYPERPARAM_KEYS = ('learning_rate', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm')
    PPO_HYPERPARAM_KEYS = ('learning_rate', 'gamma', 'tau', 'train_freq', 'gradient_step', 'learning_starts')
    ppo_hyperparams_dict = {k: cfg[k] for k in PPO_HYPERPARAM_KEYS}
    #model = PPO(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, **ppo_hyperparams_dict)
    model = DDPG(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, **ppo_hyperparams_dict)
    logger_callback = LoggerCallback(save_path=os.path.join(save_dir, "rl_logs.json"), save_freq=1000)
    action_fn_observation_grid: List[Valuation] = get_observation_grid(env)
    action_fn_callback = ActionFunctionCallback(model, env, action_fn_observation_grid, save_path=os.path.join(save_dir, "action_fn_logs.h5"), save_freq=10_000)
    # checkpoint_callback = CheckpointCallback(save_freq=20_000, save_path=save_dir, name_prefix='model_checkpoint')
    # N_YEARS_TRAINING = 50_000
    # TOTAL_TRAINING_TIMESTEPS = N_YEARS_TRAINING*TRADING_DAYS_IN_YEAR
    model.learn(total_timesteps=cfg['total_training_timesteps'], callback=[logger_callback, action_fn_callback])

    with open(os.path.join(save_dir, "training_env.pkl"), "w+b") as f:
        pickle.dump(env, f)
    model.save(os.path.join(save_dir, "fully_trained_model"))


if __name__ == '__main__':
    main()