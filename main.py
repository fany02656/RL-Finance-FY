import pandas as pd
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from torch import nn
from tqdm import trange

from golds.contracts import Currency, Stock, Option, OptionFlavor, OptionStyle, Holdings
from golds.env import AmericanOptionEnv
from golds.mkt_data import PricingSource, SingleStockGBMMarketDataSource
from golds.params import GBMParams
from golds.reward_functions import NaiveHedgingRewardFunction
from golds.tcost import NaiveTransactionCostModel

REWARD_KAPPA = 100
INITIAL_WEALTH = 1e7
SELF_FINANCING_LAMBDA = 1000
TRADING_DAYS_IN_YEAR = 252


def main():
    aapl = Stock(ticker="AAPL", is_tradable=True)
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
        aapl: 0.,
        warrant: 100.,
        cash: INITIAL_WEALTH,
    }

    universe = list(initial_holdings.keys())

    gbm_params = GBMParams(mu=0.005, sigma=0.2, risk_free_rate=0.)

    mkt_data_source = SingleStockGBMMarketDataSource(universe, gbm_params, data_reuse_num_episodes=5*3000)
    tcost_model = NaiveTransactionCostModel(universe)
    pricing_source = PricingSource(mkt_data_source, tcost_model)

    env = AmericanOptionEnv(
        episode_length=TRADING_DAYS_IN_YEAR,
        pricing_source=pricing_source,
        reward_function=NaiveHedgingRewardFunction(kappa=REWARD_KAPPA, initial_holdings=initial_holdings),
        actions_config=list(range(-100, 101))
    )

    # TODO experiment with gamma, gae_lambda, ent_coef, vf_coef, max_grad_norm (kwargs to PPO.__init__)
    # TODO experiment with batch size (how to do this?)
    # TODO Lerrel says entropy related to exploration -- increase ent_coef if agent is not exploring enough
    # TODO experiment with different number of hidden nodes per layer in "net_arch" (64? 128? more?)
    policy_kwargs = {"activation_fn": nn.ReLU, "net_arch": [32]*5}
    model = PPO(MlpPolicy, env, verbose=1, learning_rate=1e-4, policy_kwargs=policy_kwargs)
    N_YEARS_TRAINING = 5_000_000
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/', name_prefix='hedging_model')
    model.learn(total_timesteps=TRADING_DAYS_IN_YEAR*N_YEARS_TRAINING, callback=checkpoint_callback)

    # TODO should log the training output here somehow (loss over time)
    # with open("american_option_env_5e6.pkl", "w+b") as f:
    #     pickle.dump(env, f)
    # model.save("trained_hedging_model_5e6")

    # with open("american_option_env.pkl", "rb") as f:
    #     env = pickle.load(f)
    # model = PPO.load("trained_hedging_model")

    NUM_YEARS_OUT_OF_SAMPLE = 10000
    print(f"Done training. Evaluating on {NUM_YEARS_OUT_OF_SAMPLE} years of data.")

    evaluation_histories = []
    reward_histories = []

    for i in trange(NUM_YEARS_OUT_OF_SAMPLE):
        evaluation_records = []
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

    evaluation_history.to_hdf("evaluation_history.h5", key="df", mode="w")
    reward_history.to_hdf("reward_history.h5", key="df", mode="w")

    '''
    evaluation_records = []
    obs = env.reset()
    for day in range(TRADING_DAYS_IN_YEAR):
        prices = env.observation_array_to_dict(obs)
        print(f"Observed prices: {prices}")
        action, _states = model.predict(obs)
        trade = env.action_array_to_dict(action)
        print(f"Executed trade: {trade}")
        obs, rewards, dones, info = env.step(action)
        print(f"Received reward: {rewards}")
        evaluation_records.append({"obs": prices, "action": trade, "reward": rewards})
        env.render()

    pd.DataFrame.from_records(evaluation_records).to_hdf("rl_results.h5", key="df", mode="w")

    env.reward_function.persist_history_to_hdf("reward_history.h5", key="df", mode="w")
    '''


if __name__ == '__main__':
    main()
