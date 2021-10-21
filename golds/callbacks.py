import logging; logging.basicConfig(level=logging.INFO)

import gc
import json
import numpy as np
import pandas as pd

from typing import Any, List

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger

from .contracts import Valuation
from .env import AmericanOptionEnv


def _json_encode_numpy(val: Any) -> Any:
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()

    raise TypeError(f"Cannot JSON serialize object {val} of type {type(val)}")


class LoggerCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq: int, verbose: int = 0):
        super(LoggerCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            with open(self.save_path, 'a') as f:
                json.dump(logger.get_log_dict(), f, default=_json_encode_numpy)
                f.write('\n')

        return True


class ActionFunctionCallback(BaseCallback):
    def __init__(self, agent: PPO, env: AmericanOptionEnv, observation_grid: List[Valuation], save_path: str, save_freq: int, verbose: int = 0):
        super(ActionFunctionCallback, self).__init__()
        self.agent = agent
        self.universe = env.pricing_source.universe
        self.save_path = save_path
        self.save_freq = save_freq

        # Save observation_grid to file
        start = pd.Timestamp.now()

        observation_grid_df: pd.DataFrame = pd.DataFrame.from_records(observation_grid)
        observation_grid_df.to_hdf(save_path, mode='w', key='observation_grid', format='table')

        del observation_grid_df
        gc.collect()

        logging.info(f"Took {pd.Timestamp.now()-start} to construct and persist observation_grid_df to HDF5")

        # Store observation_grid as list of arrays, as this will be what is passed into the agent
        self.observation_array_grid = [env.observation_dict_to_array(obs) for obs in observation_grid]
        assert all(len(obs) == len(self.universe) for obs in self.observation_array_grid)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            start = pd.Timestamp.now()

            action_fn_records: List[dict] = []
            for obs in self.observation_array_grid:
                action_fn_record = {'timestep': self.n_calls}
                for asset, price in zip(self.universe, obs):
                    action_fn_record[asset] = price

                action, _states = self.agent.predict(obs)
                action_fn_record['action'] = json.dumps(_json_encode_numpy(action))

                action_fn_records.append(action_fn_record)

            action_fn_df: pd.DataFrame = pd.DataFrame.from_records(action_fn_records)
            action_fn_df.to_hdf(self.save_path, mode='a', key='action_fn_logs', append=True, format='table')

            del action_fn_df
            gc.collect()

            logging.info(f"ActionFunctionCallback took {pd.Timestamp.now()-start} to evaluate action function on observation grid")

        return True