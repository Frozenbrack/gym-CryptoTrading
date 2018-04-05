import os
import gym
import numpy as np
from .modules import poloniexdb
import pyximport; pyximport.install(setup_args={'include_dirs': [np.get_include(), poloniexdb]})
from .modules import sampler 


default_config = {"pairs": ["USDT_BTC"],
                  "fee": 0.025,
                  "size": 10000,
                  "period": 600,
                  "timeSteps": 20,
                  "wavelet_filtering": False, # or {"wavelet": "haar", "level": 2},
                  "variables": ["weightedAverage", "high", "low", "open", "close", "volume"],
                  "indicators": [["MA",5,"weightedAverage"],["MA",50,"weightedAverage"]],
                  "stake_main": 1.0,
                  "make_or_take": "take",
                  "stakes_pivot": [0.0]}

class CryptoTradingEnv(gym.Env):
    def __init__(self, config = default_config):
        if os.path.isdir('databases') == False:
            os.mkdir('databases')
        self.generate_environment(config)
      
        
    def step(self, action):
        done = self.take_action(action)
        observation = []
        for i in range(self.pair_num):
            observation.append(self.observation_space[i][self.t])
        reward = self.get_reward()
        info = {}
        return observation, reward, done, info

    
    def get_price(self, index):
        if self.make_or_take == "take":
            return self.prices[index][self.t]
        else:
            ##To add: order placement
            pass
        
        
    def take_action(self, action):
        action_index = np.argmax(action)
        if action_index == self.pair_num and self.stakes_pivot[action_index] > 0.00000001: # sell
            self.stake_main = self.stakes_pivot[action_index]*self.get_price(action_index)*(1.0 - self.fee)
            self.stakes_pivot[action_index] = 0.0
        elif action_index < self.pair_num and self.stake_main > 0.00000001: # buy
            self.stakes_pivot[action_index] = self.stake_main/self.get_price(action_index)*(1.0 - self.fee)
            self.stake_main = 0.0
        else:
            pass
        self.t += 1 
        if self.stake_main + sum(self.stakes_pivot) <= 0.00000001 or self.t + 1 == self.t_terminal:
            return True
        else:
            return False
            
            
    def get_reward(self):
        total = 0.0
        for i in range(self.pair_num):
            total += self.get_price(i)*self.stakes_pivot[i]
        return self.stake_main + total
    
    
    def generate_environment(self, config):
        self.config = config
        self.pair_num = len(config["pairs"])
        assert self.pair_num == len(config["stakes_pivot"])
        self.t = 0
        self.t_terminal = config["size"]
        self.stake_main = config["stake_main"]
        self.stakes_pivot = config["stakes_pivot"]
        self.fee = config["fee"]
        self.observation_space = []
        self.action_space = np.identity(self.pair_num + 2)
        self.prices = []
        self.make_or_take = config["make_or_take"]
        poloniexdb.makeDB()
        for pair in config["pairs"]:
            poloniexdb.addPairToDB(pair)
            poloniexdb.updatePair(pair)
            if config["wavelet_filtering"] == False:
                self.observation_space.append(sampler.sample(config["size"],
                                                             pair, 
                                                             config["period"],
                                                             config["timeSteps"],
                                                             config["variables"], 
                                                             config["indicators"],
                                                             "None",
                                                             0))
            else:
                self.observation_space.append(sampler.sample(config["size"],
                                                             pair, 
                                                             config["period"],
                                                             config["timeSteps"],
                                                             config["variables"], 
                                                             config["indicators"],
                                                             config["wavelet_filtering"]["wavelet"],
                                                             config["wavelet_filtering"]["level"]))
            if config["make_or_take"] == "take":
                self.prices.append(sampler.sample(config["size"],
                                                  pair, 
                                                  config["period"],
                                                  1,
                                                  ["close"], 
                                                  [],
                                                  "None",
                                                  0))
                
                
    def reset(self):
        self.t = 0
        self.stake_main = self.config["stake_main"]
        self.stakes_pivot = self.config["stakes_pivot"]


