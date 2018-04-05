from gym.envs.registration import register

register(
    id='CryptoTrading-v0',
    entry_point='gym_CryptoTrading.envs:CryptoTradingEnv',
)
