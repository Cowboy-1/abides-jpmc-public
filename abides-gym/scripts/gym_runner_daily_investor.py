import gymnasium as gym
from tqdm import tqdm

# Import to register environments
import abides_gym

if __name__ == "__main__":

    env = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
    )

    state, info = env.reset(seed=0)
    for i in tqdm(range(5)):
        state, reward, done, info = env.step(0)
