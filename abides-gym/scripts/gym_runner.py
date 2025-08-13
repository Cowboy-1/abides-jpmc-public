import gymnasium as gym
from tqdm import tqdm

# Import to register environments
import abides_gym

if __name__ == "__main__":

    env = gym.make(
        "markets-execution-v0",
        background_config="rmsc04",
    )

    obs, info = env.reset(seed=0)
    for i in tqdm(range(5)):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
