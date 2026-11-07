import argparse

import ale_py
import gymnasium as gym
import os
from utils import play_human, play_agent

ENV_ID = "ALE/Pong-v5"
gym.register_envs(ale_py)



def play_random(env_id):
    """Standard random agent plays Pong against Atari's built-in opponent."""

    # env = gym.make(env_id, render_mode="rgb_array", frameskip=4)
    env = gym.make(env_id, render_mode="human", frameskip=4)

    for episode_counter in range(10):
        
        # cada nuevo episdio se reinicia el entorno
        episode_steps = 0
        episode_reward = 0
        obs = env.reset()
        
        while True:
            action = env.action_space.sample()
            obs, rewards, terminated, truncated, infos = env.step(action)
            episode_reward += rewards
            env.render()
            episode_steps += 1
            
            if terminated or truncated:
                print(f"[{episode_counter}]Episode finished. Reward: {episode_reward:.2f} (steps {episode_steps})")
                episode_reward = 0
                break
                
    env.close()
# ============================================================
# Main
# ============================================================
def main():

    final_model = "models/PPO/final_model"
    best_model = "models/PPO/best_model/best_model"
    default_model = final_model if os.path.exists(final_model + ".zip") else best_model

    parser = argparse.ArgumentParser(description="Play Atari Pong as a human or using a PPO agent.")
    parser.add_argument("--mode", choices=["human", "agent","random"], default="agent", help="Who controls the paddle.")
    # (2) añadimos la selección del algoritmo en la ejecución de play.py
    parser.add_argument("--algorithm", choices=["PPO", "DQN", "A2C"], default="PPO", help="Algorithm used by the trained agent.")
    # (1) hasta ahora el modo de juego "agent" usaba siempre el best_model, pero el score que conseguía este modelo podía deberse a puro azar, por lo que 
    # hemos decidido utilizar final_model (si existe, es decir, si el entrenamiento ha terminado) porque, aunque pueda tener un score peor, es un modelo entrenado
    # durante más iteraciones, y por lo tanto, más fiable
    parser.add_argument("--model", type=str, default=default_model, help="Path to the PPO model.")
    args = parser.parse_args()

    # (2) cambiamos las rutas a los modelos dependiendo del algoritmo que hayamos seleccionado
    if "models/PPO" in args.model and args.algorithm != "PPO":
        final_model = f"models/{args.algorithm}/final_model"
        best_model = f"models/{args.algorithm}/best_model/best_model"
        args.model = (
            final_model
            if os.path.exists(final_model + ".zip")
            else best_model
        )

    if args.mode == "human":
        play_human(ENV_ID)
    elif args.mode == "agent":
        play_agent(args.model, ENV_ID, args.algorithm)
    else:
        play_random(ENV_ID)


if __name__ == "__main__":
    main()
