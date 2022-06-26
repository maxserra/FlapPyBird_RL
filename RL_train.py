import pickle
from datetime import datetime
import numpy as np

import RL_flappy
from RL_agents import Flappy_QAgent, FlappyNNAgent

AGENT_BACKUP_FNAME = "flappy_agent_backup.dat"

AGENT_BACKUP_BEST_FNAME = "flappy_best_agent_backup.dat"


def train(n_episodes):

    try:
        with open(AGENT_BACKUP_FNAME, "rb") as f:
            flappy_agent: Flappy_QAgent = pickle.load(f)

        print(f"""Recovered agent from {AGENT_BACKUP_FNAME}\n""" +
              f"""Agent knows {len(flappy_agent._Q_table)} states and is {flappy_agent.epsisode_age} episodes old.""" +
              f"""{flappy_agent._Q_table}""")

    except:
        print("""Starting with a new fresh agent...""")

        flappy_agent = Flappy_QAgent(alpha_start=0.2, alpha_end=0.1, alpha_decay_episodes=20000,
                                     gamma=1, epsilon=0.0)

    flappy_env = RL_flappy.Flappy_Environment(step_reward=0, score_reward=100, die_reward=-1000)

    temp_score = []

    for episode in range(1, n_episodes):

        env_state = flappy_env.set_up()

        flappy_agent.update_state(player_pos=env_state["playerPos"],
                                  player_vel=env_state["playerVelY"],
                                  lower_pipes=env_state["lowerPipes"])

        while not env_state["crashInfo"][0]:

            flappy_agent.choose_action(state=flappy_agent.curr_state)

            env_reward, env_state = flappy_env.take_action(action=flappy_agent.action)

            flappy_agent.update_state(player_pos=env_state["playerPos"],
                                      player_vel=env_state["playerVelY"],
                                      lower_pipes=env_state["lowerPipes"])

            flappy_agent.update_Q(reward=env_reward)

        flappy_agent.epsisode_age += 1
        flappy_agent.best_score = max(flappy_agent.best_score, env_state["score"])

        if env_state["score"] == flappy_agent.best_score:
            with open(AGENT_BACKUP_BEST_FNAME, "wb") as f:
                pickle.dump(flappy_agent, f)

        temp_score.append(env_state["score"])

        if (episode) % 50 == 0:
            print(f"------------\nEPISODE {episode} of {n_episodes}")
            print(f"""{datetime.now().strftime("%H:%M:%S.%f")} | """ +
                  f"""mean score: {np.mean(temp_score):.1f} ({np.min(temp_score)}, {np.max(temp_score)})of best: {flappy_agent.best_score}, final state: {flappy_agent.curr_state}""")
            temp_score = []

            # save agent to pickle file
            print(f"""Saving agent to {AGENT_BACKUP_FNAME} file.\n""" +
                  f"""Agent knows {len(flappy_agent._Q_table)} states and is {flappy_agent.epsisode_age} episodes old.""")
            with open(AGENT_BACKUP_FNAME, "wb") as f:
                pickle.dump(flappy_agent, f)


def play(autoplay=False, autoplay_best=False):

    if autoplay:
        try:
            backup = AGENT_BACKUP_BEST_FNAME if autoplay_best else AGENT_BACKUP_FNAME

            with open(backup, "rb") as f:
                flappy_agent: Flappy_QAgent = pickle.load(f)

            print(f"""Recovered agent from {backup}\n""" +
                  f"""Agent knows {len(flappy_agent._Q_table)} states and is {flappy_agent.epsisode_age} episodes old.\n""" +
                  f"""It has a best score of {flappy_agent.best_score}""")
            # print(f"""{flappy_agent._Q_table}""")

        except:
            print("""Starting with a new fresh agent...""")

            flappy_agent = Flappy_QAgent(alpha_start=0.1, alpha_end=0.1, alpha_decay_episodes=20000,
                                         gamma=1, epsilon=0.0)
    else:
        flappy_agent = None

    RL_flappy.main(QAgent=flappy_agent)


def crossover_and_mutate_agents(agents, agents_pval, n_specimens):

    new_agents = agents.tolist()

    while len(new_agents) < n_specimens:

        selected_agents = np.random.multinomial(n=1, pvals=agents_pval, size=2)

        new_agent = FlappyNNAgent()
        new_agent.crossover_and_mutate(agent_a=agents[np.where(selected_agents[0])][0],
                                       agent_b=agents[np.where(selected_agents[0])][0])

        new_agents.append(new_agent)

    return new_agents


def train_NN_agent(n_specimens, n_generations, selection_rate=0.4):

    flappy_env = RL_flappy.Flappy_Environment(step_reward=1, score_reward=100, die_reward=-1000)

    flappy_agents = []
    for _ in range(n_specimens):
        flappy_agents.append(FlappyNNAgent())

    for generation in range(n_generations):

        temp_agent_rewards = []
        temp_agent_scores = []

        for i in range(n_specimens):

            agent = flappy_agents[i]

            env_state = flappy_env.set_up()
            temp_rewards = 0

            while not env_state["crashInfo"][0]:

                agent.choose_action(state=env_state)

                env_reward, env_state = flappy_env.take_action(action=agent.action)

                temp_rewards += env_reward

            temp_agent_rewards.append(temp_rewards)
            temp_agent_scores.append(env_state['score'])

        print(f"Best agent (out of '{len(flappy_agents)}') with reward of {np.max(temp_agent_rewards)} scored {np.max(temp_agent_scores)}")

        best_agents = np.array(flappy_agents)[np.argsort(temp_agent_rewards)[-int(selection_rate * n_specimens):]]
        best_agents_rewards = np.array(temp_agent_rewards)[np.argsort(temp_agent_rewards)[-int(selection_rate * n_specimens):]]

        flappy_agents = crossover_and_mutate_agents(agents=best_agents,
                                                    agents_pval=best_agents_rewards / best_agents_rewards.sum(),
                                                    n_specimens=n_specimens)


if __name__ == '__main__':

    # train(n_episodes=30000)
    # play(autoplay=True, autoplay_best=False)

    train_NN_agent(n_specimens=80, n_generations=10000)
