import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import gaussian_filter1d


class Grid:
    terrain_color = dict(normal=[127 / 360, 0, 96 / 100],
                         objective=[26 / 360, 100 / 100, 100 / 100],
                         cliff=[247 / 360, 92 / 100, 70 / 100],
                         player=[344 / 360, 93 / 100, 100 / 100])

    def __init__(self):
        self.player = None
        self._create_grid()
        self._draw_grid()

    @staticmethod
    def change_range(values, vmin=0, vmax=1):
        start_zero = values - np.min(values)
        return (start_zero / (np.max(start_zero) + 1e-7)) * (vmax - vmin) + vmin

    def _create_grid(self):
        self.grid = self.terrain_color['normal'] * np.ones((4, 12, 3))
        self._add_objectives(self.grid)

    def _add_objectives(self, grid):
        grid[-1, 1:11] = self.terrain_color['cliff']
        grid[-1, -1] = self.terrain_color['objective']

    def _draw_grid(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.ax.grid(which='minor')
        self.q_texts = [self.ax.text(*self.id_to_position(i)[::-1], '0',
                                     fontsize=11, verticalalignment='center',
                                     horizontalalignment='center') for i in range(12 * 4)]

        self.im = self.ax.imshow(hsv_to_rgb(self.grid), cmap='terrain',
                                 interpolation='nearest', vmin=0, vmax=1)
        self.ax.set_xticks(np.arange(12))
        self.ax.set_xticks(np.arange(12) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(4))
        self.ax.set_yticks(np.arange(4) - 0.5, minor=True)

    def reset(self):
        self.player = (3, 0)
        return self._position_to_id(self.player)

    def step(self, action):
        # Possible actions
        if action == 0 and self.player[0] > 0:
            self.player = (self.player[0] - 1, self.player[1])
        if action == 1 and self.player[0] < 3:
            self.player = (self.player[0] + 1, self.player[1])
        if action == 2 and self.player[1] < 11:
            self.player = (self.player[0], self.player[1] + 1)
        if action == 3 and self.player[1] > 0:
            self.player = (self.player[0], self.player[1] - 1)

        # Rules
        if all(self.grid[self.player] == self.terrain_color['cliff']):
            reward = -100
            done = True
        elif all(self.grid[self.player] == self.terrain_color['objective']):
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        return self._position_to_id(self.player), reward, done

    @staticmethod
    def _position_to_id(pos):
        # Maps a position in x,y coordinates to a unique ID
        return pos[0] * 12 + pos[1]

    @staticmethod
    def id_to_position(idx):
        return (idx // 12), (idx % 12)

    def update_grid(self, q_values=None, action=None, max_q=False, colorize_q=False):
        assert self.player is not None, 'You first need to call .reset()'

        if colorize_q:
            assert q_values is not None, 'q_values must not be None for using colorize_q'
            grid = self.terrain_color['normal'] * np.ones((4, 12, 3))
            values = self.change_range(np.max(q_values, -1)).reshape(4, 12)
            grid[:, :, 1] = values
            self._add_objectives(grid)
        else:
            grid = self.grid.copy()

        grid[self.player] = self.terrain_color['player']
        self.im.set_data(hsv_to_rgb(grid))

        if q_values is not None:
            np.repeat(np.arange(12), 4)
            np.tile(np.arange(4), 12)

            for i, text in enumerate(self.q_texts):
                if max_q:
                    q = max(q_values[i])
                    txt = '{:.2f}'.format(q)
                    text.set_text(txt)
                else:
                    actions = ['U', 'D', 'R', 'L']
                    txt = '\n'.join(['{}: {:.2f}'.format(k, q) for k, q in zip(actions, q_values[i])])
                    text.set_text(txt)

        if action is not None:
            self.ax.set_title(action, color='r', weight='bold', fontsize=32)


class Agent:
    def __init__(self, exp_rate=0.1, sarsa=True):
        self.env = Grid()
        self.actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
        self.num_states = 4 * 12
        self.num_actions = 4
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.exp_rate = exp_rate
        self.sarsa = sarsa

    def egreedy_policy(self, state):
        # A random action is selected with epsilon probability, else select the best action.
        if np.random.random() < self.exp_rate:
            return np.random.choice(4)
        else:
            return np.argmax(self.q_values[state])

    def sarsa_or_q_learning(self, episodes=500, render=True,
                            learning_rate=0.1, discount_rate=1, sarsa=True):
        self.q_values = np.zeros((self.num_states, self.num_actions))
        ep_rewards = []

        for _ in range(episodes):
            state = self.env.reset()

            # Display current state
            print("Episode start- current state", self.env.id_to_position(state))

            done = False
            reward_sum = 0

            if sarsa:
                # Choose A from S using policy derived from Q (e-greedy)
                action = self.egreedy_policy(state)

            while not done:
                if not sarsa:
                    # Choose A from S using policy derived from Q (e-greedy)
                    action = self.egreedy_policy(state)

                # Do the action
                next_state, reward, done = self.env.step(action)
                reward_sum += reward

                if sarsa:
                    # Choose next action
                    next_action = self.egreedy_policy(state=next_state)

                    # Next q value is the value of the next action
                    td_target = reward + discount_rate * self.q_values[next_state][next_action]
                    td_error = td_target - self.q_values[state][action]
                    # Update q value
                    self.q_values[state][action] += learning_rate * td_error
                    # Update state and action
                    state = next_state
                    action = next_action
                else:
                    # Update q_values
                    td_target = reward + discount_rate * np.max(self.q_values[next_state])
                    td_error = td_target - self.q_values[state][action]
                    self.q_values[state][action] += learning_rate * td_error
                    # Update state
                    state = next_state

                # Display next state
                print("moved to state", self.env.id_to_position(next_state))

                # Uncomment for displaying the grid in pycharm!!!
                # self.env._draw_grid()

                if render:
                    self.env.update_grid(q_values=self.q_values, action=self.actions[action], colorize_q=True)

            ep_rewards.append(reward_sum)

        return ep_rewards, self.q_values


if __name__ == "__main__":
    exploration_rate = 0.5
    ag_s = Agent(exp_rate=exploration_rate, sarsa=True)
    ag_q = Agent(exp_rate=exploration_rate, sarsa=False)

    # Q-learning
    _, ag_q.q_values = ag_q.sarsa_or_q_learning(render=True, sarsa=False)
    ag_q.env.update_grid(ag_q.q_values, colorize_q=True)
    q_learning_rewards, _ = zip(*[ag_q.sarsa_or_q_learning(render=False, sarsa=False) for _ in range(10)])

    # Sarsa
    _, ag_s.q_values = ag_s.sarsa_or_q_learning(render=True, sarsa=True)
    sarsa_rewards, _ = zip(*[ag_s.sarsa_or_q_learning(render=False, sarsa=True) for _ in range(10)])

    # Plot graph
    avg_q_learning_rewards = np.mean(q_learning_rewards, axis=0)
    avg_sarsa_rewards = np.mean(sarsa_rewards, axis=0)
    fig, ax = plt.subplots()
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Sum of rewards during episode')
    ax.plot(gaussian_filter1d(avg_q_learning_rewards, sigma=6), label="Q-learning")
    ax.plot(gaussian_filter1d(avg_sarsa_rewards, sigma=6), label="Sarsa")
    ax.legend()
    plt.show()

    # Mean reward
    mean_reward_q = [np.mean(avg_q_learning_rewards)] * len(avg_q_learning_rewards)
    mean_reward_sarsa = [np.mean(avg_sarsa_rewards)] * len(avg_sarsa_rewards)
    print('Q-learning Mean Reward: {}'.format(mean_reward_q[0]))
    print('Sarsa Mean Reward: {}'.format(mean_reward_sarsa[0]))
