from __future__ import print_function
from omok_env import OmokEnv
import time
import sys
from collections import deque, defaultdict
import numpy as np
from numpy import random, sqrt, argwhere, zeros

N, Q = 0, 1
CURRENT = 0
OPPONENT = 1
COLOR = 2
BLACK = 1
WHITE = 0
BOARD_SIZE = 9
HISTORY = 2
N_SIMUL = 1600
GAME = 30


class MCTS:
    def __init__(self, board_size, n_history, n_simul):
        self.env_simul = OmokEnv(board_size, n_history, display=False)
        self.board_size = board_size
        self.n_simul = n_simul
        self.tree = None
        self.root = None
        self.state = None
        self.board = None
        # used for backup
        self.key_memory = deque()
        self.action_memory = deque()
        self.reset_tree()

    def reset_tree(self):
        self.tree = defaultdict(lambda: zeros((self.board_size**2, 2)))

    def get_action(self, state, board):
        self.root = state.copy()
        self._simulation(state)
        # init root board after simulatons
        self.board = board
        board_fill = self.board[CURRENT] + self.board[OPPONENT]
        self.legal_move = argwhere(board_fill == 0).flatten()
        # root state's key
        root_key = hash(self.root.tostring())
        # argmax Q or argmin Q
        action = self._selection(root_key, c_pucb=0)
        return action

    def _simulation(self, state):
        start = time.time()
        finish = 0
        for sim in range(self.n_simul):
            print('\rsimulation: {}'.format(sim + 1), end='')
            sys.stdout.flush()
            # reset state
            self.state, self.board = self.env_simul.reset(state)
            done = False
            is_expansion = True

            while not done:
                key = hash(self.state.tostring())
                # search my tree
                if key in self.tree:
                    # selection
                    action = self._selection(key, c_pucb=5)
                    self.action_memory.appendleft(action)
                    self.key_memory.appendleft(key)
                else:
                    # expansion
                    legal_move = self._get_legal_move(self.board)
                    action = random.choice(legal_move)
                    if is_expansion:
                        self.action_memory.appendleft(action)
                        self.key_memory.appendleft(key)
                        is_expansion = False

                self.state, self.board, reward, done = self.env_simul.step(action)

            if done:
                # backup & reset memory
                self._backup(reward)
                finish = time.time() - start
                # if finish >= self.think_time:
                #     break
        print('\r{} simulations end ({:0.0f}s)'.format(sim + 1, finish))

    def _get_legal_move(self, board):
        board_fill = board[CURRENT] + board[OPPONENT]
        legal_move = argwhere(board_fill == 0).flatten()
        return legal_move

    def _selection(self, key, c_pucb):
        edges = self.tree[key]
        pucb = self._get_pucb(edges, c_pucb)

        if c_pucb == 0:
            visit = edges[:, N]
            print('\nvisit count')
            print(visit.reshape(self.board_size, self.board_size).round())
            action = argwhere(visit == visit.max()).flatten()
            action = action[random.choice(len(action))]
            return action

        if self.board[COLOR][0] == WHITE:
            # black's choice
            action = argwhere(pucb == pucb.max()).flatten()
        else:
            # white's choice
            action = argwhere(pucb == pucb.min()).flatten()
        action = action[random.choice(len(action))]
        return action

    def _get_pucb(self, edges, c_pucb):
        legal_move = self._get_legal_move(self.board)
        prior = 1/len(legal_move)
        total_N = edges.sum(0)[N]
        # black's pucb
        if self.board[COLOR][0] == WHITE:
            pucb = zeros(self.board_size**2) - np.inf
            for move in legal_move:
                pucb[move] = edges[move][Q] + \
                    c_pucb * prior * sqrt(total_N) / (edges[move][N] + 1)
        # white's pucb
        else:
            pucb = zeros(self.board_size**2) + np.inf
            for move in legal_move:
                pucb[move] = edges[move][Q] - \
                    c_pucb * prior * sqrt(total_N) / (edges[move][N] + 1)
        return pucb

    def _backup(self, reward):
        # update edges in my tree
        while self.action_memory:
            key = self.key_memory.popleft()
            action = self.action_memory.popleft()
            edges = self.tree[key]
            edges[action][N] += 1
            edges[action][Q] += (reward - edges[action][Q]) / edges[action][N]


def play():
    env = OmokEnv(BOARD_SIZE, HISTORY)
    mcts = MCTS(BOARD_SIZE, HISTORY, N_SIMUL)
    result = {'Black': 0, 'White': 0, 'Draw': 0}
    for g in range(GAME):
        print('#' * (BOARD_SIZE - 4),
              ' GAME: {} '.format(g + 1),
              '#' * (BOARD_SIZE - 4))
        # reset state
        state, board = env.reset()
        done = False
        while not done:
            env.render()
            # start simulations
            action = mcts.get_action(state, board)
            state, board, z, done = env.step(action)
        if done:
            if z == 1:
                result['Black'] += 1
            elif z == -1:
                result['White'] += 1
            else:
                result['Draw'] += 1
            # render & reset tree
            env.render()
            mcts.reset_tree()
        # result
        print('')
        print('=' * 20, " {}  Game End  ".format(g + 1), '=' * 20)
        blw, whw, drw = result['Black'], result['White'], result['Draw']
        stat = (
            'Black Win: {}  White Win: {}  Draw: {}  Winrate: {:0.1f}%'.format(
                blw, whw, drw,
                1 / (1 + np.exp(whw / (g + 1)) / np.exp(blw / (g + 1))) * 100))
        print(stat, '\n')


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    play()
