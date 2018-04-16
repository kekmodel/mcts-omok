from __future__ import print_function
from omok_env import OmokEnv
import time
import sys
from collections import deque, defaultdict
import numpy as np
from numpy import random, sqrt, argwhere, zeros

N, W, Q = 0, 1, 2
CURRENT = 0
OPPONENT = 1
COLOR = 2
BLACK = 1
WHITE = 0
BOARD_SIZE = 9
HISTORY = 2
N_SIMUL = 8000
GAME = 1


class MCTS:
    def __init__(self, board_size, n_history, n_simul):
        self.env_simul = OmokEnv(board_size, n_history, display=False)
        self.board_size = board_size
        self.n_simul = n_simul
        self.tree = None
        self.root = None
        self.state = None
        self.board = None
        self.legal_move = None
        self.no_legal_move = None

        # used for backup
        self.key_memory = None
        self.action_memory = None

        # init
        self._reset()
        self.reset_tree()

    def _reset(self):
        self.key_memory = deque(maxlen=self.board_size**2)
        self.action_memory = deque(maxlen=self.board_size**2)

    def reset_tree(self):
        self.tree = defaultdict(lambda: zeros((self.board_size**2, 3)))

    def get_action(self, state, board):
        self.root = state.copy()
        self._simulation(state)
        # init root board after simulatons
        self.board = board
        board_fill = self.board[CURRENT] + self.board[OPPONENT]
        self.legal_move = argwhere(board_fill == 0).flatten()
        self.no_legal_move = argwhere(board_fill != 0).flatten()
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
            n_selection = 0
            n_expansion = 0
            while not done:
                board_fill = self.board[CURRENT] + self.board[OPPONENT]
                self.legal_move = argwhere(board_fill == 0).flatten()
                self.no_legal_move = argwhere(board_fill != 0).flatten()
                key = hash(self.state.tostring())
                # search my tree
                if key in self.tree:
                    # selection
                    action = self._selection(key, c_pucb=5)
                    self.action_memory.appendleft(action)
                    self.key_memory.appendleft(key)
                    n_selection += 1
                elif n_expansion == 0:
                    # expansion
                    action = self._expansion(key)
                    self.action_memory.appendleft(action)
                    self.key_memory.appendleft(key)
                    n_expansion += 1
                else:
                    # rollout
                    action = random.choice(self.legal_move)
                self.state, self.board, reward, done = \
                    self.env_simul.step(action)
            if done:
                # backup & reset memory
                self._backup(reward, n_selection + n_expansion)
                self._reset()
                finish = time.time() - start
                # if finish >= self.think_time:
                #     break
        print('\r{} simulations end ({:0.0f}s)'.format(sim + 1, finish))

    def _selection(self, key, c_pucb):
        edges = self.tree[key]
        pucb = self._get_pucb(edges, c_pucb)

        if c_pucb == 0:
            visit = zeros(self.board_size**2)
            for i, edge in enumerate(edges):
                visit[i] = edge[N]
            print('')
            print('visit count')
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

    def _expansion(self, key):
        # only select once for rollout
        action = self._selection(key, c_pucb=5)
        return action

    def _get_pucb(self, edges, c_pucb):
        total_N = 0
        prior = 1/len(self.legal_move)
        pucb = zeros(self.board_size**2)
        for i in range(self.board_size**2):
            total_N += edges[i][N]
        # black's pucb
        if self.board[COLOR][0] == WHITE:
            for move in self.legal_move:
                pucb[move] = edges[move][Q] + \
                    c_pucb * prior * sqrt(total_N) / (edges[move][N] + 1)
            for move in self.no_legal_move:
                pucb[move] = -np.inf
        # white's pucb
        else:
            for move in self.legal_move:
                pucb[move] = edges[move][Q] - \
                    c_pucb * prior * sqrt(total_N) / (edges[move][N] + 1)
            for move in self.no_legal_move:
                pucb[move] = np.inf
        return pucb

    def _backup(self, reward, steps):
        # steps is n_selection + n_expansion
        # update edges in my tree
        for i in range(steps):
            edges = self.tree[self.key_memory[i]]
            action = self.action_memory[i]
            edges[action][N] += 1
            edges[action][W] += reward
            edges[action][Q] = edges[action][W] / edges[action][N]


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
    play()
