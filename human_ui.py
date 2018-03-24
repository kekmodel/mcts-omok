from __future__ import print_function
from omok_env import OmokEnv
from mcts_uct import MCTS
import numpy as np
from numba import jit

N, Q = 0, 1
CURRENT = 0
OPPONENT = 1
COLOR = 2
BLACK = 1
WHITE = 0
BOARD_SIZE = 9
HISTORY = 4
COLUMN = {"a": 0, "b": 1, "c": 2,
          "d": 3, "e": 4, "f": 5,
          "g": 6, "h": 7, "i": 8,
          "j": 9, "k": 10, "l": 11,
          "m": 12, "n": 13, "o": 14,
          "A": 0, "B": 1, "C": 2,
          "D": 3, "E": 4, "F": 5,
          "G": 6, "H": 7, "I": 8,
          "J": 9, "K": 10, "L": 11,
          "M": 12, "N": 13, "O": 14}

SIMULATION = BOARD_SIZE**2 * 30
GAME = 1

@jit
class HumanAgent:
    def get_action(self):
        laskt_str = str(BOARD_SIZE)
        for k, v in COLUMN.items():
            if v == BOARD_SIZE - 1:
                laskt_str += k
                break
        move_target = str(input('1a ~ {}: '.format(laskt_str)))
        row = int(move_target[:1]) - 1
        col = COLUMN[move_target[1:2]]
        action = row * BOARD_SIZE + col
        return action


class HumanUI:
    def __init__(self):
        self.human = HumanAgent()
        self.ai = MCTS(SIMULATION, BOARD_SIZE, HISTORY)

    def get_action(self, state, board, idx):
        if idx % 2 == 0:
            action = self.human.get_action()
        else:
            action = self.ai.get_action(state, board)
        return action

@jit
def play():
    env = OmokEnv(BOARD_SIZE, HISTORY)
    manager = HumanUI()
    result = {-1: 0, 0: 0, 1: 0}
    z = 0
    g = 0
    idx = 0
    for g in range(GAME):
        print('##########   Game: {}   ##########'.format(g + 1))
        state, board = env.reset()
        done = False
        idx = 0
        while not done:
            env.render()
            # start simulations
            action = manager.get_action(state, board, idx)
            state, board, z, done = env.step(action)
            idx += 1
        if done:
            if z == 1:
                result['Black'] += 1
            elif z == -1:
                result['White'] += 1
            else:
                result['Draw'] += 1
            # render & reset tree
            env.render()
            manager.ai.reset_tree()
        # result
        print('')
        print('=' * 20, " {}  Game End  ".format(g + 1), '=' * 20)
        blw, whw, drw = result['Black'], result['White'], result['Draw']
        stat = ('Black Win: {}  White Win: {}  Draw: {}  Winrate: {:0.1f}%'.format(
            blw, whw, drw, 1 / (1 + np.exp(whw / (g + 1)) / np.exp(blw / (g + 1))) * 100))
        print(stat, '\n')


if __name__ == '__main__':
    play()
