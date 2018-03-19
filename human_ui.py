from omok_env_history import OmokEnv
from mcts_uct import MCTS
import numpy as np

N, Q = 0, 1
CURRENT = 0
OPPONENT = 1
COLOR = 2
BLACK = 1
WHITE = 0
BOARD_SIZE = 15
COLUMN = {"a":  0, "b":  1, "c":  2,
          "d":  3, "e":  4, "f":  5,
          "g":  6, "h":  7, "i":  8,
          "j":  9, "k": 10, "l": 11,
          "m": 12, "n": 13, "o": 14,
          "A":  0, "B":  1, "C":  2,
          "D":  3, "E":  4, "F":  5,
          "G":  6, "H":  7, "I":  8,
          "J":  9, "K": 10, "L": 11,
          "M": 12, "N": 13, "O": 14}

SIMULATIONS = 30000
GAMES = 1


class HumanAgent:
    def get_action(self):
        move_target = str(input('1a ~ 15: '))
        row = int(move_target[:1]) - 1
        col = COLUMN[move_target[1:2]]
        action = row * BOARD_SIZE + col
        return action


class HumanUI:
    def __init__(self):
        self.human = HumanAgent()
        self.ai = MCTS(SIMULATIONS)

    def get_action(self, state, board, idx):
        if idx % 2 == 0:
            action = self.human.get_action()
        else:
            action = self.ai.get_action(state, board)
        return action


if __name__ == '__main__':
    env = OmokEnv()
    manager = HumanUI()
    result = {-1: 0, 0: 0, 1: 0}
    for game in range(GAMES):
        print('##########   Game: {}   ##########'.format(game + 1))
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
        print("=" * 20, " {}  Game End  ".format(game + 1), "=" * 20)
        stat_game = ('Black Win: {}  White Win: {}  Draw: {}  Winrate: {:0.1f}%'.format(
            result['Black'], result['White'], result['Draw'],
            1 / (1 + np.exp(result['White'] / (game + 1)) /
                 np.exp(result['Black'] / (game + 1))) * 100))
        print(stat_game, '\n')
