from collections import deque
import numpy as np

CURRENT = 0
OPPONENT = 1
COLOR = 2
BLACK = 1
WHITE = 0
COLOR_DICT = {1: 'Black', 0: 'White'}
BOARD_SIZE = 15


class OmokEnv:
    def __init__(self):
        self.state = None
        self.board = None
        self.history = None
        self.done = None

    def reset(self, state=None):
        if state is None:  # initialize state
            self.state = np.zeros((17 * BOARD_SIZE**2), 'int8')
            self.history = deque([np.zeros((BOARD_SIZE**2), 'int8')] * 16, maxlen=16)
            self.board = np.zeros((3, BOARD_SIZE**2), 'int8')
        else:  # pass the state to the simulation's root
            self.state = state.copy()
            state_origin = self.state.reshape(17, BOARD_SIZE**2)
            self.history = deque([state_origin[i] for i in range(16)], maxlen=16)
            self.board = np.zeros((3, BOARD_SIZE**2), 'int8')
            self.board[CURRENT] = state_origin[1]
            self.board[OPPONENT] = state_origin[0]
            self.board[COLOR] = state_origin[16]
        return self.state, self.board

    def step(self, action):
        # board
        state_origin = self.state.reshape(17, BOARD_SIZE**2)
        self.board = np.zeros((3, BOARD_SIZE**2), 'int8')
        self.board[CURRENT] = state_origin[1]
        self.board[OPPONENT] = state_origin[0]
        self.board[COLOR] = state_origin[16]
        self.board_fill = (self.board[CURRENT] + self.board[OPPONENT])
        if self.board_fill[action] == 1:
            raise ValueError("No Legal Move!")

        # action
        self.board[CURRENT][action] = 1
        self.history.appendleft(self.board[CURRENT])
        self.board[COLOR] = abs(self.board[COLOR] - 1)
        self.state = np.r_[np.asarray(self.history).flatten(),
                           np.asarray(self.board[COLOR]).flatten()]
        return self._check_win(self.board[CURRENT].reshape(BOARD_SIZE, BOARD_SIZE))

    def render(self):
        if self.board[COLOR][0] == BLACK:
            board = (self.board[CURRENT] + self.board[OPPONENT] * 2).reshape(
                BOARD_SIZE, BOARD_SIZE)
        else:
            board = (self.board[CURRENT] * 2 + self.board[OPPONENT]).reshape(
                BOARD_SIZE, BOARD_SIZE)
        count = np.sum(self.board[CURRENT] + self.board[OPPONENT])
        board_str = '\n   A B C D E F G H I J K L M N O\n'
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if j == 0:
                    board_str += '{:2}'.format(i + 1)
                if board[i][j] == 0:
                    board_str += ' .'
                if board[i][j] == 1:
                    board_str += ' O'
                if board[i][j] == 2:
                    board_str += ' X'
                if j == BOARD_SIZE - 1:
                    board_str += ' \n'
            if i == BOARD_SIZE - 1:
                board_str += '   --------  MOVE: {}  ---------'.format(count)
        print(board_str)

    def _check_win(self, board):
        current_grid = np.zeros((5, 5))
        for row in range(BOARD_SIZE - 5 + 1):
            for col in range(BOARD_SIZE - 5 + 1):
                current_grid = board[row: row + 5, col: col + 5]
                sum_horizontal = np.sum(current_grid, axis=1)
                sum_vertical = np.sum(current_grid, axis=0)
                sum_diagonal_1 = np.sum(current_grid.diagonal())
                sum_diagonal_2 = np.sum(np.flipud(current_grid).diagonal())
                if 5 in sum_horizontal or 5 in sum_vertical:
                    done = True
                    color = self.board[COLOR][0]
                    if color == BLACK:
                        reward = 1
                    else:
                        reward = -1
                    print('#########   {} Win!  #########'.format(COLOR_DICT[color]))
                    return self.state, self.board, reward, done
                if sum_diagonal_1 == 5 or sum_diagonal_2 == 5:
                    reward = 1
                    done = True
                    color = self.board[COLOR][0]
                    if color == BLACK:
                        reward = 1
                    else:
                        reward = -1
                    print('#########   {} Win!  #########'.format(COLOR_DICT[color]))
                    return self.state, self.board, reward, done
        if np.sum(self.board_fill) == BOARD_SIZE**2 - 1:
            reward = 0
            done = True
            print('#########     Draw!    #########')
            return self.state, self.board, reward, done
        else:  # continue
            reward = 0
            done = False
            return self.state, self.board, reward, done
    __check_win = _check_win


class OmokEnvSimul(OmokEnv):
    def _check_win(self, board):
        current_grid = np.zeros((5, 5))
        for row in range(BOARD_SIZE - 5 + 1):
            for col in range(BOARD_SIZE - 5 + 1):
                current_grid = board[row: row + 5, col: col + 5]
                sum_horizontal = np.sum(current_grid, axis=1)
                sum_vertical = np.sum(current_grid, axis=0)
                sum_diagonal_1 = np.sum(current_grid.diagonal())
                sum_diagonal_2 = np.sum(np.flipud(current_grid).diagonal())
                if 5 in sum_horizontal or 5 in sum_vertical:
                    done = True
                    color = self.board[COLOR][0]
                    if color == BLACK:
                        reward = 1
                    else:
                        reward = -1
                    return self.state, self.board, reward, done
                if sum_diagonal_1 == 5 or sum_diagonal_2 == 5:
                    reward = 1
                    done = True
                    color = self.board[COLOR][0]
                    if color == BLACK:
                        reward = 1
                    else:
                        reward = -1
                    return self.state, self.board, reward, done
        if np.sum(self.board_fill) == BOARD_SIZE**2 - 1:
            reward = 0
            done = True
            return self.state, self.board, reward, done
        else:  # continue
            reward = 0
            done = False
            return self.state, self.board, reward, done

if __name__ == '__main__':
    env = OmokEnv()
    env.reset()
    env.render()
