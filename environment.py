from agent import BaseAgent
from typing import List, Tuple

from collections import defaultdict
import numpy as np
from numpy import random

from agent import BaseAgent


class Environment:
    def __init__(self, agents: List[BaseAgent]):
        self.agent1, self.agent2 = agents
        self.board = np.zeros((3, 3), dtype=int)
        self.num_actions = 0
        self.is_over = False

    def get_board_hash(self):
        state = self.board.ravel().astype(str).tolist()
        state = "".join(state)
        return state

    def _check_if_game_is_over(self) -> Tuple[bool]:
        diagonal = self.board.diagonal()
        anti_diag = np.fliplr(self.board).diagonal()
        combinations_sum = (
            np.abs(
                np.r_[
                    diagonal.sum(),
                    anti_diag.sum(),
                    np.sum(self.board, axis=0),
                    np.sum(self.board, axis=1)
                ]
            )
        )
        board = self.board.ravel()
        # Проверим если какой-либо из игроков выиграл
        if any(np.abs(combinations_sum) == 3):
            return True, np.where(board == 0)[0].size
        # Если не выполняется, то игра продолжается
        return False, np.where(board == 0)[0].size

    def _feed_rewards(self):
        win, empties = self._check_if_game_is_over()
        if win:
            if self.num_actions % 2 == 0:
                self.agent1.set_reward(0.)
                self.agent2.set_reward(1.)
            else:
                self.agent1.set_reward(1.)
                self.agent2.set_reward(0.)
            self.is_over = True
        else:
            # print("HERE")
            self.agent1.set_reward(0.1)
            self.agent2.set_reward(0.1)
        if empties == 0:
            self.is_over = True

    def get_empty_cells(self):
        board = self.board.ravel()
        return np.where(board == 0)[0]

    def _set_sign(self, sign, cell):
        self.num_actions += 1
        board = self.board.ravel()
        board[cell] = sign
        self.board = board.reshape((3, 3))
    
    def set_action(self, agent, cell):
        self._set_sign(agent.sign, cell)
        self._feed_rewards()
