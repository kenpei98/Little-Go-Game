import sys
import math
from functools import lru_cache
from queue import Queue
from collections import defaultdict, Counter

def readInput(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()
        piece_type = int(lines[0])
        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]
        return piece_type, previous_board, board

def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])
    with open(path, 'w') as f:
        f.write(res)

class GO:
    def __init__(self, n):
        self.size = n
        self.previous_board = None
        self.board = None
        self.died_pieces = []
        self.history_boards = []
        self.history_died_pieces = []
        self.board_values = []
        for i in range(self.size):
            lis = []
            for j in range(self.size):
                x =  min(i, self.size - 1 - i) + min(j, self.size - 1 - j)
                lis.append(1.6 * x + 1)
            self.board_values.append(lis)

    def set_board(self, piece_type, previous_board, board):
        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        self.previous_board = previous_board
        self.history_boards.append(self.previous_board)
        self.board = board

    @lru_cache(None)
    def get_neighbors(self, i, j):
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
        if i < self.size - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < self.size - 1:
            neighbors.append((i, j + 1))
        return neighbors

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def valid_place_check(self, i, j, piece_type):
        for p, q in self.get_neighbors(i, j):
            if self.board[p][q] == 0:
                return True

        self.board[i][j] = piece_type
        if self.find_liberty(i, j):
            self.board[i][j] = 0
            return True
        test_go = self.copy_board()
        self.board[i][j] = 0
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            return False
        elif self.died_pieces and self.compare_board(self.previous_board, test_go.board):
            return False
        return True
    
    def clone_board(self, board):
        new_board = []
        for i in range(len(board)):
            new_board.append([x for x in board[i]])
        return new_board

    def copy_board(self):
        new_board = GO(self.size)
        new_board.previous_board = self.clone_board(self.previous_board)
        new_board.board = self.clone_board(self.board)
        new_board.died_pieces = [x for x in self.died_pieces]
        return new_board

    def ally_dfs(self, i, j, visited):
        yield i, j
        visited.add((i, j))
        for p, q in self.get_neighbors(i, j):
            if self.board[p][q] == self.board[i][j] and (p, q) not in visited:
                yield from self.ally_dfs(p, q, visited)

    def find_liberty(self, i, j):
        # check neighbors first
        for piece in self.get_neighbors(i, j):
            if self.board[piece[0]][piece[1]] == 0:
                return True

        for p, q in self.ally_dfs(i, j, set()):
            for piece in self.get_neighbors(p, q):
                if self.board[piece[0]][piece[1]] == 0:
                    return True
        return False

    def find_died_pieces(self, piece_type):
        died_pieces = set()
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == piece_type:
                    if not self.find_liberty(i, j):
                        died_pieces.add((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces:
            return []
        for piece in died_pieces:
            self.board[piece[0]][piece[1]] = 0
        return died_pieces

    def game_end(self, action="MOVE"):
        return self.compare_board(self.previous_board, self.board) and action == "PASS"

    def do_action(self, action, piece_type):
        self.previous_board = self.clone_board(self.board)
        self.history_boards.append(self.previous_board)
        self.history_died_pieces.append([x for x in self.died_pieces])
        if action != "PASS":
            self.board[action[0]][action[1]] = piece_type
            self.died_pieces = list(self.remove_died_pieces(3 - piece_type))
        else:
            self.died_pieces = []

    def undo_action(self):
        self.board = self.history_boards.pop()
        self.died_pieces = self.history_died_pieces.pop()
        self.previous_board = self.history_boards[-1]


class MyPlayer():
    def __init__(self):
        self.type = 'my_player'
        self.previous_boards = []
        self.died_pieces = []
        self.max_depth = 3
        self.go = None
        self.my_piece_type = None
        self.oppo_piece_type = None

    def get_input(self, go, piece_type):
        self.go = go
        actions = self.get_possible_actions(piece_type)
        if actions[0] == "PASS":
            return "PASS"
        elif len(actions) == 25:
            return (2, 2)
        else:
            if len(actions) <= 5:
                self.max_depth = 5
            self.my_piece_type = piece_type
            self.oppo_piece_type = 3 - piece_type
            best_action, best_value, alpha, beta = None, None, -float('inf'), float('inf')
            for action in actions:
                self.go.do_action(action, piece_type)
                value = self.alphabeta_search(self.max_depth, alpha, beta, 3 - piece_type, action)
                # print("action", action, round(value, 2))
                if best_value is None or value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
                self.go.undo_action()
                if alpha >= beta:
                    break

            if math.isinf(best_value):
                return 'PASS'

            return best_action

    def alphabeta_search(self, depth, alpha, beta, piece_type, last_action):
        actions = self.get_possible_actions(piece_type)
        if depth == 0 or self.go.game_end(last_action):
            return self.evaluate_board()

        if piece_type == self.my_piece_type:
            best_value = -float('inf')
            for action in actions:
                self.go.do_action(action, piece_type)
                best_value = max(best_value, self.alphabeta_search(depth - 1, alpha, beta, 3 - piece_type, action))
                alpha = max(alpha, best_value)
                self.go.undo_action()
                if alpha >= beta:
                    break
            return best_value
        else:
            best_value = float('inf')
            for action in actions:
                self.go.do_action(action, piece_type)
                best_value = min(best_value, self.alphabeta_search(depth - 1, alpha, beta, 3 - piece_type, action))
                beta = min(beta, best_value)
                self.go.undo_action()
                if alpha >= beta:
                    break
            return best_value

    def get_possible_actions(self, piece_type):
        possible_actions = []
        for i in range(self.go.size):
            for j in range(self.go.size):
                if self.go.board[i][j] == 0 and self.go.valid_place_check(i, j, piece_type):
                    possible_actions.append((i, j))

        if not possible_actions:
            return ["PASS"]
        else:
            return list(possible_actions)

    def evaluate_board(self):
        piece_scores = defaultdict(int)
        piece_counter = defaultdict(int)
        go = self.go
        board_score = 0

        visited = set()
        root_pieces = defaultdict(list)
        queue = Queue()
        for i in range(0, go.size):
            for j in range(0, go.size):
                if go.board[i][j] == 0:
                    queue.put(((i, j), (i, j)))
        if queue.qsize() < 15:
            while not queue.empty():
                (p, q), root = queue.get()
                if (p, q) in visited:
                    continue
                visited.add((p, q))
                for a, b in go.get_neighbors(p, q):
                    if go.board[a][b] == 0:
                        if (a, b) not in visited:
                            queue.put(((a, b), root))
                    else:
                        root_pieces[root].append(go.board[a][b])

            for root, pieces in root_pieces.items():
                c = Counter(pieces)
                if c[self.my_piece_type] == 0 and c[self.oppo_piece_type] > 0:
                    board_score -= c[self.oppo_piece_type] * 0.1
                elif c[self.my_piece_type] > 0 and c[self.oppo_piece_type] == 0:
                    board_score += c[self.my_piece_type] * 0.1

        for i in range(0, go.size):
            for j in range(0, go.size):
                piece_type = go.board[i][j]
                if piece_type != 0:
                    piece_counter[piece_type] += 1
                    piece_counter[piece_type] += self.go.board_values[i][j]
                    neighbors = go.get_neighbors(i, j)
                    neighbor_counter = Counter([go.board[p][q] for p, q in neighbors])
                    liberty_cnt = neighbor_counter[0]
                    self_cnt = neighbor_counter[piece_type]
                    oppo_cnt = neighbor_counter[3 - piece_type]
                    score = 0

                    if len(neighbors) == liberty_cnt:
                        score += 6
                    if len(neighbors) == 4:
                        if self_cnt == 0:
                            if liberty_cnt == 0 and oppo_cnt > 1:
                                score -= 5
                            elif oppo_cnt > 1:
                                score -= min(5, liberty_cnt * 0.5)
                        elif liberty_cnt >= 2:
                            score += 1
                        elif liberty_cnt >= 3:
                            score += 0.6
                    else:
                        if len(neighbors) <= 2:
                            score -= 3
                        else:
                            score -= 2
                        if self_cnt == 0:
                            if liberty_cnt == 0 and oppo_cnt > 1:
                                score -= 5
                            elif oppo_cnt > 1:
                                score -= min(5, liberty_cnt * 0.5)
                        elif liberty_cnt > 2:
                            score += 0.5
                        elif self_cnt > 1:
                            score += 0.2
                    piece_scores[piece_type] += score

        board_score += (piece_scores[self.my_piece_type] * 0.1 + piece_counter[self.my_piece_type])
        board_score -= (piece_scores[self.oppo_piece_type] * 0.1 + piece_counter[self.oppo_piece_type])
        return board_score


def main():
    args = sys.argv[1:]
    if len(args):
        input_file = args[0]
    else:
        input_file = 'input.txt'
    piece_type, previous_board, board = readInput(5, input_file)
    go_board = GO(5)
    go_board.set_board(piece_type, previous_board, board)
    player = MyPlayer()
    action = player.get_input(go_board, piece_type)
    writeOutput(action)


if __name__ == "__main__":
    main()
