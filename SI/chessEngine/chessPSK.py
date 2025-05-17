#!/usr/bin/env python3

import chess
from chess import Board, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK
import chess.polyglot
import chess.syzygy
import time
from enum import Enum
import sys
import random

'''
in python chess white color = 1
'''

class ZobristEntry(Enum):
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2


class chessPSK:
    def __init__(self, bin_path='bjbraams.bin', tb_path = './3-4-5_pieces_Syzygy/3-4-5'):
        self.book = chess.polyglot.open_reader(bin_path)
        self.board = chess.Board()
        self.tablebase = chess.syzygy.open_tablebase(tb_path)
        self.tt = {} # zobrist key hash trasnposition table
        self.max_depth = 64
        self.killer_moves = {d: [] for d in range(self.max_depth)}
        self.history = [[0] * 64 for _ in range(64)]
        self.VALUES = { PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 0}

    def get_book_move(self, board):
        entry = self.book.weighted_choice(board)
        return entry.move if entry else None

    def close_book(self):
        self.book.close()
    
    def draw(self):
        print(self.board)
    
    def count_pieces(self):
        return len(self.board.piece_map())

    def get_syzygy_move(self):
        if self.count_pieces() > 5:
            return None
        
        best_move = None
        best_score = -1
        best_dtz = None

        for move in self.board.legal_moves:
            self.board.push(move)
            
            wdl = self.tablebase.probe_wdl(self.board)
            if wdl is None:
                # no position in tb 
                self.board.pop()
                continue
            
            if wdl > best_score:
                best_score = wdl
                best_move = move
                best_dtz = self.tablebase.probe_dtz(self.board)
            elif wdl == best_score == 2:
                dtz = self.tablebase.probe_dtz(self.board)
                if dtz < best_dtz:
                    best_move = move
                    best_dtz = dtz
            
            self.board.pop()

            if best_score == 2:
                break
        
        return best_move

    def probe_tb_score(self):
        """
        Zamienia wynik z syzygy (wdl, dtz) na centipiony:
          – wdl > 0 → wygrana → duży bonus minus dtz (krótsze maty lepsze)
          – wdl < 0 → porażka → minus duży bonus plus dtz (opóźnij porażkę)
          – wdl == 0 lub None → remis lub brak TB → 0
        """
        wdl = self.tablebase.probe_wdl(self.board)
        dtz = self.tablebase.probe_dtz(self.board) or 0

        MATE_SCORE = 100_000

        if wdl is None:
            return 0

        if wdl > 0:
            return MATE_SCORE - dtz
        elif wdl < 0:
            return -MATE_SCORE + dtz
        else:
            return 0

    # maintainance of zobrist transposition table
    def probe_tt(self, key, depth, alpha, beta):
        entry = self.tt.get(key)
        if entry is None:
            return None
        e_depth, e_score, e_flag, e_move = entry
        if e_depth < depth:
            return None
        
        if e_flag == ZobristEntry.EXACT:
            return e_score, e_move
        elif e_flag == ZobristEntry.LOWERBOUND and e_score > alpha:
            alpha = e_score
        elif e_flag == ZobristEntry.UPPERBOUND and e_score < beta:
            beta = e_score
        
        if alpha >= beta:
            return e_score, e_move
        return None
    
    def store_tt(self, key, depth, score, flag, move):
        self.tt[key] = (depth, score, flag, move)

    def evaluate(self):
        PST_PAWN = [
            0,   0,   0,   0,   0,   0,   0,   0,
            5,  10,  10, -20, -20,  10,  10,   5,
            5,  -5, -10,   0,   0, -10,  -5,   5,
            0,   0,   0,  20,  20,   0,   0,   0,
            5,   5,  10,  25,  25,  10,   5,   5,
            10,  10,  20,  30,  30,  20,  10,  10,
            50,  50,  50,  50,  50,  50,  50,  50,
            0,   0,   0,   0,   0,   0,   0,   0,
        ]

        PST_KNIGHT = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
        ]

        PST_BISHOP = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
        ]

        PST_ROOK = [
            0,   0,   5,  10,  10,   5,   0,   0,
            0,   0,   5,  10,  10,   5,   0,   0,
            0,   0,   5,  10,  10,   5,   0,   0,
            0,   0,   5,  10,  10,   5,   0,   0,
            0,   0,   5,  10,  10,   5,   0,   0,
            0,   0,   5,  10,  10,   5,   0,   0,
            25,  25,  25,  25,  25,  25,  25,  25,
            0,   0,   0,   0,   0,   0,   0,   0,
        ]

        PST_QUEEN = [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
            -5,   0,   5,   5,   5,   5,   0,  -5,
            0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        ]

        PST_KING = [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
            20,  20,   0,   0,   0,   0,  20,  20,
            20,  30,  10,   0,   0,  10,  30,  20,
        ]

        board = self.board
        score = 0
        VALUES = self.VALUES

        # material + piece-square table
        for sq, piece in board.piece_map().items():
            val = VALUES[piece.piece_type]
            color = 1 if piece.color == WHITE else -1

            score += color * val
            idx = sq if piece.color == WHITE else chess.square_mirror(sq)
            if piece.piece_type == PAWN: 
                pst = PST_PAWN
            elif piece.piece_type == KNIGHT:
                pst = PST_KNIGHT
            elif piece.piece_type == BISHOP:
                pst = PST_BISHOP
            elif piece.piece_type == ROOK:
                pst = PST_ROOK
            elif piece.piece_type == QUEEN:
                pst = PST_QUEEN
            elif piece.piece_type == KING:
                pst = PST_KING
            else:
                continue
        
            score += color * pst[idx]

        # legal_moves white - legal_moves black
        my_moves = len(list(board.legal_moves))
        board.push(chess.Move.null())
        opponents_moves = len(list(board.legal_moves))
        board.pop()
        score += 0.1 * (my_moves - opponents_moves)

        # centre control
        for center in (chess.E4, chess.D4, chess.E5, chess.D5):
            score += 0.2 * (len(board.attackers(WHITE, center))
                        - len(board.attackers(BLACK, center)))
        
        # pawn structure
        for color in (WHITE, BLACK):
            dir_ = 1 if color == WHITE else -1
            pawns = board.pieces(PAWN, color)
            files = [chess.square_file(sq) for sq in pawns]
            for f in set(files):
                cnt = files.count(f)
                if cnt > 1:
                    score += dir_ * (-0.5 * (cnt - 1))
            
            for sq in pawns:
                f = chess.square_file(sq)
                # isolated
                if f - 1 not in files and f + 1 not in files:
                    score += dir_ * -0.7
                opponent_pawns = board.pieces(PAWN, not color)
                opponent_files = [chess.square_file(x) for x in opponent_pawns]
                if all(x not in opponent_files for x in (f - 1, f, f + 1)):
                    rank = chess.square_rank(sq)
                    dist = (7 - rank) if color == WHITE else rank
                    score += dir_ * (0.2 * dist)
        
        # bishop pair
        if len(board.pieces(BISHOP, WHITE)) >= 2:
            score += 0.5
        if len(board.pieces(BISHOP, BLACK)) >= 2:
            score -= 0.5
        
        # rooks open and semi open lines
        for color in (WHITE, BLACK):
            dir_ = 1 if color == WHITE else -1
            pawn_files = {chess.square_file(sq) for sq in board.pieces(PAWN, color)}
            for sq in board.pieces(ROOK, color):
                f = chess.square_file(sq)
                if f not in pawn_files:
                    score += dir_ * 0.3
                    
        return score

    def update(self, uci_move):
        move = chess.Move.from_uci(uci_move)
        if move not in self.board.legal_moves:
            raise ValueError(f'Illegal move: {uci_move}')
        self.board.push(move)
    
    def moves(self):
        return [m.uci() for m in self.board.legal_moves]

    def search(self, move_time=0.03):
        try:
            m = self.get_book_move(self.board)
        except IndexError:
            m = None
        if m:
            return m

        m = self.get_syzygy_move()
        if m:
            return m

        # uci = self.iterative_deepening(move_time)
        return self.iterative_deepening(move_time)

    def iterative_deepening(self, move_time, max_depth=12):
        start = time.time()
        best_move = None
        INF = float('inf')
        NEG_INF = float('-inf')
        maximizing = self.board.turn

        for depth in range(1, max_depth + 1):
            try:
                _, cand = self.alpha_beta_search(depth, NEG_INF, INF, maximizing, start, move_time)
                if cand is not None:
                    best_move = cand
                
            except TimeoutError:
                break
            
        return best_move

    def order_moves(self, depth):
        moves = list(self.board.legal_moves)
        
        # position stored in tt -> hash move heuristics
        key = self.board._transposition_key()
        entry = self.tt.get(key)
        if entry and entry[0] >= depth:
            _, _, _, hash_move = entry
            if hash_move in moves:
                moves.remove(hash_move)
                yield hash_move
        
        # MVV-LVA heuristics
        captures = [move for move in moves if self.board.is_capture(move)]
        def mvv_lva_key(m):
            # typ zbitego piona
            victim = self.board.piece_type_at(m.to_square)
            # dla en passant -> pion
            if victim is None and self.board.is_en_passant(m):
                victim = PAWN
            # typ atakującego
            attacker = self.board.piece_type_at(m.from_square)
            return (self.VALUES[victim], -self.VALUES[attacker])
        
        captures.sort(key=mvv_lva_key, reverse=True)
        for move in captures:
            moves.remove(move)
            yield(move)

        # killer heuristics
        for killer in self.killer_moves.get(depth, []):
            if killer in moves:
                moves.remove(killer)
                yield killer
        
        # history heuristics
        quiets = [move for move in moves if not self.board.is_capture(move)]
        quiets.sort(key=lambda m: self.history[m.from_square][m.to_square], reverse=True)
        for move in quiets:
            moves.remove(move)
            yield move
        
        for move in moves:
            yield move

    def alpha_beta_search(self, depth, alpha, beta, maximizing, start, move_time):
        if time.time() - start >= move_time:
            raise TimeoutError
        
        # check whether this was searched previously
        key = self.board._transposition_key()
        tt_hit = self.probe_tt(key, depth, alpha, beta)
        if tt_hit is not None:
            return tt_hit

        if self.board.is_game_over():
            outcome = self.board.outcome()
            if outcome.winner is None:
                return 0, None
            return (float('inf'), None) if outcome.winner else (float('-inf'), None)

        if depth == 0:
            # we check if we can use syzygy
            if self.count_pieces() <= 5:
                return self.probe_tb_score(), None
            return self.evaluate(), None

        best_move = None
        alpha_orig = alpha
        beta_orig = beta

        if maximizing:
            max_score = float('-inf')
            for mv in self.order_moves(depth):
                self.board.push(mv)
                score, move = self.alpha_beta_search(depth - 1, alpha, beta, False, start, move_time)
                self.board.pop()

                if score > max_score:
                    best_move = mv
                    max_score = score
                alpha = max(alpha, score)
                if beta <= alpha:
                    killer_list = self.killer_moves[depth]
                    if mv not in killer_list:
                        killer_list.append(mv)
                        if len(killer_list) > 2:
                            killer_list.pop(0)
                    if not self.board.is_capture(mv):
                        f, t = mv.from_square, mv.to_square
                        self.history[f][t] += depth**2
                    break

            value = max_score
        else:
            min_score = float('inf')
            for mv in self.order_moves(depth):
                self.board.push(mv)
                score, move = self.alpha_beta_search(depth - 1, alpha, beta, True, start, move_time)
                self.board.pop()

                if score < min_score:
                    best_move = mv
                    min_score = score
                beta = min(beta, score)
                if beta <= alpha:
                    killer_list = self.killer_moves[depth]
                    if mv not in killer_list:
                        killer_list.append(mv)
                        if len(killer_list) > 2:
                            killer_list.pop(0)
                    if not self.board.is_capture(mv):
                        f, t = mv.from_square, mv.to_square
                        self.history[f][t] += depth**2
                    break

            value = min_score

        if value <= alpha_orig:
            flag = ZobristEntry.UPPERBOUND
        elif value >= beta_orig:
            flag = ZobristEntry.LOWERBOUND
        else:
            flag = ZobristEntry.EXACT
        
        key = self.board._transposition_key()
        self.store_tt(key, depth, value, flag, best_move)

        return value, best_move

class Player(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.game = chessPSK()
        self.my_player = 1
        self.say('RDY')

    def say(self, what):
        sys.stdout.write(what)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def hear(self):
        line = sys.stdin.readline().split()
        return line[0], line[1:]

    def loop(self):
        while True:
            cmd, args = self.hear()

            if cmd == 'HEDID':
                # przeciwnik wykonał ruch
                self.game.update(args[2])
                # tutaj już nie robimy `continue`
            elif cmd == 'UGO':
                # pierwsze wezwanie – też idziemy do wspólnego kodu
                pass
            elif cmd == 'ONEMORE':
                self.reset()
                continue
            elif cmd == 'BYE':
                self.game.close_book()
                break
            else:
                raise RuntimeError(f"Nieoczekiwana komenda: {cmd}")

            # obie ścieżki (HEDID i UGO) trafiają tu:
            move_time = float(args[0]) - 0.1    # <-- tu wyciągasz limit czasu
            move = self.game.search(move_time=move_time)
            # aktualizujesz stan i odsyłasz ruch
            self.game.update(move.uci())
            self.say('IDO ' + move.uci())

if __name__ == '__main__':
    player = Player()
    player.loop()
