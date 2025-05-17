#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Reversi do dueller.py
bit-board + αβ + iter-deep + transpozycja
"""

import sys, time, random
from typing import Dict, Tuple, List, Optional


# ───────────────────────────────────────────────────────── Silnik ──
class ReversiEngine:
    ZOBRIST = {}

    def __init__(self):
        self.white_board = (1 << 27) | (1 << 36)   # 'o' – zaczyna
        self.black_board = (1 << 28) | (1 << 35)   # '#'
        self.init_zobrist()
        self.tt = {}                               # transpozycja

    # --- inicjalizacja kluczy Zobrista
    def init_zobrist(self) -> None:
        for color in ('w', 'b'):
            for sq in range(64):
                ReversiEngine.ZOBRIST[(color, sq)] = random.getrandbits(64)
        ReversiEngine.ZOBRIST['white_to_move'] = random.getrandbits(64)

    def compute_zobrist(self, white_to_move: bool) -> int:
        h = 0
        wb = self.white_board
        while wb:
            sq = (wb & -wb).bit_length() - 1
            h ^= ReversiEngine.ZOBRIST[('w', sq)]
            wb &= wb - 1
        bb = self.black_board
        while bb:
            sq = (bb & -bb).bit_length() - 1
            h ^= ReversiEngine.ZOBRIST[('b', sq)]
            bb &= bb - 1
        if white_to_move:
            h ^= ReversiEngine.ZOBRIST['white_to_move']
        return h

    # --- generacja ruchów
    @staticmethod
    def _empty(one: int, two: int) -> int:
        return ~(one | two) & ((1 << 64) - 1)

    def get_legal_moves(self, player: int, adversary: int) -> Dict[int, int]:
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        legal, empties = {}, self._empty(player, adversary)
        while empties:
            sq_num = (empties & -empties).bit_length() - 1
            empties &= empties - 1
            flip_mask = 0
            r, c = divmod(sq_num, 8)
            for dx, dy in directions:
                nr, nc = r + dx, c + dy
                path = 0
                while 0 <= nr < 8 and 0 <= nc < 8 and \
                        (1 << (nr*8 + nc)) & adversary:
                    path |= 1 << (nr*8 + nc)
                    nr += dx; nc += dy
                if 0 <= nr < 8 and 0 <= nc < 8 and \
                        (1 << (nr*8 + nc)) & player:
                    flip_mask |= path
            if flip_mask:
                legal[sq_num] = flip_mask
        return legal

    def create_new_state(self, player: int, adversary: int,
                         move: int, flip: int) -> Tuple[int, int]:
        player |= (1 << move) | flip
        adversary &= ~flip
        return player, adversary

    def is_terminal_state(self) -> bool:
        return self._empty(self.white_board, self.black_board) == 0

    # --- ewaluacja (zawsze z perspektywy białych!)
    def piece_diff(self) -> int:
        return self.white_board.bit_count() - self.black_board.bit_count()

    def legal_moves_diff(self) -> int:
        wl = len(self.get_legal_moves(self.white_board, self.black_board))
        bl = len(self.get_legal_moves(self.black_board, self.white_board))
        return wl - bl

    def safe_pieces(self, board: int) -> int:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        cnt = 0
        for corner in (0, 7, 56, 63):
            if (1 << corner) & board:
                cnt += 1
                r, c = divmod(corner, 8)
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < 8 and 0 <= nc < 8 and \
                            (1 << (nr*8 + nc)) & board:
                        cnt += 1
                        nr += dr; nc += dc
        return cnt

    def safe_pieces_diff(self) -> int:
        return self.safe_pieces(self.white_board) - self.safe_pieces(self.black_board)

    def phase_weights(self, empties: int) -> List[float]:
        if empties > 48:   return [0.25, 0.50, 0.25]
        if empties > 31:   return [0.25, 0.40, 0.35]
        if empties > 15:   return [0.35, 0.25, 0.40]
        return [0.50, 0.20, 0.30]

    def evaluate(self) -> float:
        empt = self._empty(self.white_board, self.black_board).bit_count()
        w1, w2, w3 = self.phase_weights(empt)
        return (w1 * self.piece_diff()
              + w2 * self.legal_moves_diff()
              + w3 * self.safe_pieces_diff())

    # --- transpozycja
    def insert_into_tt(self, z, val, depth, flag, best):
        old = self.tt.get(z)
        if old is None or depth >= old[1]:
            self.tt[z] = (val, depth, flag, best)

    def lookup_tt(self, z, depth, alpha, beta):
        e = self.tt.get(z)
        if e is None or e[1] < depth:
            return None
        val, _, flag, move = e
        if   flag == 0: return val, move
        elif flag == 1 and val <= alpha: return alpha, move
        elif flag == 2 and val >= beta:  return beta, move
        return None

    # --- minimax / negamax
    def minimax(self, depth, alpha, beta,
                start, limit, white_to_move, passed=False):

        # liść lub timeout
        if depth == 0 or self.is_terminal_state():
            v = self.evaluate()
            return (v if white_to_move else -v), None
        if time.time() - start >= limit:
            raise TimeoutError

        z = self.compute_zobrist(white_to_move)
        cached = self.lookup_tt(z, depth, alpha, beta)
        if cached: return cached

        player, adv = ((self.white_board, self.black_board)
                       if white_to_move else
                       (self.black_board, self.white_board))
        moves = self.get_legal_moves(player, adv)

        if not moves:                        # brak ruchu
            if passed:                       # obaj pasowali
                v = self.evaluate()
                return (v if white_to_move else -v), None
            val, _ = self.minimax(depth, -beta, -alpha,
                                   start, limit, not white_to_move, True)
            return -val, None

        best_val, best_move = -float('inf'), None
        a0, b0 = alpha, beta

        for mv, flip in moves.items():
            if white_to_move:
                nw, nb = self.create_new_state(self.white_board, self.black_board,
                                               mv, flip)
            else:
                nb, nw = self.create_new_state(self.black_board, self.white_board,
                                               mv, flip)
            # recurse
            pw, pb = self.white_board, self.black_board
            self.white_board, self.black_board = nw, nb
            val, _ = self.minimax(depth-1, -beta, -alpha,
                                   start, limit, not white_to_move, False)
            self.white_board, self.black_board = pw, pb
            val = -val

            if val > best_val:
                best_val, best_move = val, mv
            alpha = max(alpha, val)
            if alpha >= beta:
                break

        flag = 0
        if best_val <= a0: flag = 2
        elif best_val >= b0: flag = 1
        self.insert_into_tt(z, best_val, depth, flag, best_move)
        return best_val, best_move

    # --- iteracyjne pogłębianie (≤ 7)
    def iterative_deepening(self, time_limit: float, white_to_move: bool) -> Optional[int]:
        best, start, depth = None, time.time(), 1
        while depth <= 7 and time.time() - start < time_limit:
            try:
                _, mv = self.minimax(depth, -float('inf'), float('inf'),
                                     start, time_limit, white_to_move)
                if mv is not None:
                    best = mv
            except TimeoutError:
                break
            depth += 1
        return best


class DuelerPlayer:
    @staticmethod
    def _say(msg): sys.stdout.write(msg + '\n'); sys.stdout.flush()
    @staticmethod
    def _hear():
        ln = sys.stdin.readline()
        if not ln: sys.exit(0)
        p = ln.strip().split()
        return p[0], p[1:]

    @staticmethod
    def _sq(x, y): return y * 8 + x
    @staticmethod
    def _xy(sq):  return sq % 8, sq // 8

    def __init__(self): self.reset()
    def reset(self):
        self.engine = ReversiEngine()
        self.me = 1                 # 1 = ‘#’  (zmienimy gdy UGO)
        self._say("RDY")

    # -------- aktualizacja pozycji po ruchu -------------
    def _apply(self, move: Optional[Tuple[int,int]], player: int):
        if move is None: return
        sq = self._sq(*move)

        if player == 0:                           # gracz ‘o’  →  engine.black_board
            legal = self.engine.get_legal_moves(self.engine.black_board,
                                                self.engine.white_board)
            mask = legal.get(sq);  # None ⇒ nielegalny (ignorujemy)
            if mask is None: return
            self.engine.black_board, self.engine.white_board = \
                self.engine.create_new_state(self.engine.black_board,
                                             self.engine.white_board,
                                             sq, mask)

        else:                                    # gracz ‘#’  →  engine.white_board
            legal = self.engine.get_legal_moves(self.engine.white_board,
                                                self.engine.black_board)
            mask = legal.get(sq)
            if mask is None: return
            self.engine.white_board, self.engine.black_board = \
                self.engine.create_new_state(self.engine.white_board,
                                             self.engine.black_board,
                                             sq, mask)

    # -------- wybór ruchu -------------
    def _choose(self, move_time: float) -> Tuple[int,int]:
        safe = max(0.01, move_time - 0.10)        # 0.10 s bufor

        # lista ruchów dla bieżącego gracza
        if self.me == 0:   # ‘o’  == engine.black_board
            legal = self.engine.get_legal_moves(self.engine.black_board,
                                                self.engine.white_board)
            white_to_move = False
        else:              # ‘#’  == engine.white_board
            legal = self.engine.get_legal_moves(self.engine.white_board,
                                                self.engine.black_board)
            white_to_move = True

        if not legal:
            return (-1, -1)                       # pass

        sq = self.engine.iterative_deepening(safe, white_to_move)
        return (-1, -1) if sq is None else self._xy(sq)

    # -------- pętla protokołu -------------
    def loop(self):
        while True:
            cmd, args = self._hear()

            if cmd == "UGO":                      # rozpoczynamy partię
                self.me = 0                      # 0 = ‘o’ zaczyna
                t = float(args[0])
                mv = self._choose(t)
                if mv != (-1, -1): self._apply(mv, self.me)
                self._say(f"IDO {mv[0]} {mv[1]}")

            elif cmd == "HEDID":                  # ruch przeciwnika
                t = float(args[0])
                opp = tuple(map(int, args[2:]))
                if opp == (-1, -1): opp = None
                self._apply(opp, 1 - self.me)

                mv = self._choose(t)
                if mv != (-1, -1): self._apply(mv, self.me)
                self._say(f"IDO {mv[0]} {mv[1]}")

            elif cmd == "ONEMORE":
                self.reset()
            elif cmd == "BYE":
                break


# ───────────────────────────────────────────────────────────── main ──
if __name__ == "__main__":
    DuelerPlayer().loop()
