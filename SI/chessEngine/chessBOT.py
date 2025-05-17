#!/usr/bin/env python3
import sys, time, random
import chess
from chess import (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
                   WHITE, BLACK)
import chess.polyglot
import chess.syzygy
from enum import Enum

# ──────────────────────────────────────────────────────────────
class ZobristEntry(Enum):
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2
# ──────────────────────────────────────────────────────────────
class chessPSK:
    def __init__(self,
                 bin_path='bjbraams.bin',
                 tb_path='./3-4-5_pieces_Syzygy/3-4-5'):
        try:
            self.book = chess.polyglot.open_reader(bin_path)
        except FileNotFoundError:
            self.book = None
        self.board = chess.Board()
        try:
            self.tablebase = chess.syzygy.open_tablebase(tb_path)
        except FileNotFoundError:
            self.tablebase = None

        self.tt = {}
        self.tt_limit = 2_000_000        # max pozycji w hash-tablicy
        self.max_depth = 64
        self.killer_moves = {d: [] for d in range(self.max_depth)}
        self.history = [[0] * 64 for _ in range(64)]
        self.VALUES = {PAWN: 1, KNIGHT: 3, BISHOP: 3,
                       ROOK: 5, QUEEN: 9, KING: 0}

    # ── podstawowe pomocnicze ────────────────────────────────
    def count_pieces(self):
        return len(self.board.piece_map())

    # ── książka i Syzygy ─────────────────────────────────────
    def get_book_move(self):
        if self.book is None:
            return None
        try:
            entry = self.book.weighted_choice(self.board)
        except IndexError:               # brak pozycji w pliku .bin
            return None
        return entry.move if entry else None


    def get_syzygy_move(self):
        if self.tablebase is None or self.count_pieces() > 5:
            return None
        best_move, best_score, best_dtz = None, -1, None
        for mv in self.board.legal_moves:
            self.board.push(mv)
            try:
                wdl = self.tablebase.probe_wdl(self.board)
                if wdl is None:
                    continue
                if wdl > best_score:
                    best_score, best_move = wdl, mv
                    best_dtz = self.tablebase.probe_dtz(self.board)
                elif wdl == best_score == 2:
                    dtz = self.tablebase.probe_dtz(self.board)
                    if dtz < best_dtz:
                        best_move, best_dtz = mv, dtz
            finally:
                self.board.pop()
            if best_score == 2:
                break
        return best_move

    def probe_tb_score(self):
        if self.tablebase is None or self.count_pieces() > 5:
            return 0
        wdl = self.tablebase.probe_wdl(self.board)
        dtz = self.tablebase.probe_dtz(self.board) or 0
        MATE = 100_000
        if wdl is None:
            return 0
        if wdl > 0:
            return MATE - dtz
        if wdl < 0:
            return -MATE + dtz
        return 0

    # ── tablica transpozycji ────────────────────────────────
    def probe_tt(self, key, depth, alpha, beta):
        e = self.tt.get(key)
        if e is None:
            return None
        d, score, flag, mv = e
        if d < depth:
            return None
        if flag == ZobristEntry.EXACT:
            return score, mv
        if flag == ZobristEntry.LOWERBOUND and score > alpha:
            alpha = score
        elif flag == ZobristEntry.UPPERBOUND and score < beta:
            beta = score
        if alpha >= beta:
            return score, mv
        return None

    def store_tt(self, key, depth, score, flag, move):
        if len(self.tt) > self.tt_limit:
            self.tt.clear()
        self.tt[key] = (depth, score, flag, move)

    # ── ocena pozycji  ───────────────────────────────────────
    def evaluate(self):
        # PST-y skrócone (jak w pierwotnym kodzie) – nie pokazuję ponownie
        # ... [tu wklej PST_PAWN, PST_KNIGHT, ... PST_KING,
        #      dokładnie jak w Twojej dotychczasowej wersji] ...
        # ────────────────────────────────────────────────────

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

        board, score = self.board, 0
        for sq, pc in board.piece_map().items():
            val = self.VALUES[pc.piece_type]
            col = 1 if pc.color == WHITE else -1
            score += col * val
            idx = sq if pc.color == WHITE else chess.square_mirror(sq)
            pst = (PST_PAWN, PST_KNIGHT, PST_BISHOP,
                   PST_ROOK, PST_QUEEN, PST_KING)[pc.piece_type - 1]
            score += col * pst[idx]
        # mobilność
        my_moves = len(list(board.legal_moves))
        board.push(chess.Move.null())
        opp_moves = len(list(board.legal_moves))
        board.pop()
        score += 0.1 * (my_moves - opp_moves)
        # kontrola centrum
        for c in (chess.E4, chess.D4, chess.E5, chess.D5):
            score += 0.2 * (len(board.attackers(WHITE, c)) -
                            len(board.attackers(BLACK, c)))
        # bishop pair
        if len(board.pieces(BISHOP, WHITE)) >= 2:
            score += 0.5
        if len(board.pieces(BISHOP, BLACK)) >= 2:
            score -= 0.5
        # tapered-eval: lekkie figury mniej warte w końcówce
        phase = min(24, len(board.piece_map()))
        eg = (24 - phase) / 24
        score -= eg * 0.05 * (len(board.pieces(KNIGHT, WHITE)) -
                              len(board.pieces(KNIGHT, BLACK)))
        score -= eg * 0.05 * (len(board.pieces(BISHOP, WHITE)) -
                              len(board.pieces(BISHOP, BLACK)))
        return score

        # fallback – bardzo uproszczony static exchange evaluation
    def _see_lite(self, move):
        if not self.board.is_capture(move):
            return 0
        victim = self.board.piece_type_at(move.to_square)
        attacker = self.board.piece_type_at(move.from_square)
        return self.VALUES.get(victim, 0) - self.VALUES.get(attacker, 0)


    # ── generowanie i porządkowanie ruchów ───────────────────
    def order_moves(self, depth):
        moves = list(self.board.legal_moves)
        key = self.board._transposition_key()
        e = self.tt.get(key)
        if e and e[0] >= depth:
            _, _, _, hash_move = e
            if hash_move in moves:
                moves.remove(hash_move)
                yield hash_move
        good_caps, bad_caps, quiets = [], [], []
        for mv in moves:
            if mv.promotion:
                good_caps.append(mv)
            elif self.board.is_capture(mv):
                see_val = (self.board.see(mv) if hasattr(self.board, "see")
                                else self._see_lite(mv))
                (good_caps if see_val >= 0 else bad_caps).append(mv)
            else:
                quiets.append(mv)
        yield from sorted(good_caps,
                          key=lambda m: self.VALUES.get(
                              self.board.piece_type_at(m.to_square), 0),
                          reverse=True)
        for k in self.killer_moves.get(depth, []):
            if k in quiets:
                quiets.remove(k)
                yield k
        quiets.sort(key=lambda m: self.history[m.from_square][m.to_square],
                    reverse=True)
        yield from quiets
        yield from bad_caps

    # ── quiescence search ────────────────────────────────────
    def quiescence(self, alpha, beta, start, move_time):
        if time.time() - start >= move_time:
            raise TimeoutError
        stand = self.evaluate()
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand
        for mv in self.board.legal_moves:
            if not (self.board.is_capture(mv) or mv.promotion):
                continue
            self.board.push(mv)
            try:
                val = -self.quiescence(-beta, -alpha, start, move_time)
            finally:
                self.board.pop()
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val
        return alpha

    # ── główne przeszukiwanie ────────────────────────────────
    def alpha_beta_search(self, depth, alpha, beta,
                          maximizing, start, move_time):
        if time.time() - start >= move_time:
            raise TimeoutError
        key = self.board._transposition_key()
        hit = self.probe_tt(key, depth, alpha, beta)
        if hit:
            return hit
        if self.board.is_game_over():
            out = self.board.outcome()
            if out.winner is None:
                return 0, None
            return (float('inf'), None) if out.winner else (-float('inf'), None)
        # null-move pruning
        if depth >= 3 and not self.board.is_check():
            self.board.push(chess.Move.null())
            try:
                score, _ = self.alpha_beta_search(
                    depth - 3, -beta, -beta + 1,
                    not maximizing, start, move_time)
                score = -score
            finally:
                self.board.pop()
            if score >= beta:
                return beta, None
        if depth == 0:
            return self.quiescence(alpha, beta, start, move_time), None
        best_move = None
        alpha_orig, beta_orig = alpha, beta
        if maximizing:
            best = float('-inf')
            for mv in self.order_moves(depth):
                self.board.push(mv)
                try:
                    val, _ = self.alpha_beta_search(
                        depth - 1, alpha, beta,
                        False, start, move_time)
                finally:
                    self.board.pop()
                if val > best:
                    best, best_move = val, mv
                alpha = max(alpha, val)
                if beta <= alpha:
                    self._update_killers(depth, mv)
                    break
        else:
            best = float('inf')
            for mv in self.order_moves(depth):
                self.board.push(mv)
                try:
                    val, _ = self.alpha_beta_search(
                        depth - 1, alpha, beta,
                        True, start, move_time)
                finally:
                    self.board.pop()
                if val < best:
                    best, best_move = val, mv
                beta = min(beta, val)
                if beta <= alpha:
                    self._update_killers(depth, mv)
                    break
        flag = (ZobristEntry.UPPERBOUND if best <= alpha_orig else
                ZobristEntry.LOWERBOUND if best >= beta_orig else
                ZobristEntry.EXACT)
        self.store_tt(key, depth, best, flag, best_move)
        return best, best_move

    def _update_killers(self, depth, mv):
        k = self.killer_moves[depth]
        if mv not in k:
            k.append(mv)
            if len(k) > 2:
                k.pop(0)
        if not self.board.is_capture(mv):
            self.history[mv.from_square][mv.to_square] += depth ** 2

    # ── iterative deepening + aspiration windows ─────────────
    def iterative_deepening(self, move_time, max_depth=12):
        start = time.time()
        moves = list(self.board.legal_moves)
        if not moves:
            return None
        best_move = moves[0]
        score = 0
        for d in range(1, max_depth + 1):
            window = 25
            alpha, beta = score - window, score + window
            while True:
                try:
                    score, cand = self.alpha_beta_search(
                        d, alpha, beta, self.board.turn,
                        start, move_time)
                except TimeoutError:
                    return best_move
                if cand:
                    best_move = cand
                if score <= alpha:
                    alpha -= window
                elif score >= beta:
                    beta += window
                else:
                    break
        return best_move

    # ── główny interfejs dla pętli I/O ──────────────────────
    def search(self, move_time=0.1):
        move_time = max(0.01, move_time)
        m = self.get_book_move()
        if m:
            return m
        m = self.get_syzygy_move()
        if m:
            return m
        m = self.iterative_deepening(move_time)
        return m if m else chess.Move.null()

    def update(self, uci):
        m = chess.Move.from_uci(uci)
        if m not in self.board.legal_moves:
            raise ValueError(f'illegal move: {uci}')
        self.board.push(m)

    def close_book(self):
        if self.book:
            self.book.close()
# ──────────────────────────────────────────────────────────────
class Player:
    def __init__(self):
        self.reset()

    def say(self, txt):
        sys.stdout.write(txt + '\n')
        sys.stdout.flush()

    def hear(self):
        line = sys.stdin.readline().split()
        return line[0], line[1:]

    def reset(self):
        self.game = chessPSK()
        self.say('RDY')

    def loop(self):
        while True:
            cmd, args = self.hear()
            if cmd == 'HEDID':
                self.game.update(args[-1])
            elif cmd == 'UGO':
                pass
            elif cmd == 'ONEMORE':
                self.reset()
                continue
            elif cmd == 'BYE':
                self.game.close_book()
                break
            else:
                raise RuntimeError('bad cmd')
            move_time = 1.5
            mv = self.game.search(move_time)
            self.game.update(mv.uci())
            self.say('IDO ' + mv.uci())
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    Player().loop()
