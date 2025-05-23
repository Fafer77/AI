#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reversi – szybki bit-boardowy bot (α–β + TT + killer/history + LMR + null-move)
Komunikuje się zgodnie z protokółem: RDY / UGO / HEDID / ONEMORE / BYE
"""
from __future__ import annotations
import sys, time, random
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SILNIK  (Twój kod – bez zmian)
# ─────────────────────────────────────────────────────────────────────────────
class BitboardReversiAgent:
    """Szybki bot do Othello: α–β + PV-Search + killer/history, LMR, null-move."""

    BOARD  = 0xFFFFFFFFFFFFFFFF
    NOT_A  = 0xFEFEFEFEFEFEFEFE
    NOT_H  = 0x7F7F7F7F7F7F7F7F
    CORNERS = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)

    PSQ = [
        120, -20,  20,   5,   5,  20, -20, 120,
        -20, -40,  -5,  -5,  -5,  -5, -40, -20,
         20,  -5,  15,   3,   3,  15,  -5,  20,
          5,  -5,   3,   3,   3,   3,  -5,   5,
          5,  -5,   3,   3,   3,   3,  -5,   5,
         20,  -5,  15,   3,   3,  15,  -5,  20,
        -20, -40,  -5,  -5,  -5,  -5, -40, -20,
        120, -20,  20,   5,   5,  20, -20, 120,
    ]

    EXACT, LOWER, UPPER = 0, 1, 2              # flagi w TT
    MAX_PLY = 64                               # maks. głębokość (wystarczy)

    def __init__(self, time_per_move: float = 0.005):
        self.time_per_move = time_per_move
        self.TT: dict[tuple[int, int], tuple[int, int, int, int | None]] = {}
        self.killers = [[None, None] for _ in range(self.MAX_PLY)]
        self.history = [0] * 64                # history heuristic

        # funkcje przesunięć o 8 kierunków
        self.DIRS = (
            lambda b: (b << 8)  & 0xFFFFFFFFFFFFFF00,                 # N
            lambda b: (b >> 8)  & 0x00FFFFFFFFFFFFFF,                 # S
            lambda b: (b << 1)  & self.NOT_A,                         # E
            lambda b: (b >> 1)  & self.NOT_H,                         # W
            lambda b: (b << 9)  & self.NOT_A & 0xFFFFFFFFFFFFFF00,    # NE
            lambda b: (b << 7)  & self.NOT_H & 0xFFFFFFFFFFFFFF00,    # NW
            lambda b: (b >> 7)  & self.NOT_A & 0x00FFFFFFFFFFFFFF,    # SE
            lambda b: (b >> 9)  & self.NOT_H & 0x00FFFFFFFFFFFFFF,    # SW
        )

    # ------------- Bit-utilities -------------
    @staticmethod
    def _pop(x: int) -> int:
        return x.bit_count()

    def _legal(self, me, opp):
        empty, moves = ~(me | opp) & self.BOARD, 0
        for sh in self.DIRS:
            x = sh(me) & opp
            while x:
                x = sh(x)
                if not x: break
                moves |= x & empty
                x &= opp
        return moves

    def _do_move(self, me, opp, idx: int):
        mv, flips = 1 << idx, 0
        for sh in self.DIRS:
            mask, cap = sh(mv) & opp, 0
            while mask:
                cap |= mask
                mask = sh(mask)
                if mask & me: flips |= cap; break
                if not mask & opp: break
        return me | mv | flips, opp ^ flips

    # ------------- heurystyka -------------
    def _eval(self, me, opp):
        diff  = self._pop(me) - self._pop(opp)
        m_me  = self._pop(self._legal(me,  opp))
        m_op  = self._pop(self._legal(opp, me))
        mob   = 100 * (m_me - m_op) / (m_me + m_op) if m_me + m_op else 0
        c_me  = self._pop(me  & self.CORNERS)
        c_op  = self._pop(opp & self.CORNERS)
        corn  = 25 * (c_me - c_op)

        psq = 0
        bb = me
        while bb:
            lsb  = bb & -bb
            psq += self.PSQ[lsb.bit_length() - 1]
            bb  ^= lsb
        bb = opp
        while bb:
            lsb  = bb & -bb
            psq -= self.PSQ[lsb.bit_length() - 1]
            bb  ^= lsb

        empties = 64 - self._pop(me | opp)
        if empties < 12:                         # końcówka
            return diff * 1000 + corn + mob + psq
        return psq + corn + mob + diff

    # ---------- α–β z TT, killerami, history, LMR, null-move ----------
    def _ab(self, me, opp, depth, α, β, passed, dl, ply):
        if time.perf_counter() >= dl:
            return 0, None
        
        # ─── D O D A J  T O ───────────────────────────────────────────
        if ply >= len(self.killers):                # powiększ tablicę killerów
            self.killers.extend([[None, None]       # gdy drzewo jest głębsze
                                for _ in range(ply - len(self.killers) + 1)])
        # ──────────────────────────────────────────────────────────────

        key = (me, opp)
        ent = self.TT.get(key)
        if ent and ent[0] >= depth:
            _, flag, val, best = ent
            if flag == self.EXACT:  return val, best
            if flag == self.LOWER:  α = max(α, val)
            if flag == self.UPPER:  β = min(β, val)
            if α >= β:              return val, best

        moves = self._legal(me, opp)
        if depth == 0 or (moves == 0 and passed):
            return self._eval(me, opp), None
        if moves == 0:
            val, _ = self._ab(opp, me, depth, -β, -α, True, dl, ply+1)
            return -val, None

        if depth >= 3 and (me | opp).bit_count() <= 46:
            null_depth = depth - 3
            val, _ = self._ab(opp, me, null_depth, -β, -β+1, False, dl, ply+1)
            if -val >= β:
                return β, None

        order = []
        if ent and ent[3] is not None:
            order.append(ent[3]); moves ^= 1 << ent[3]
        k1, k2 = self.killers[ply]
        for k in (k1, k2):
            if k is not None and (moves >> k) & 1:
                order.append(k); moves ^= 1 << k
        corners = moves & self.CORNERS
        while corners:
            m = (corners & -corners).bit_length() - 1
            order.append(m); moves ^= 1 << m; corners ^= 1 << m
        rest = []
        while moves:
            m = (moves & -moves).bit_length() - 1
            rest.append(m); moves ^= 1 << m
        rest.sort(key=lambda mv: self.history[mv], reverse=True)
        order.extend(rest)

        best, α0, value = None, α, -float('inf')
        for i, mv in enumerate(order):
            nm, no = self._do_move(me, opp, mv)
            reduction = 1 if (depth >= 3 and i >= 4) else 0
            child_depth = depth - 1 - reduction
            sc, _ = self._ab(no, nm, child_depth, -β, -α, False, dl, ply+1)
            sc = -sc
            if reduction and sc > α:
                sc, _ = self._ab(no, nm, depth-1, -β, -α, False, dl, ply+1)
                sc = -sc
            if sc > value:
                value, best = sc, mv
            if value > α:
                α = value
            if α >= β:
                if mv != self.killers[ply][0]:
                    self.killers[ply][1] = self.killers[ply][0]
                    self.killers[ply][0] = mv
                self.history[mv] += depth * depth
                break

        flag = self.EXACT if α0 < value < β else (self.LOWER if value >= β else self.UPPER)
        self.TT[key] = (depth, flag, value, best)
        return value, best

    # ---------- API ----------
    def choose_move(self, my_bits: int, opp_bits: int) -> int | None:
        self.killers = [[None, None] for _ in range(len(self.killers))]

        deadline = time.perf_counter() + self.time_per_move
        depth, best = 1, None
        while True:
            if time.perf_counter() >= deadline: break
            val, mv = self._ab(my_bits, opp_bits, depth,
                               -float('inf'), float('inf'), False,
                               deadline, 0)
            if time.perf_counter() >= deadline: break
            if mv is not None: best = mv
            depth += 1
        return best

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ADAPTER PROTOKOŁU  (nowa klasa Player)
# ─────────────────────────────────────────────────────────────────────────────
class Player:
    def __init__(self):
        self.agent = BitboardReversiAgent()
        self.reset()
        self._send("RDY")

    # ---------- komunikacja ----------
    @staticmethod
    def _send(msg: str):
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    @staticmethod
    def _read():
        line = sys.stdin.readline()
        return line.strip().split() if line else []

    # ---------- konwersje ----------
    @staticmethod
    def _xy2idx(x: int, y: int) -> int:
        return y * 8 + x

    @staticmethod
    def _idx2xy(idx: int) -> tuple[int, int]:
        return idx % 8, idx // 8

    # ---------- stan gry ----------
    def reset(self):
        self.bits = [0, 0]                       # 0 = 'o', 1 = '#'
        self.bits[1] = (1 << self._xy2idx(3, 3)) | (1 << self._xy2idx(4, 4))
        self.bits[0] = (1 << self._xy2idx(4, 3)) | (1 << self._xy2idx(3, 4))
        self.my = None                           # mój kolor zostanie ustalony przy UGO/HEDID

    def _apply_move(self, color: int, move: tuple[int, int] | None):
        if move is None: return
        idx = self._xy2idx(*move)
        self.bits[color], self.bits[1 - color] = self.agent._do_move(
            self.bits[color], self.bits[1 - color], idx
        )

    # ---------- wybór ruchu ----------
    def _choose(self, timeout: float) -> tuple[int, int]:
        self.agent.time_per_move = 0.01
        idx = self.agent.choose_move(self.bits[self.my], self.bits[1 - self.my])
        if idx is None:
            return -1, -1
        move = self._idx2xy(idx)
        self._apply_move(self.my, move)
        return move

    # ---------- główna pętla ----------
    def loop(self):
        while True:
            parts = self._read()
            if not parts: break
            cmd, args = parts[0], parts[1:]

            if cmd == "UGO":
                self.my = 0
                move = self._choose(float(args[0]))
                self._send(f"IDO {move[0]} {move[1]}")

            elif cmd == "HEDID":
                timeout = float(args[0])
                x, y = int(args[-2]), int(args[-1])
                opp_move = None if (x, y) == (-1, -1) else (x, y)
                if self.my is None:
                    self.my = 1                    # jestem drugim graczem
                self._apply_move(1 - self.my, opp_move)
                move = self._choose(timeout)
                self._send(f"IDO {move[0]} {move[1]}")

            elif cmd == "ONEMORE":
                self.reset()
                self._send("RDY")

            elif cmd == "BYE":
                break

# ─────────────────────────────────────────────────────────────────────────────
# 3.  start
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Player().loop()
