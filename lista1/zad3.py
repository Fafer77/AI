# zauwazmy ze za kazdym razem gdy mamy ten sam uklad to wygrywa figurant, bo bedzie mial wiekszego high carda
# wystarczy w duzej petli losowac karty dla figuranta i blotkarza i patrzec kto wygral. Zgodnie z prawem
# wielkich liczb bedzie dazyc do realnego prawodpodobienstwa wygranej (chyba to sie nazywa Monte Carlo)

import numpy as np
from collections import Counter

CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}


def is_straigth(hand):
    return all(hand[i] + 1 == hand[i + 1] for i in range(0, len(hand)-1))


def is_flush(hand):
    return len(set(hand)) == 1


def evaluate_hand(hand):
    values = sorted([CARD_RANKS[r] for r, c in hand])
    suits = [c for r, c in hand]

    # royal flush
    if values[4] == 14 and is_flush(suits) and is_straigth(values):
        return 10

    # straight flush
    if is_flush(suits) and is_straigth(values):
        return 9

    # flush
    if is_flush(suits):
        return 6

    # straight
    if is_straigth(values):
        return 5
    
    counter = Counter(values)
    
    is_three = False
    is_pair = 0

    for _, count in counter.items():
        if count == 4:
            return 8
        elif count == 3:
            is_three = True
        elif count == 2:
            is_pair += 1
    
    if is_three and is_pair:
        return 7
    elif is_three:
        return 4
    elif is_pair == 2:
        return 3
    elif is_pair == 1:
        return 2
    else:
        return 1


def poker_probability(figurant, blotkarz):
    blotkarz_wins = 0

    for _ in range(100000):
        # permutation
        figurant_cards_idx = np.random.choice(len(figurant), size=5, replace=False)
        figurant_hand = figurant[figurant_cards_idx]

        blotkarz_cards_idx = np.random.choice(len(blotkarz), size=5, replace=False)
        blotkarz_hand = blotkarz[blotkarz_cards_idx]

        blotkarz_pattern = evaluate_hand(blotkarz_hand)
        figurant_pattern = evaluate_hand(figurant_hand)

        if blotkarz_pattern > figurant_pattern:
            blotkarz_wins += 1

    return blotkarz_wins / 100000 * 100


if __name__ == '__main__':
    suits = ['S', 'D', 'H', 'C']
    figurant = ['A', 'K', 'Q', 'J']
    blotkarz = [str(i) for i in range(2, 11)]

    figurant_deck = np.array([(rank, suit) for rank in figurant for suit in suits], dtype=object)
    blotkarz_deck = np.array([(rank, suit) for rank in blotkarz for suit in suits], dtype=object)

    blotkarz_winner = np.array([(rank, 'S') for rank in blotkarz])

    print(poker_probability(figurant_deck, blotkarz_deck))
    print(poker_probability(figurant_deck, blotkarz_winner))
