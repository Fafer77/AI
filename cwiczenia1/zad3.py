'''
plan
using zad3 from set 1 output file with modified text
create algoirhtm to randomly split text and output it to file
compare both outputs: number of correct lines / all lines

Random algorithm:
use Trie
1. read words one by one from polish_words and insert them into Trie
2. 

'''
from typing import List
import random

BIGGEST_SQUARES_FILENAME = '/lista1/zad3_output.txt'
RANDOM_ALG_FILENAME = '/cwiczenia1/random_tadeusz.txt'


class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        curr = self.root

        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.end_of_word = True
    
    def search(self, word: str) -> bool:
        curr = self.root

        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        
        return curr.end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        curr = self.root

        for c in prefix:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        
        return True


def prefixes_list(word: str, trie_root: Trie) -> List[str]:
    result = []
    for i in range(1, len(word) + 1):
        if trie_root.search(word[:i]):
            result.append(word[:i])
    return result


def random_split_line(line: str, trie: Trie) -> List[str] | None:
    def backtrack(i: int, result: List[str]) -> bool:
        if i == len(line):
            return True
        
        prefix_list = prefixes_list(line[i:], trie)
        random.shuffle(prefix_list)

        for prefix in prefix_list:
            result.append(prefix)
            if backtrack(i + len(prefix), result):
                return True
            result.pop()
        
        return False

    result = []
    flag = backtrack(0, result)
    return result if flag else None


def random_split_words(lines: List[str], trie: Trie) -> List[List[str]]:
    split_lines = []
    for line in lines:
        line = line.strip()
        splitted = random_split_line(line, trie)
        if splitted is None:
            split_lines.append('IMPOSSIBLE TO SPLIT')
        else:
            split_lines.append(splitted)

    with open('cwiczenia1/random_tadeusz.txt', 'w') as f:
        for line in split_lines:
            f.write(' '.join(line) + '\n')

    return split_lines


def measure_accuracy(random_lines: List[List[str]], original_lines: List[str], 
                    squares_lines: List[str]) -> tuple[float, float]:
    random_points = 0
    squares_points = 0
    random_lines = [' '.join(line) for line in random_lines]

    for i in range(9944):
        original_line = original_lines[i]
        if original_line == random_lines[i]:
            random_points += 1
        if original_line == squares_lines[i]:
            squares_points += 1

    print(random_points)
    return (random_points / 9944 * 100, squares_points / 9944 * 100)


if __name__ == "__main__":
    trie_root = Trie()
    with open('lista1/polish_words.txt', 'r') as f:
        words = f.readlines()
        for word in words:
            trie_root.insert(word.strip())

    with open('lista1/pan_tadeusz_bez_spacji.txt', 'r') as f:
        lines = f.readlines()

    random_lines = random_split_words(lines, trie_root)

    with open('pan_tadeusz_max_squares.txt', 'r') as f:
        squares_lines = f.readlines()
        squares_lines = [line.strip() for line in squares_lines]
    
    with open('cwiczenia1/tadeusz-clean.txt', 'r') as f:
        original_lines = f.readlines()
        original_lines = [line.strip() for line in original_lines]

    random_accuracy, squares_accuracy = measure_accuracy(random_lines, original_lines, squares_lines)
    print(f'random accuracy: {random_accuracy}, squares accuracy: {squares_accuracy}')
