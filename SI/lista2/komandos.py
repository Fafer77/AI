from typing import List
from collections import deque


def create_new_state(state: frozenset[tuple[int, int]], move: tuple[int, int], maze: List[List[str]]) -> frozenset:
    new_state = set()
    for x, y in state:
        dx, dy = move
        if maze[x + dx][y + dy] != '#':
            new_state.add((x + dx, y + dy))
        else:
            new_state.add((x, y))
    return frozenset(new_state)


def preprocess_states(maze: List[List[str]]) -> tuple[frozenset[tuple[int, int]], List[str]]:
    state = set()
    n = len(maze)
    m = len(maze[0])

    for i in range(1, n - 1):
        for j in range(1, m - 1):    
            if maze[i][j] in ('S', 'B'):
                k = i
                l = j

                while maze[k-1][j] in ('S', 'B', 'G'):
                    k -= 1

                while maze[k][l-1] in ('S', 'B', 'G'):
                    l -= 1

                for _ in range(m):
                    if maze[k+1][l] != '#':
                        k += 1
                    if maze[k][l+1] != '#':
                        l += 1

                # left
                for _ in range(m):
                    if maze[k][l-1] != '#':
                        l -= 1
                # up
                for _ in range(n):
                    if maze[k-1][l] != '#':
                        k -= 1
                
                state.add((k, l))
    
    state = frozenset(state)
    
    return state


def winning_state(state: frozenset[tuple[int, int]], maze: List[List[str]]) -> bool:
    return all(maze[x][y] in ('B', 'G') for x, y in state)


def find_winning_path(start_state: frozenset[tuple[int, int]], maze: List[List[str]]) -> List[str]:
    queue = deque([(start_state, [])])
    visited = set()
    directions = {
        (1, 0): 'D',
        (-1, 0): 'U',
        (0, 1): 'R',
        (0, -1): 'L'
    }

    while queue:
        state, path = queue.popleft()
        # print(state)

        if winning_state(state, maze):
            return path

        visited.add(state)

        for move, direction in directions.items():
            new_state = create_new_state(state, move, maze)

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [direction]))

    return []


if __name__ == '__main__':
    maze = []
    with open('zad_input.txt', 'r') as f:
        maze = [list(line.strip()) for line in f]
    
    height = len(maze) - 2
    width = len(maze[0]) - 2

    preprocess_moves = 'U' * height + 'L' * width + 'DR' * width + 'L' * width + 'U' * height

    start_state = preprocess_states(maze)
    start_path = preprocess_moves

    path = find_winning_path(start_state, maze)
    final_path = start_path + ''.join(path)
    
    with open('zad_output.txt', 'w') as f:
        f.write(final_path)
    
