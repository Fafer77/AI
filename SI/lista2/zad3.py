from typing import List
import heapq
from collections import deque

def bfs_preprocess(maze: List[List[str]], goal_points: List[tuple[int, int]]) -> List[List[int]]:
    # visited nodes are marked with their dist being other than -1
    height = len(maze)
    width = len(maze[0])
    queue = deque()
    heuristic_maze = [[-1 for _ in range(width)] for _ in range(height)]

    for x, y in goal_points:
        heuristic_maze[x][y] = 0
        queue.append((x, y))

    while queue:
        x, y = queue.popleft()

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_x, new_y = x + dx, y + dy
            if 1 <= new_x <= height - 1 and 1 <= new_y <= width:
                if maze[new_x][new_y] != '#' and heuristic_maze[new_x][new_y] == -1:
                    heuristic_maze[new_x][new_y] = heuristic_maze[x][y] + 1
                    queue.append((new_x, new_y))

    return heuristic_maze


def find_starts(maze: List[List[str]]) -> List[tuple[int, int]]:
    starts = []
    for i in range(1, len(maze) - 1):
        for j in range(1, len(maze[0]) - 1):
            if maze[i][j] in ('S', 'B'):
                starts.append((i, j))
    return starts


def find_goals(maze: List[List[str]]) -> List[tuple[int, int]]:
    goals = []
    for i in range(1, len(maze) - 1):
        for j in range(1, len(maze[0]) - 1):
            if maze[i][j] in ('G', 'B'):
                goals.append((i, j))
    return goals


def calculate_h_value(state: set, heuristic_maze: List[List[int]]) -> float:
    if not state:
        return 0
    return max(heuristic_maze[x][y] for x, y in state)


def create_new_state(state: frozenset, move: tuple[int, int], maze: List[List[str]],
        heuristic_maze: List[List[int]]) -> tuple[frozenset, float]:
    new_state = set()
    for x, y in state:
        dx, dy = move
        if maze[x + dx][y + dy] != '#':
            new_state.add((x + dx, y + dy))
        else:
            new_state.add((x, y))
    
    h_value = calculate_h_value(new_state, heuristic_maze)
    return frozenset(new_state), h_value


def find_winning_path(maze: List[List[str]], heuristic_maze: List[List[int]], 
                      start_points: List[tuple[int, int]]) -> List[str]:
    start_state = frozenset(start_points)

    pq = []
    start_h = calculate_h_value(start_state, heuristic_maze)
    heapq.heappush(pq, (start_h, start_state, []))

    visited = {}

    directions = {
        (1, 0): 'D',
        (-1, 0): 'U',
        (0, 1): 'R',
        (0, -1): 'L'
    }

    while pq:
        f, state, path = heapq.heappop(pq)
        g = len(path)

        if state in visited and visited[state] <= g:
            continue

        visited[state] = g

        if calculate_h_value(state, heuristic_maze) == 0:
            return path

        for move, direction in directions.items():
            new_state, h_value = create_new_state(state, move, maze, heuristic_maze)
            new_path = path + [direction]
            new_g = len(new_path)
            new_f = new_g + h_value

            if new_state not in visited or visited[new_state] > new_g:
                heapq.heappush(pq, (new_f, new_state, new_path))

    return []


if __name__ == '__main__':
    with open('zad_input.txt', 'r') as f:
        maze = [list(line.strip()) for line in f]

    start_points = find_starts(maze)
    goal_points = find_goals(maze)

    heuristic_maze = bfs_preprocess(maze, goal_points)

    path = find_winning_path(maze, heuristic_maze, start_points)
    
    with open('zad_output.txt', 'w') as f:
        f.write(''.join(path))
