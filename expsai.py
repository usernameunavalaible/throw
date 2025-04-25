#A-SEARCH

def misplaced_tiles(state, goal):
    return sum(state[i][j] != goal[i][j] and state[i][j] != 0 for i in range(3) for j in range(3))

def get_neighbors(state):
    x, y = [(i, row.index(0)) for i, row in enumerate(state) if 0 in row][0]
    moves = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}
    neighbors = []
    for move, (dx, dy) in moves.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [r[:] for r in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append((move, new_state))
    return neighbors

def a_star_search(initial, goal):
    open_list = []
    h = misplaced_tiles(initial, goal)
    open_list.append((h, 0, h, initial, []))
    visited = set()

    while open_list:
        # Manually find and pop the lowest f(n) node
        open_list.sort()
        f, g, h, state, path = open_list.pop(0)
        state_tuple = tuple(map(tuple, state))
        if state == goal:
            return path, g, misplaced_tiles(initial, goal), g + misplaced_tiles(initial, goal)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for move, neighbor in get_neighbors(state):
            neighbor_tuple = tuple(map(tuple, neighbor))
            if neighbor_tuple not in visited:
                new_g = g + 1
                new_h = misplaced_tiles(neighbor, goal)
                new_path = path + [(move, neighbor, new_g, new_h)]
                open_list.append((new_g + new_h, new_g, new_h, neighbor, new_path))
    return None, 0, 0, 0

def print_solution(path, initial, g, h, f):
    p = lambda row: " ".join(str(x) if x else " " for x in row)
    print("Initial State:")
    print(f"g(n): {g} | h(n): {h} | f(n): {f}")
    for row in initial: print(p(row))
    print("\nSolution Steps:")
    for move, state, g, h in path:
        print(f"\nMove: {move} | g(n): {g} | h(n): {h} | f(n): {g + h}")
        for row in state: print(p(row))
    print("\nGoal Reached!")

def read_state(prompt):
    print(prompt)
    return [list(map(int, input(f"Enter row {i+1} : ").split())) for i in range(3)]

initial = read_state("Enter the Initial State :")
goal = read_state("\nEnter the Goal State :")

path, g, h, f = a_star_search(initial, goal)
print_solution(path, initial, g, h, f) if path else print("No solution found.")

#BFS-DFS

def bfs(adj, n, start):
    visited = [False] * n
    q = [start]
    visited[start] = True
    print(f"BFS starting from vertex {start}: ", end="")
    while q:
        v = q.pop(0)
        print(v, end=" ")
        for i in range(n):
            if adj[v][i] and not visited[i]:
                visited[i] = True
                q.append(i)
    print()

def dfs(adj, n, start):
    visited = [False] * n
    print(f"DFS starting from vertex {start}: ", end="")
    def visit(v):
        visited[v] = True
        print(v, end=" ")
        for i in range(n):
            if adj[v][i] and not visited[i]:
                visit(i)
    visit(start)
    print()

def main():
    n = int(input("Enter number of vertices: "))
    print("Enter the adjacency matrix row by row:")
    adj = [list(map(int, input().split())) for _ in range(n)]
    start = int(input("Enter starting vertex for BFS and DFS: "))
    bfs(adj, n, start)
    dfs(adj, n, start)

if __name__ == "__main__":
    main()

#Bayes Network

# Priors
P_B, P_E = 0.001, 0.002
P_notB, P_notE = 1 - P_B, 1 - P_E

# Conditional Probabilities
P_A = {(1,1): 0.95, (1,0): 0.94, (0,1): 0.29, (0,0): 0.001}
P_J = {1: 0.9, 0: 0.05}
P_M = {1: 0.7, 0: 0.01}

def compute_joint_probability(b, e, a, j, m):
    pb = P_B if b else P_notB
    pe = P_E if e else P_notE
    pa = P_A[(b, e)] if a else 1 - P_A[(b, e)]
    pj = P_J[a] if j else 1 - P_J[a]
    pm = P_M[a] if m else 1 - P_M[a]
    return pb * pe * pa * pj * pm

def main():
    print("Welcome to the Burglary Alarm Joint Probability Calculator!\n")
    get = lambda q: input(q).strip().lower() == 'yes'
    b, e, a, j, m = map(get, [
        "Did a burglary occur? (yes/no): ",
        "Did an earthquake occur? (yes/no): ",
        "Did the alarm sound? (yes/no): ",
        "Did John call? (yes/no): ",
        "Did Mary call? (yes/no): "
    ])
    jp = compute_joint_probability(b, e, a, j, m)
    print(f"\nThe joint probability of the observed events is: {jp:.8f}")

if __name__ == "__main__":
    main()

#MINIMAX

import math

P, O, E = 'X', 'O', '_'

def print_board(b): print('\n'.join(' '.join(r) for r in b), '\n')

def moves_left(b): return any(E in r for r in b)

def evaluate(b):
    for i in range(3):
        if b[i][0] == b[i][1] == b[i][2] != E: return 1 if b[i][0] == P else -1
        if b[0][i] == b[1][i] == b[2][i] != E: return 1 if b[0][i] == P else -1
    if b[0][0] == b[1][1] == b[2][2] != E: return 1 if b[0][0] == P else -1
    if b[0][2] == b[1][1] == b[2][0] != E: return 1 if b[0][2] == P else -1
    return 0

def minimax(b, d, max_turn):
    score = evaluate(b)
    if score or not moves_left(b): return score
    best = -math.inf if max_turn else math.inf
    for i in range(3):
        for j in range(3):
            if b[i][j] == E:
                b[i][j] = P if max_turn else O
                val = minimax(b, d+1, not max_turn)
                b[i][j] = E
                best = max(best, val) if max_turn else min(best, val)
    return best

def best_move(b):
    move, best_val = (-1, -1), -math.inf
    for i in range(3):
        for j in range(3):
            if b[i][j] == E:
                b[i][j] = P
                val = minimax(b, 0, False)
                b[i][j] = E
                print(f"Position ({i},{j}) has utility: {val}")
                if val > best_val: move, best_val = (i, j), val
    return move

def main():
    b = [[E]*3 for _ in range(3)]
    print("Initial Board:"); print_board(b)
    while moves_left(b) and not evaluate(b):
        x, y = best_move(b); b[x][y] = P
        print("AI plays:"); print_board(b)
        if evaluate(b) or not moves_left(b): break
        try:
            r, c = map(int, input("Enter your move (row col): ").split())
            if not (0 <= r < 3 and 0 <= c < 3) or b[r][c] != E: raise ValueError
            b[r][c] = O
        except: print("Invalid move. Try again."); continue
        print("After your move:"); print_board(b)
        if evaluate(b) or not moves_left(b): break
    print("AI wins!" if evaluate(b) == 1 else "You win!" if evaluate(b) == -1 else "It's a draw!")

if __name__ == "__main__":
    main()

#POP

def apply_operator(state, op):
    return ((state - op['del_effects']) | op['add_effects'], True) if op['preconditions'].issubset(state) else (state, False)

def achieve_goal(state, goal, operators):
    return next((op for op in operators if goal in op['add_effects']), None)

def resolve_threats(plan):
    print("Checking for and resolving threats in the plan...")
    return plan

def plan_steps(init_state, goal_state, operators):
    state, plan = set(init_state), []
    print("Initial State:", state)
    for goal in goal_state:
        if goal not in state:
            op = achieve_goal(state, goal, operators)
            if op:
                print(f"Applying: {op['name']}")
                plan.append(op['name'])
                state, ok = apply_operator(state, op)
                if not ok: print("Failed to apply operator for", goal)
                print("New State:", state)
    plan = resolve_threats(plan)
    print("Final Plan (after threat resolution):", plan)

# Example usage
parse = lambda s: set(s.replace(" ", "").replace("):", ")").split(','))
init_state = parse(input("Enter initial state: "))
goal_state = parse(input("Enter goal state: "))

operators = [
    {'name': 'Move(C, A, Table)', 'preconditions': {'On(C, A)', 'Clear(C)', 'Clear(Table)'}, 'add_effects': {'On(C, Table)', 'Clear(A)'}, 'del_effects': {'On(C, A)'}},
    {'name': 'Move(B, Table, C)', 'preconditions': {'On(B, Table)', 'Clear(B)', 'Clear(C)'}, 'add_effects': {'On(B, C)', 'Clear(Table)'}, 'del_effects': {'On(B, Table)'}},
    {'name': 'Move(A, Table, B)', 'preconditions': {'On(A, Table)', 'Clear(A)', 'Clear(B)'}, 'add_effects': {'On(A, B)', 'Clear(Table)'}, 'del_effects': {'On(A, Table)'}}
]

plan_steps(init_state, goal_state, operators)

#VACCUM CLEANER

def print_rooms(rooms, vacuum_location):
    for i, status in enumerate(rooms):
        tag = " (V)" if i == vacuum_location else ""
        print(f"Room {'A' if i == 0 else 'B'}: {status}{tag}")

def main():
    rooms = [
        input("Enter the status of Room A (C for Clean, D for Dirty): ").strip().upper(),
        input("Enter the status of Room B (C for Clean, D for Dirty): ").strip().upper()
    ]

    if any(r not in ('C', 'D') for r in rooms):
        print("Invalid input! Please restart the program and use 'C' or 'D'.")
        return

    pos = input("Enter the initial position of the vacuum cleaner (A or B): ").strip().upper()
    vacuum = 0 if pos == 'A' else 1 if pos == 'B' else -1

    if vacuum == -1:
        print("Invalid input! Please restart the program and use 'A' or 'B'.")
        return

    print("Initial State:")
    print_rooms(rooms, vacuum)

    for step in range(5):
        print(f"\nStep {step + 1}:")
        if rooms[vacuum] == 'D':
            print(f"Cleaning Room {'A' if vacuum == 0 else 'B'}")
            rooms[vacuum] = 'C'
        else:
            print(f"Room {'A' if vacuum == 0 else 'B'} is already clean.")
            vacuum = 1 - vacuum
            print(f"Moving to Room {'A' if vacuum == 0 else 'B'}")
        print_rooms(rooms, vacuum)
        if rooms == ['C', 'C']:
            print("\nBoth rooms are clean. Ending function.")
            break

if __name__ == "__main__":
    main()

% Facts for male
male(jerome).
male(victor).
male(austin).
male(bryn).
male(sconny).
male(ryan).
male(tom).

% Facts for female
female(philomena).
female(loretta).
female(noella).
female(taylor).
female(araina).

% Facts for mother
mother(philomena, austin).
mother(philomena, taylor).
mother(philomena, bryn).
mother(loretta, sconny).
mother(sconny, ryan).
mother(sconny, tom).
mother(noella, araina).

% Facts for father
father(jerome, austin).
father(jerome, taylor).
father(jerome, bryn).
father(victor, sconny).
father(austin, ryan).
father(austin, tom).
father(bryn, araina).

% Parent rule
parent(X, Y) :- mother(X, Y).
parent(X, Y) :- father(X, Y).

% Siblings rule
siblings(X, Y) :- 
    parent(P, X),
    parent(P, Y),
    X \= Y.

% Brother rule
brother(X, Y) :- 
    male(X),  % X must be male
    siblings(X, Y).

% Sister rule
sister(X, Y) :- 
    female(X),  % X must be female
    siblings(X, Y).

% Uncle rule
uncle(X, Y) :- 
    male(X),            % X must be male
    parent(P, Y),       % P is Y's parent
    siblings(X, P).     % X is sibling of P

uncle(X, Y) :-          % Uncle by marriage (parent's sibling's husband)
    parent(P, Y),       % P is Y's parent
    siblings(A, P),     % A is sibling of P
    married(X, A),      % X is married to A
    male(X).            % Ensure X is male

% Aunt rule
aunt(X, Y) :- 
    parent(P, Y),  % P is Y's parent
    siblings(X, P), % X is P's sibling
    female(X).  % X is female
aunt(X, Y) :-     % Aunt by marriage (any parent's sibling's wife)
    married(U, X), % X is married to U
    parent(P, Y),  % P is Y's parent
    siblings(U, P). % U is sibling of P

% Grandfather rule
grandfather(X, Y) :- 
    father(X, P),
    parent(P, Y).

% Grandmother rule
grandmother(X, Y) :- 
    mother(X, P),
    parent(P, Y).

% Cousins rule
cousins(X, Y) :- 
    parent(P1, X),
    parent(P2, Y),
    siblings(P1, P2),
    X \= Y.

% Married rule
married(jerome, philomena).
married(victor, loretta).
married(austin, sconny).
married(bryn, noella).
