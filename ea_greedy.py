import pygame
import random
import heapq
import sys

# ============== CONFIGURATION ==============
GRID_SIZE = 15
CELL_SIZE = 40
MARGIN = 200
WIDTH = GRID_SIZE * CELL_SIZE + MARGIN
HEIGHT = GRID_SIZE * CELL_SIZE + 60

SHELVES = [
    (3,2),(3,3),(3,4),(3,5),(7,2),(7,3),(7,4),(7,5),
    (11,2),(11,3),(11,4),(11,5),(3,9),(3,10),(3,11),(3,12),
    (7,9),(7,10),(7,11),(7,12),(11,9),(11,10),(11,11),(11,12),
]

PACKAGES = [
    {'id': 1, 'x': 1, 'y': 3, 'visits': 2},
    {'id': 2, 'x': 5, 'y': 5, 'visits': 1},
    {'id': 3, 'x': 9, 'y': 3, 'visits': 3},
    {'id': 4, 'x': 13, 'y': 5, 'visits': 1},
    {'id': 5, 'x': 1, 'y': 11, 'visits': 2},
    {'id': 6, 'x': 5, 'y': 10, 'visits': 1},
    {'id': 7, 'x': 9, 'y': 11, 'visits': 2},
    {'id': 8, 'x': 13, 'y': 10, 'visits': 1},
]

DEPOSIT = {'x': 7, 'y': 7}

ROBOTS_INIT = [
    {'id': 1, 'x': 0, 'y': 0, 'color': (231, 76, 60)},
    {'id': 2, 'x': 14, 'y': 0, 'color': (52, 152, 219)},
    {'id': 3, 'x': 0, 'y': 14, 'color': (46, 204, 113)},
]

ROBOT_CAPACITY = 3

# Colors
BG_COLOR = (26, 26, 46)
GRID_COLOR = (51, 51, 51)
SHELF_COLOR = (139, 69, 19)
DEPOSIT_COLOR = (243, 156, 18)
PACKAGE_COLOR = (155, 89, 182)
TEXT_COLOR = (255, 255, 255)


# ============== A* PATHFINDING ==============
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(node):
    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    neighbors = []
    for d in dirs:
        nx, ny = node[0] + d[0], node[1] + d[1]
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if (nx, ny) not in SHELVES:
                neighbors.append((nx, ny))
    return neighbors

def a_star(start, goal):
    if start == goal:
        return [start]
    open_set = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for neighbor in get_neighbors(current):
            tent_g = g_score[current] + 1
            if neighbor not in g_score or tent_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tent_g
                heapq.heappush(open_set, (tent_g + heuristic(neighbor, goal), tent_g, neighbor))
    return []


# ============== DISTANCE MATRIX ==============
def build_distance_matrix(locations):
    n = len(locations)
    matrix = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0
        for j in range(i + 1, n):
            path = a_star(locations[i], locations[j])
            dist = len(path) - 1 if path else 9999
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix


# ============== GREEDY MTMV SOLVER ==============
def solve_greedy_mtmv(robots, packages, deposit):
    locations = [(deposit['x'], deposit['y'])]
    locations += [(p['x'], p['y']) for p in packages]
    locations += [(r['x'], r['y']) for r in robots]
    
    dist_matrix = build_distance_matrix(locations)
    
    m = len(robots)
    deposit_idx = 0
    robot_start_indices = [len(packages) + 1 + i for i in range(m)]
    
    # Track visits remaining per package
    visits_remaining = [p['visits'] for p in packages]
    # Track which robots visited which packages
    robot_visited = [set() for _ in range(m)]
    # Robot tours (package IDs)
    tours = [[] for _ in range(m)]
    # Robot current distances
    robot_dist = [0] * m
    # Robot current positions
    robot_pos = robot_start_indices[:]
    
    total_visits = sum(p['visits'] for p in packages)
    assigned_visits = 0
    
    while assigned_visits < total_visits:
        best_robot, best_pkg, best_makespan = -1, -1, float('inf')
        
        for r in range(m):
            for p in range(len(packages)):
                if visits_remaining[p] <= 0 or p in robot_visited[r]:
                    continue
                
                pkg_loc_idx = p + 1
                dist_to_pkg = dist_matrix[robot_pos[r]][pkg_loc_idx]
                dist_to_dep = dist_matrix[pkg_loc_idx][deposit_idx]
                
                new_dist = robot_dist[r] + dist_to_pkg + dist_to_dep
                new_makespan = max(new_dist if i == r else robot_dist[i] for i in range(m))
                
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    best_robot, best_pkg = r, p
        
        if best_robot == -1:
            break
        
        tours[best_robot].append(packages[best_pkg]['id'])
        robot_visited[best_robot].add(best_pkg)
        visits_remaining[best_pkg] -= 1
        assigned_visits += 1
        
        pkg_loc_idx = best_pkg + 1
        robot_dist[best_robot] += dist_matrix[robot_pos[best_robot]][pkg_loc_idx] + dist_matrix[pkg_loc_idx][deposit_idx]
        robot_pos[best_robot] = deposit_idx
    
    makespan = max(robot_dist)
    return tours, makespan


# ============== SIMULATION CLASS ==============
class Simulation:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.robots = [{'id': r['id'], 'x': r['x'], 'y': r['y'], 'color': r['color'],
                        'tour': [], 'tour_idx': 0, 'path': [], 'carrying': [],
                        'phase': 'idle', 'waiting': 0} for r in ROBOTS_INIT]
        self.packages = [{'id': p['id'], 'x': p['x'], 'y': p['y'], 
                          'visits': p['visits'], 'visits_done': 0} for p in PACKAGES]
        self.solved = False
        self.makespan = 0
        self.logs = []
        self.running = False
        self.speed = 5
    
    def log(self, msg):
        self.logs.append(msg)
        if len(self.logs) > 12:
            self.logs.pop(0)
    
    def solve_greedy(self):
        self.log("Building distance matrix...")
        self.log("Solving MTMV Greedy...")
        
        tours, makespan = solve_greedy_mtmv(ROBOTS_INIT, PACKAGES, DEPOSIT)
        self.makespan = makespan
        
        for r in range(len(self.robots)):
            self.robots[r]['tour'] = tours[r]
            self.robots[r]['phase'] = 'toPackage' if tours[r] else 'idle'
        
        self.solved = True
        self.log(f"Greedy done! Makespan: {makespan}")
        for r in self.robots:
            self.log(f"R{r['id']}: {r['tour']}")
    
    def step(self):
        if not self.solved:
            return
        
        for r in self.robots:
            if r['waiting'] > 0:
                r['waiting'] -= 1
                continue
            
            # Assign path if needed
            if not r['path'] and r['phase'] != 'idle':
                if r['phase'] == 'toPackage' and r['tour_idx'] < len(r['tour']):
                    pkg = next((p for p in self.packages if p['id'] == r['tour'][r['tour_idx']]), None)
                    if pkg:
                        r['path'] = a_star((r['x'], r['y']), (pkg['x'], pkg['y']))[1:]
                elif r['phase'] == 'toDeposit':
                    r['path'] = a_star((r['x'], r['y']), (DEPOSIT['x'], DEPOSIT['y']))[1:]
            
            # Move
            if r['path']:
                next_pos = r['path'][0]
                # Collision check (skip idle robots)
                collision = False
                collider = None
                for other in self.robots:
                    if other['id'] == r['id'] or other['phase'] == 'idle':
                        continue
                    if (next_pos[0], next_pos[1]) == (other['x'], other['y']):
                        collision = True
                        collider = other
                        break
                
                if collision and r['id'] > collider['id']:
                    # Sidestep
                    sidestepped = False
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        side = (r['x'] + dx, r['y'] + dy)
                        if 0 <= side[0] < GRID_SIZE and 0 <= side[1] < GRID_SIZE:
                            if side not in SHELVES and not any(o['x']==side[0] and o['y']==side[1] for o in self.robots if o['id']!=r['id']):
                                r['x'], r['y'] = side
                                if r['phase'] == 'toDeposit':
                                    target = (DEPOSIT['x'], DEPOSIT['y'])
                                else:
                                    pkg = next((p for p in self.packages if p['id'] == r['tour'][r['tour_idx']]), None)
                                    target = (pkg['x'], pkg['y']) if pkg else (DEPOSIT['x'], DEPOSIT['y'])
                                r['path'] = a_star(side, target)[1:]
                                self.log(f"R{r['id']} sidestepped")
                                sidestepped = True
                                break
                    if not sidestepped:
                        r['waiting'] = 2
                else:
                    r['x'], r['y'] = next_pos
                    r['path'] = r['path'][1:]
            
            # Pickup
            if r['phase'] == 'toPackage' and not r['path'] and r['tour_idx'] < len(r['tour']):
                pkg = next((p for p in self.packages if p['id'] == r['tour'][r['tour_idx']]), None)
                if pkg and r['x'] == pkg['x'] and r['y'] == pkg['y']:
                    pkg['visits_done'] += 1
                    r['carrying'].append(pkg['id'])
                    self.log(f"R{r['id']} picked #{pkg['id']}")
                    r['tour_idx'] += 1
                    
                    if len(r['carrying']) >= ROBOT_CAPACITY or r['tour_idx'] >= len(r['tour']):
                        r['phase'] = 'toDeposit'
            
            # Deposit
            if r['phase'] == 'toDeposit' and not r['path']:
                if r['x'] == DEPOSIT['x'] and r['y'] == DEPOSIT['y']:
                    self.log(f"R{r['id']} delivered {r['carrying']}")
                    r['carrying'] = []
                    r['phase'] = 'toPackage' if r['tour_idx'] < len(r['tour']) else 'idle'
    
    def all_done(self):
        return all(p['visits_done'] >= p['visits'] for p in self.packages) and all(not r['carrying'] for r in self.robots)


# ============== PYGAME MAIN ==============
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MTMV Greedy Warehouse Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)
    
    sim = Simulation()
    frame_count = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if sim.solved:
                        sim.running = not sim.running
                elif event.key == pygame.K_r:
                    sim.reset()
                elif event.key == pygame.K_g and not sim.solved:
                    sim.solve_greedy()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    sim.speed = min(20, sim.speed + 1)
                elif event.key == pygame.K_MINUS:
                    sim.speed = max(1, sim.speed - 1)
                elif event.key == pygame.K_s and sim.solved:
                    sim.step()
        
        # Update
        if sim.running and not sim.all_done():
            frame_count += 1
            if frame_count >= (21 - sim.speed):
                sim.step()
                frame_count = 0
        
        # Draw
        screen.fill(BG_COLOR)
        
        # Grid
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pygame.draw.rect(screen, GRID_COLOR, (i*CELL_SIZE, j*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        
        # Shelves
        for sx, sy in SHELVES:
            pygame.draw.rect(screen, SHELF_COLOR, (sx*CELL_SIZE+2, sy*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4))
        
        # Deposit
        pygame.draw.rect(screen, DEPOSIT_COLOR, (DEPOSIT['x']*CELL_SIZE+2, DEPOSIT['y']*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4))
        txt = small_font.render("DEP", True, (0,0,0))
        screen.blit(txt, (DEPOSIT['x']*CELL_SIZE+8, DEPOSIT['y']*CELL_SIZE+12))
        
        # Packages
        for p in sim.packages:
            color = (50,50,50) if p['visits_done'] >= p['visits'] else PACKAGE_COLOR
            pygame.draw.rect(screen, color, (p['x']*CELL_SIZE+6, p['y']*CELL_SIZE+6, CELL_SIZE-12, CELL_SIZE-12))
            if p['visits'] > 1:
                pygame.draw.rect(screen, DEPOSIT_COLOR, (p['x']*CELL_SIZE+6, p['y']*CELL_SIZE+6, CELL_SIZE-12, CELL_SIZE-12), 2)
            txt = small_font.render(f"{p['id']}({p['visits_done']}/{p['visits']})", True, TEXT_COLOR)
            screen.blit(txt, (p['x']*CELL_SIZE+8, p['y']*CELL_SIZE+12))
        
        # Robot paths
        for r in sim.robots:
            if r['path']:
                points = [(r['x']*CELL_SIZE+CELL_SIZE//2, r['y']*CELL_SIZE+CELL_SIZE//2)]
                points += [(p[0]*CELL_SIZE+CELL_SIZE//2, p[1]*CELL_SIZE+CELL_SIZE//2) for p in r['path']]
                if len(points) > 1:
                    pygame.draw.lines(screen, r['color'], False, points, 2)
        
        # Robots
        for r in sim.robots:
            pygame.draw.circle(screen, r['color'], (r['x']*CELL_SIZE+CELL_SIZE//2, r['y']*CELL_SIZE+CELL_SIZE//2), CELL_SIZE//2-4)
            pygame.draw.circle(screen, TEXT_COLOR, (r['x']*CELL_SIZE+CELL_SIZE//2, r['y']*CELL_SIZE+CELL_SIZE//2), CELL_SIZE//2-4, 2)
            txt = font.render(str(r['id']), True, TEXT_COLOR)
            screen.blit(txt, (r['x']*CELL_SIZE+CELL_SIZE//2-5, r['y']*CELL_SIZE+CELL_SIZE//2-8))
            if r['carrying']:
                pygame.draw.circle(screen, PACKAGE_COLOR, (r['x']*CELL_SIZE+CELL_SIZE-8, r['y']*CELL_SIZE+8), 8)
                txt = small_font.render(str(len(r['carrying'])), True, TEXT_COLOR)
                screen.blit(txt, (r['x']*CELL_SIZE+CELL_SIZE-12, r['y']*CELL_SIZE+2))
        
        # UI Panel
        panel_x = GRID_SIZE * CELL_SIZE + 10
        pygame.draw.rect(screen, (40,40,60), (panel_x-5, 0, MARGIN, HEIGHT))
        
        y = 10
        screen.blit(font.render("MTMV Greedy Simulation", True, DEPOSIT_COLOR), (panel_x, y)); y += 30
        screen.blit(small_font.render("G = Solve Greedy", True, TEXT_COLOR), (panel_x, y)); y += 20
        screen.blit(small_font.render("SPACE = Start/Pause", True, TEXT_COLOR), (panel_x, y)); y += 20
        screen.blit(small_font.render("S = Step", True, TEXT_COLOR), (panel_x, y)); y += 20
        screen.blit(small_font.render("R = Reset", True, TEXT_COLOR), (panel_x, y)); y += 20
        screen.blit(small_font.render("+/- = Speed", True, TEXT_COLOR), (panel_x, y)); y += 30
        
        screen.blit(font.render(f"Speed: {sim.speed}", True, TEXT_COLOR), (panel_x, y)); y += 25
        screen.blit(font.render(f"Makespan: {sim.makespan}", True, DEPOSIT_COLOR), (panel_x, y)); y += 25
        status = "DONE!" if sim.all_done() else ("Running" if sim.running else "Paused")
        screen.blit(font.render(f"Status: {status}", True, (46,204,113) if sim.all_done() else TEXT_COLOR), (panel_x, y)); y += 30
        
        screen.blit(font.render("Robot Tours:", True, TEXT_COLOR), (panel_x, y)); y += 20
        for r in sim.robots:
            txt = f"R{r['id']}: {r['tour']}"
            screen.blit(small_font.render(txt[:22], True, r['color']), (panel_x, y)); y += 18
        
        y += 10
        screen.blit(font.render("Log:", True, TEXT_COLOR), (panel_x, y)); y += 20
        for log in sim.logs[-8:]:
            screen.blit(small_font.render(log[:24], True, (180,180,180)), (panel_x, y)); y += 16
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()