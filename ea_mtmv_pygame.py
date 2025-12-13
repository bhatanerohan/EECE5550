import pygame
import random
import heapq
import sys

# ============== CONFIGURATION ==============
# environment values
GRID_SIZE = 15
CELL_SIZE = 40
MARGIN = 200
WIDTH = GRID_SIZE * CELL_SIZE + MARGIN
HEIGHT = GRID_SIZE * CELL_SIZE + 60

# barrier locations
SHELVES = [
    (3,2),(3,3),(3,4),(3,5),(7,2),(7,3),(7,4),(7,5),
    (11,2),(11,3),(11,4),(11,5),(3,9),(3,10),(3,11),(3,12),
    (7,9),(7,10),(7,11),(7,12),(11,9),(11,10),(11,11),(11,12),
]

# package locations and visit requirements
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

# deposit box location
DEPOSIT = {'x': 7, 'y': 7}

# robot initialization. smaller id number means higher priority
ROBOTS_INIT = [
    {'id': 1, 'x': 0, 'y': 0, 'color': (231, 76, 60)},
    {'id': 2, 'x': 14, 'y': 0, 'color': (52, 152, 219)},
    {'id': 3, 'x': 0, 'y': 14, 'color': (46, 204, 113)},
]

# EA Parameters
POP_SIZE = 50
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
GENERATIONS = 200
ROBOT_CAPACITY = 3

# Colors
BG_COLOR = (26, 26, 46)
GRID_COLOR = (51, 51, 51)
SHELF_COLOR = (139, 69, 19)
DEPOSIT_COLOR = (243, 156, 18)
PACKAGE_COLOR = (155, 89, 182)
TEXT_COLOR = (255, 255, 255)


# ============== A* PATHFINDING ==============
# diagonal moves not taken into account
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# return list of neighbors that are not shelves or the border. other robots not detected here
def get_neighbors(node):
    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    neighbors = []
    for d in dirs:
        nx, ny = node[0] + d[0], node[1] + d[1]
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if (nx, ny) not in SHELVES:
                neighbors.append((nx, ny))
    return neighbors

# path find with A*
def a_star(start, goal):
    # check s and g are not same location
    if start == goal:
        return [start]
    
    # priority queue
    open_set = [(heuristic(start, goal), 0, start)]
    came_from = {} # reconstruction path
    g_score = {start: 0} # cost from start to node
    
    while open_set:
        _, g, current = heapq.heappop(open_set)
        
        # reconstruct and return path when goal found
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        # check neighboring positions
        for neighbor in get_neighbors(current):
            tent_g = g_score[current] + 1
            # update if a better path found
            if neighbor not in g_score or tent_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tent_g
                heapq.heappush(open_set, (tent_g + heuristic(neighbor, goal), tent_g, neighbor))
    return [] # no path found


# ============== DISTANCE MATRIX ==============
# contains the distance between all package locations, the deposit zone, and robot initial positions
# used to prevent redundant calculations
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


# ============== EA FUNCTIONS ==============
# returns length of path of slowest robot (makespan)
# returns fitness, 1/(max_dist+1), where the smaller the max_dist, the higher the fitness
def calc_fitness(individual, dist_matrix, num_pkgs, num_robots):
    deposit_idx = 0
    max_dist = 0 # track longest path (makespan)
    for r in range(num_robots):
        tour = individual[r]
        if not tour:
            continue
        robot_start_idx = num_pkgs + 1 + r
        dist, pos, carried = 0, robot_start_idx, 0
        # simulate tour
        for pkg_idx in tour:
            pkg_loc_idx = pkg_idx + 1
            dist += dist_matrix[pos][pkg_loc_idx]
            pos = pkg_loc_idx
            carried += 1
            # return to deposit at end of tour or capacity reached
            if carried >= ROBOT_CAPACITY or pkg_idx == tour[-1]:
                dist += dist_matrix[pos][deposit_idx]
                pos = deposit_idx
                carried = 0
        max_dist = max(max_dist, dist)
    return {'makespan': max_dist, 'fitness': 1 / (max_dist + 1)}

# initialize solution for EA population
def random_individual(packages, num_robots):
    individual = [[] for _ in range(num_robots)]
    for pkg_idx, pkg in enumerate(packages):
        robots_needed = min(pkg['visits'], num_robots)
        robot_order = list(range(num_robots))
        random.shuffle(robot_order)
        # distribute visits across different robots 
        for v in range(robots_needed):
            individual[robot_order[v]].append(pkg_idx)
    # randomize order in tours
    for r in range(num_robots):
        random.shuffle(individual[r])
    return individual

# creates solution array with initializer function for EA population
def init_population(pop_size, packages, num_robots):
    return [random_individual(packages, num_robots) for _ in range(pop_size)]

# choose parents at random with probability determined by fitness (roulette wheel)
def select_parent(population, fitnesses):
    total_fit = sum(f['fitness'] for f in fitnesses)
    r = random.random() * total_fit
    for i, f in enumerate(fitnesses):
        r -= f['fitness']
        if r <= 0:
            return population[i]
    return population[-1]

# Exchange path segments between parent individuals at random crossover points
def crossover(p1, p2, num_robots):
    if random.random() > CROSSOVER_RATE:
        return [list(t) for t in p1], [list(t) for t in p2]
    c1, c2 = [list(t) for t in p1], [list(t) for t in p2]
    r = random.randint(0, num_robots - 1)
    if not c1[r] or not c2[r]:
        return c1, c2
    point = random.randint(0, min(len(c1[r]), len(c2[r])) - 1)
    c1[r], c2[r] = c1[r][:point] + c2[r][point:], c2[r][:point] + c1[r][point:]
    return c1, c2

# ensure solution is valid by moving repeats in a tour and adding locations to reach visit requirements
def repair_mtmv(individual, packages, num_robots):
    result = [[] for _ in range(num_robots)]
    visit_count = [0] * len(packages)
    robot_visited = [set() for _ in range(num_robots)]

    # copy valid visits to output, ignore repeats in a tour
    for r in range(num_robots):
        for pkg_idx in individual[r]:
            if 0 <= pkg_idx < len(packages):
                if pkg_idx not in robot_visited[r] and visit_count[pkg_idx] < packages[pkg_idx]['visits']:
                    result[r].append(pkg_idx)
                    robot_visited[r].add(pkg_idx)
                    visit_count[pkg_idx] += 1

    # add locations to first availability if visit requirement not met
    for pkg_idx, pkg in enumerate(packages):
        while visit_count[pkg_idx] < pkg['visits']:
            for r in range(num_robots):
                if pkg_idx not in robot_visited[r]:
                    result[r].append(pkg_idx)
                    robot_visited[r].add(pkg_idx)
                    visit_count[pkg_idx] += 1
                    break
            else:
                break
    return result

# Randomly swap elements between different robots’ paths with probability Pm
# limited to at most n/4 swaps per mutation
def mutate(individual, num_robots):
    if random.random() > MUTATION_RATE:
        return individual
    result = [list(t) for t in individual]
    
    # perform multiple swaps
    for _ in range(random.randint(1, 3)):
        r1, r2 = random.randint(0, num_robots-1), random.randint(0, num_robots-1)
        if not result[r1]:
            continue
        i1 = random.randint(0, len(result[r1]) - 1)
        # swap within tour
        if r1 == r2 and len(result[r1]) > 1:
            i2 = random.randint(0, len(result[r1]) - 1)
            result[r1][i1], result[r1][i2] = result[r1][i2], result[r1][i1]
        # swap between tours
        elif r1 != r2:
            pkg = result[r1].pop(i1)
            result[r2].insert(random.randint(0, len(result[r2])), pkg)
    return result

# evolves population of solutions
# get fitness of current population
# preserve current best solution
# generate offspring with select_parent, crossover, mutate and repair functions
# return new population
def evolve(population, dist_matrix, packages, num_robots):
    fitnesses = [calc_fitness(ind, dist_matrix, len(packages), num_robots) for ind in population]
    best_idx = min(range(len(population)), key=lambda i: fitnesses[i]['makespan'])
    new_pop = [[list(t) for t in population[best_idx]]]
    
    while len(new_pop) < len(population):
        p1, p2 = select_parent(population, fitnesses), select_parent(population, fitnesses)
        c1, c2 = crossover(p1, p2, num_robots)
        c1 = repair_mtmv(mutate(c1, num_robots), packages, num_robots)
        c2 = repair_mtmv(mutate(c2, num_robots), packages, num_robots)
        new_pop.extend([c1, c2] if len(new_pop) < len(population) - 1 else [c1])
    return new_pop[:len(population)]

# returns solution with lowest makespan in population and its makespan
def get_best(population, dist_matrix, packages, num_robots):
    best, best_makespan = None, float('inf')
    for ind in population:
        m = calc_fitness(ind, dist_matrix, len(packages), num_robots)['makespan']
        if m < best_makespan:
            best_makespan, best = m, ind
    return best, best_makespan


# ============== SIMULATION CLASS ==============
# handles EA execution, robot movement, collision avoidance, package pickup/delivery
class Simulation:
    def __init__(self):
        self.generations = 200
        self.reset()
    
    def reset(self):
        # initialize robots at starting positions
        self.robots = [{'id': r['id'], 'x': r['x'], 'y': r['y'], 'color': r['color'],
                        'tour': [], 'tour_idx': 0, 'path': [], 'carrying': [],
                        'phase': 'idle', 'waiting': 0} for r in ROBOTS_INIT]
        # initialize packages
        self.packages = [{'id': p['id'], 'x': p['x'], 'y': p['y'], 
                          'visits': p['visits'], 'visits_done': 0} for p in PACKAGES]
        # sim states and values
        self.solved = False
        self.makespan = 0
        self.logs = []
        self.running = False
        self.speed = 5
        self.ea_done = False
        self.generation = 0

    # add messages to log
    def log(self, msg):
        self.logs.append(msg)
        if len(self.logs) > 12:
            self.logs.pop(0)

    # run EA genetic algorithm
    # build distance matrix, evolve pop, assign tours
    def solve_ea(self):
        self.log("Building distance matrix...")
        # construct location list: deposit, packages, robot starts
        locations = [(DEPOSIT['x'], DEPOSIT['y'])]
        locations += [(p['x'], p['y']) for p in PACKAGES]
        locations += [(r['x'], r['y']) for r in ROBOTS_INIT]
        dist_matrix = build_distance_matrix(locations)

        # run EA
        self.log(f"Running EA ({self.generations} gens)...")
        population = init_population(POP_SIZE, PACKAGES, len(ROBOTS_INIT))
        
        for gen in range(self.generations):
            population = evolve(population, dist_matrix, PACKAGES, len(ROBOTS_INIT))
            self.generation = gen + 1

        # get best solution
        best, makespan = get_best(population, dist_matrix, PACKAGES, len(ROBOTS_INIT))
        self.makespan = makespan

        # assign tours
        for r in range(len(self.robots)):
            self.robots[r]['tour'] = [PACKAGES[i]['id'] for i in best[r]]
            self.robots[r]['phase'] = 'toPackage' if self.robots[r]['tour'] else 'idle'
        
        self.solved = True
        self.ea_done = True
        self.log(f"EA done! Makespan: {makespan}")
        for r in self.robots:
            self.log(f"R{r['id']}: {r['tour']}")
    
    def step(self):
        # execute one sim time step
        # path assignment, movement, collision avoidance
        if not self.solved:
            return
        
        for r in self.robots:
            # handle wait for collision avoidance
            if r['waiting'] > 0:
                r['waiting'] -= 1
                continue
            
            # Assign path if needed and robot not idle
            if not r['path'] and r['phase'] != 'idle':
                if r['phase'] == 'toPackage' and r['tour_idx'] < len(r['tour']):
                    # path to next package in tour
                    pkg = next((p for p in self.packages if p['id'] == r['tour'][r['tour_idx']]), None)
                    if pkg:
                        r['path'] = a_star((r['x'], r['y']), (pkg['x'], pkg['y']))[1:]
                elif r['phase'] == 'toDeposit':
                    # path to deposit zone
                    r['path'] = a_star((r['x'], r['y']), (DEPOSIT['x'], DEPOSIT['y']))[1:]
            
            # ===== Move and colission avoidance =====
            if r['path']:
                next_pos = r['path'][0]
                # Collision check (skip idle robots)
                collision = False
                for other in self.robots:
                    if other['id'] == r['id'] or other['phase'] == 'idle':
                        continue
                    if (next_pos[0], next_pos[1]) == (other['x'], other['y']):
                        collision = True
                        break
                        
                # lower priority robot yields
                if collision and r['id'] > other['id']:
                    # attempt sidestep
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        side = (r['x'] + dx, r['y'] + dy)
                        # check if sidestep valid
                        if 0 <= side[0] < GRID_SIZE and 0 <= side[1] < GRID_SIZE:
                            if side not in SHELVES and not any(o['x']==side[0] and o['y']==side[1] for o in self.robots if o['id']!=r['id']):
                                # perform sidestep and recompute path
                                r['x'], r['y'] = side
                                target = (DEPOSIT['x'], DEPOSIT['y']) if r['phase']=='toDeposit' else (next((p for p in self.packages if p['id']==r['tour'][r['tour_idx']]))['x'], next((p for p in self.packages if p['id']==r['tour'][r['tour_idx']]))['y'])
                                r['path'] = a_star(side, target)[1:]
                                self.log(f"R{r['id']} sidestepped")
                                break
                    else:
                        # wait if no sidestep valid
                        r['waiting'] = 2
                else:
                    # no collision or higher priority triggers next move
                    r['x'], r['y'] = next_pos
                    r['path'] = r['path'][1:]
            
            # package pickup
            if r['phase'] == 'toPackage' and not r['path'] and r['tour_idx'] < len(r['tour']):
                pkg = next((p for p in self.packages if p['id'] == r['tour'][r['tour_idx']]), None)
                if pkg and r['x'] == pkg['x'] and r['y'] == pkg['y']:
                    # pickup package
                    pkg['visits_done'] += 1
                    r['carrying'].append(pkg['id'])
                    self.log(f"R{r['id']} picked #{pkg['id']}")
                    r['tour_idx'] += 1

                    # check if need to deposit (end of tour or package capacity)
                    if len(r['carrying']) >= ROBOT_CAPACITY or r['tour_idx'] >= len(r['tour']):
                        r['phase'] = 'toDeposit'
            
            # ===== Package deposit =====
            if r['phase'] == 'toDeposit' and not r['path']:
                if r['x'] == DEPOSIT['x'] and r['y'] == DEPOSIT['y']:
                    # deliver held packages
                    self.log(f"R{r['id']} delivered {r['carrying']}")
                    r['carrying'] = []
                    # continue on tour or go idle
                    r['phase'] = 'toPackage' if r['tour_idx'] < len(r['tour']) else 'idle'
    
    def all_done(self):
        # check all packages deliveded and robots empty
        return all(p['visits_done'] >= p['visits'] for p in self.packages) and all(not r['carrying'] for r in self.robots)


# ============== PYGAME MAIN ==============
# main sim loop
# handles event processing, sim updates, rendering
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MTMV EA Warehouse Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)
    
    sim = Simulation()
    frame_count = 0
    
    while True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # pause/play
                    if sim.solved:
                        sim.running = not sim.running
                elif event.key == pygame.K_r:
                    # reset sim
                    sim.reset()
                elif event.key == pygame.K_e and not sim.ea_done:
                    # run ea on environment
                    sim.solve_ea()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # increase sim speed
                    sim.speed = min(20, sim.speed + 1)
                elif event.key == pygame.K_MINUS:
                    # decrease sim speed
                    sim.speed = max(1, sim.speed - 1)
                elif event.key == pygame.K_s and sim.solved:
                    # move one time step in sim
                    sim.step()
                elif event.key == pygame.K_UP and not sim.ea_done:
                    # increase ea generation count
                    sim.generations = min(2000, sim.generations + 50)
                elif event.key == pygame.K_DOWN and not sim.ea_done:
                    # decrease ea generation count
                    sim.generations = max(50, sim.generations - 50)
        
        # ===== Simulation Update =====
        # update sim at speed controlled rate
        if sim.running and not sim.all_done():
            frame_count += 1
            if frame_count >= (21 - sim.speed):
                sim.step()
                frame_count = 0
        
        # ===== Rendering =====
        screen.fill(BG_COLOR)
        
        # draw grid lines
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pygame.draw.rect(screen, GRID_COLOR, (i*CELL_SIZE, j*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        
        # draw shelves
        for sx, sy in SHELVES:
            pygame.draw.rect(screen, SHELF_COLOR, (sx*CELL_SIZE+2, sy*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4))
        
        # draw deposit zone
        pygame.draw.rect(screen, DEPOSIT_COLOR, (DEPOSIT['x']*CELL_SIZE+2, DEPOSIT['y']*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4))
        txt = small_font.render("DEP", True, (0,0,0))
        screen.blit(txt, (DEPOSIT['x']*CELL_SIZE+8, DEPOSIT['y']*CELL_SIZE+12))
        
        # draw packages with visit progress
        for p in sim.packages:
            # indicate fully satisfied locations
            color = (50,50,50) if p['visits_done'] >= p['visits'] else PACKAGE_COLOR
            pygame.draw.rect(screen, color, (p['x']*CELL_SIZE+6, p['y']*CELL_SIZE+6, CELL_SIZE-12, CELL_SIZE-12))
            
            # show if package needs multiple visits
            if p['visits'] > 1:
                pygame.draw.rect(screen, DEPOSIT_COLOR, (p['x']*CELL_SIZE+6, p['y']*CELL_SIZE+6, CELL_SIZE-12, CELL_SIZE-12), 2)
            
            # display package id and visit count
            txt = small_font.render(f"{p['id']}({p['visits_done']}/{p['visits']})", True, TEXT_COLOR)
            screen.blit(txt, (p['x']*CELL_SIZE+8, p['y']*CELL_SIZE+12))
        
        # robot planned paths
        for r in sim.robots:
            if r['path']:
                # create lines from current position through path
                points = [(r['x']*CELL_SIZE+CELL_SIZE//2, r['y']*CELL_SIZE+CELL_SIZE//2)]
                points += [(p[0]*CELL_SIZE+CELL_SIZE//2, p[1]*CELL_SIZE+CELL_SIZE//2) for p in r['path']]
                if len(points) > 1:
                    pygame.draw.lines(screen, r['color'], False, points, 2)
        
        # draw robots
        for r in sim.robots:
            # robot body
            pygame.draw.circle(screen, r['color'], (r['x']*CELL_SIZE+CELL_SIZE//2, r['y']*CELL_SIZE+CELL_SIZE//2), CELL_SIZE//2-4)
            pygame.draw.circle(screen, TEXT_COLOR, (r['x']*CELL_SIZE+CELL_SIZE//2, r['y']*CELL_SIZE+CELL_SIZE//2), CELL_SIZE//2-4, 2)
            
            # robot id
            txt = font.render(str(r['id']), True, TEXT_COLOR)
            screen.blit(txt, (r['x']*CELL_SIZE+CELL_SIZE//2-5, r['y']*CELL_SIZE+CELL_SIZE//2-8))

            # identify robots holding packages
            if r['carrying']:
                pygame.draw.circle(screen, PACKAGE_COLOR, (r['x']*CELL_SIZE+CELL_SIZE-8, r['y']*CELL_SIZE+8), 8)
                txt = small_font.render(str(len(r['carrying'])), True, TEXT_COLOR)
                screen.blit(txt, (r['x']*CELL_SIZE+CELL_SIZE-12, r['y']*CELL_SIZE+2))
        
        # ===== UI Panel =====
        panel_x = GRID_SIZE * CELL_SIZE + 10
        pygame.draw.rect(screen, (40,40,60), (panel_x-5, 0, MARGIN, HEIGHT))

        # title and controls
        y = 10
        screen.blit(font.render("MTMV EA Simulation", True, DEPOSIT_COLOR), (panel_x, y)); y += 30
        screen.blit(small_font.render("E = Run EA", True, TEXT_COLOR), (panel_x, y)); y += 20
        screen.blit(small_font.render("SPACE = Start/Pause", True, TEXT_COLOR), (panel_x, y)); y += 20
        screen.blit(small_font.render("S = Step", True, TEXT_COLOR), (panel_x, y)); y += 20
        screen.blit(small_font.render("R = Reset", True, TEXT_COLOR), (panel_x, y)); y += 20
        screen.blit(small_font.render("+/- = Speed", True, TEXT_COLOR), (panel_x, y)); y += 20
        screen.blit(small_font.render("UP/DOWN = Generations", True, TEXT_COLOR), (panel_x, y)); y += 30

        # sim params and status
        screen.blit(font.render(f"Generations: {sim.generations}", True, (255,200,100) if not sim.ea_done else (100,100,100)), (panel_x, y)); y += 25
        screen.blit(font.render(f"Speed: {sim.speed}", True, TEXT_COLOR), (panel_x, y)); y += 25
        screen.blit(font.render(f"Makespan: {sim.makespan}", True, DEPOSIT_COLOR), (panel_x, y)); y += 25
        
        # sim status
        status = "DONE!" if sim.all_done() else ("Running" if sim.running else "Paused")
        screen.blit(font.render(f"Status: {status}", True, (46,204,113) if sim.all_done() else TEXT_COLOR), (panel_x, y)); y += 30

        # robot tour assignments
        screen.blit(font.render("Robot Tours:", True, TEXT_COLOR), (panel_x, y)); y += 20
        for r in sim.robots:
            txt = f"R{r['id']}: {r['tour']}"
            screen.blit(small_font.render(txt[:22], True, r['color']), (panel_x, y)); y += 18

        # event log
        y += 10
        screen.blit(font.render("Log:", True, TEXT_COLOR), (panel_x, y)); y += 20
        for log in sim.logs[-8:]:
            screen.blit(small_font.render(log[:24], True, (180,180,180)), (panel_x, y)); y += 16
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
