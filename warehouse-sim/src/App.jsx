import { useState, useEffect, useCallback } from 'react';

const GRID_SIZE = 15;
const CELL_SIZE = 35;

const SHELVES = [
  [3,2],[3,3],[3,4],[3,5],
  [7,2],[7,3],[7,4],[7,5],
  [11,2],[11,3],[11,4],[11,5],
  [3,9],[3,10],[3,11],[3,12],
  [7,9],[7,10],[7,11],[7,12],
  [11,9],[11,10],[11,11],[11,12],
];

// MTMV: packages with required visits (some need multiple visits by different robots)
const INIT_PACKAGES = [
  {id: 1, x: 1, y: 3, visits: 2},   // needs 2 different robots
  {id: 2, x: 5, y: 5, visits: 1},
  {id: 3, x: 9, y: 3, visits: 3},   // needs all 3 robots
  {id: 4, x: 13, y: 5, visits: 1},
  {id: 5, x: 1, y: 11, visits: 2},
  {id: 6, x: 5, y: 10, visits: 1},
  {id: 7, x: 9, y: 11, visits: 2},
  {id: 8, x: 13, y: 10, visits: 1},
];

const DEPOSIT = {x: 7, y: 7};

const INIT_ROBOTS = [
  {id: 1, x: 0, y: 0, color: '#e74c3c'},
  {id: 2, x: 14, y: 0, color: '#3498db'},
  {id: 3, x: 0, y: 14, color: '#2ecc71'},
];

// EA Parameters
const POP_SIZE = 50;
const MUTATION_RATE = 0.15;
const CROSSOVER_RATE = 0.8;
const ROBOT_CAPACITY = 3;

const heuristic = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

const getNeighbors = (node, shelves) => {
  const dirs = [{x:0,y:-1},{x:0,y:1},{x:-1,y:0},{x:1,y:0}];
  return dirs.map(d => ({x: node.x + d.x, y: node.y + d.y}))
    .filter(n => n.x >= 0 && n.x < GRID_SIZE && n.y >= 0 && n.y < GRID_SIZE)
    .filter(n => !shelves.some(s => s[0] === n.x && s[1] === n.y));
};

const aStar = (start, goal, shelves) => {
  if (start.x === goal.x && start.y === goal.y) return [start];
  const openSet = [{...start, g: 0, f: heuristic(start, goal)}];
  const cameFrom = new Map();
  const gScore = new Map();
  gScore.set(`${start.x},${start.y}`, 0);
  
  while (openSet.length > 0) {
    openSet.sort((a, b) => a.f - b.f);
    const current = openSet.shift();
    
    if (current.x === goal.x && current.y === goal.y) {
      const path = [];
      let c = current;
      while (c) { path.unshift({x: c.x, y: c.y}); c = cameFrom.get(`${c.x},${c.y}`); }
      return path;
    }
    
    for (const neighbor of getNeighbors(current, shelves)) {
      const tentG = gScore.get(`${current.x},${current.y}`) + 1;
      const key = `${neighbor.x},${neighbor.y}`;
      if (!gScore.has(key) || tentG < gScore.get(key)) {
        cameFrom.set(key, current);
        gScore.set(key, tentG);
        if (!openSet.some(n => n.x === neighbor.x && n.y === neighbor.y)) {
          openSet.push({...neighbor, g: tentG, f: tentG + heuristic(neighbor, goal)});
        }
      }
    }
  }
  return [];
};

const buildDistanceMatrix = (locations, shelves) => {
  const n = locations.length;
  const matrix = Array(n).fill(null).map(() => Array(n).fill(Infinity));
  for (let i = 0; i < n; i++) {
    matrix[i][i] = 0;
    for (let j = i + 1; j < n; j++) {
      const path = aStar(locations[i], locations[j], shelves);
      const dist = path.length > 0 ? path.length - 1 : 9999;
      matrix[i][j] = dist;
      matrix[j][i] = dist;
    }
  }
  return matrix;
};

// MTMV Fitness: individual[r] = list of package indices for robot r
// Each package can appear in multiple robots' tours (up to its visit requirement)
const calcFitness = (individual, distMatrix, packages, numRobots) => {
  const depositIdx = 0;
  let totalDist = 0;
  let maxDist = 0;
  
  for (let r = 0; r < numRobots; r++) {
    const tour = individual[r];
    if (tour.length === 0) continue;
    
    const robotStartIdx = packages.length + 1 + r;
    let dist = 0;
    let pos = robotStartIdx;
    let carried = 0;
    
    for (const pkgIdx of tour) {
      const pkgLocIdx = pkgIdx + 1;
      dist += distMatrix[pos][pkgLocIdx];
      pos = pkgLocIdx;
      carried++;
      
      if (carried >= ROBOT_CAPACITY || pkgIdx === tour[tour.length - 1]) {
        dist += distMatrix[pos][depositIdx];
        pos = depositIdx;
        carried = 0;
      }
    }
    
    totalDist += dist;
    maxDist = Math.max(maxDist, dist);
  }
  
  return { makespan: maxDist, totalDist, fitness: 1 / (maxDist + 1) };
};

// MTMV Random Individual: distribute visits across robots
const randomIndividual = (packages, numRobots) => {
  const individual = Array.from({length: numRobots}, () => []);
  
  // For each package, assign required number of visits to different robots
  for (let pkgIdx = 0; pkgIdx < packages.length; pkgIdx++) {
    const pkg = packages[pkgIdx];
    const robotsNeeded = Math.min(pkg.visits, numRobots);
    
    // Shuffle robot indices and pick required number
    const robotOrder = Array.from({length: numRobots}, (_, i) => i);
    for (let i = robotOrder.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [robotOrder[i], robotOrder[j]] = [robotOrder[j], robotOrder[i]];
    }
    
    for (let v = 0; v < robotsNeeded; v++) {
      individual[robotOrder[v]].push(pkgIdx);
    }
  }
  
  // Shuffle each robot's tour
  for (let r = 0; r < numRobots; r++) {
    for (let i = individual[r].length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [individual[r][i], individual[r][j]] = [individual[r][j], individual[r][i]];
    }
  }
  
  return individual;
};

const initPopulation = (popSize, packages, numRobots) => {
  return Array.from({length: popSize}, () => randomIndividual(packages, numRobots));
};

const selectParent = (population, fitnesses) => {
  const totalFit = fitnesses.reduce((a, b) => a + b.fitness, 0);
  let r = Math.random() * totalFit;
  for (let i = 0; i < population.length; i++) {
    r -= fitnesses[i].fitness;
    if (r <= 0) return population[i];
  }
  return population[population.length - 1];
};

// MTMV Crossover
const crossover = (p1, p2, numRobots) => {
  if (Math.random() > CROSSOVER_RATE) return [p1.map(t => [...t]), p2.map(t => [...t])];
  
  const c1 = p1.map(t => [...t]);
  const c2 = p2.map(t => [...t]);
  
  // Exchange portion of one robot's tour
  const r = Math.floor(Math.random() * numRobots);
  const len1 = c1[r].length, len2 = c2[r].length;
  if (len1 === 0 || len2 === 0) return [c1, c2];
  
  const point = Math.floor(Math.random() * Math.min(len1, len2));
  const seg1 = c1[r].splice(point);
  const seg2 = c2[r].splice(point);
  c1[r] = c1[r].concat(seg2);
  c2[r] = c2[r].concat(seg1);
  
  return [c1, c2];
};

// MTMV Path Planning Strategy (Algorithm 1 from paper)
// Ensures: each robot visits each package at most once, total visits = required
const repairMTMV = (individual, packages, numRobots) => {
  const result = Array.from({length: numRobots}, () => []);
  const visitCount = Array(packages.length).fill(0);
  const robotVisited = Array.from({length: numRobots}, () => new Set());
  
  // Forward pass: keep valid, move duplicates to next robot
  for (let r = 0; r < numRobots - 1; r++) {
    for (const pkgIdx of individual[r]) {
      if (pkgIdx < 0 || pkgIdx >= packages.length) continue;
      
      if (!robotVisited[r].has(pkgIdx) && visitCount[pkgIdx] < packages[pkgIdx].visits) {
        result[r].push(pkgIdx);
        robotVisited[r].add(pkgIdx);
        visitCount[pkgIdx]++;
      } else if (!robotVisited[r].has(pkgIdx)) {
        // Move to next robot
        individual[r + 1].splice(Math.floor(Math.random() * (individual[r + 1].length + 1)), 0, pkgIdx);
      }
    }
  }
  
  // Last robot
  for (const pkgIdx of individual[numRobots - 1]) {
    if (pkgIdx < 0 || pkgIdx >= packages.length) continue;
    if (!robotVisited[numRobots - 1].has(pkgIdx) && visitCount[pkgIdx] < packages[pkgIdx].visits) {
      result[numRobots - 1].push(pkgIdx);
      robotVisited[numRobots - 1].add(pkgIdx);
      visitCount[pkgIdx]++;
    }
  }
  
  // Backward pass: handle remaining from last robot
  for (let r = numRobots - 1; r > 0; r--) {
    const overflow = [];
    for (const pkgIdx of result[r]) {
      if (robotVisited[r].has(pkgIdx) && visitCount[pkgIdx] > packages[pkgIdx].visits) {
        overflow.push(pkgIdx);
      }
    }
    for (const pkgIdx of overflow) {
      result[r] = result[r].filter(p => p !== pkgIdx || !robotVisited[r].has(pkgIdx));
      if (!robotVisited[r - 1].has(pkgIdx) && visitCount[pkgIdx] < packages[pkgIdx].visits) {
        result[r - 1].push(pkgIdx);
        robotVisited[r - 1].add(pkgIdx);
      }
    }
  }
  
  // Final pass: ensure all packages have required visits
  for (let pkgIdx = 0; pkgIdx < packages.length; pkgIdx++) {
    while (visitCount[pkgIdx] < packages[pkgIdx].visits) {
      // Find robot that hasn't visited this package
      for (let r = 0; r < numRobots; r++) {
        if (!robotVisited[r].has(pkgIdx)) {
          result[r].push(pkgIdx);
          robotVisited[r].add(pkgIdx);
          visitCount[pkgIdx]++;
          break;
        }
      }
      // Safety: if all robots already visited (shouldn't happen if visits <= numRobots)
      if (visitCount[pkgIdx] < packages[pkgIdx].visits) break;
    }
  }
  
  return result;
};

// MTMV Mutation
const mutate = (individual, numRobots, packages) => {
  if (Math.random() > MUTATION_RATE) return individual;
  
  const result = individual.map(t => [...t]);
  const times = Math.floor(Math.random() * 3) + 1;
  
  for (let t = 0; t < times; t++) {
    const r1 = Math.floor(Math.random() * numRobots);
    const r2 = Math.floor(Math.random() * numRobots);
    
    if (result[r1].length === 0) continue;
    const i1 = Math.floor(Math.random() * result[r1].length);
    
    if (r1 === r2 && result[r1].length > 1) {
      const i2 = Math.floor(Math.random() * result[r1].length);
      [result[r1][i1], result[r1][i2]] = [result[r1][i2], result[r1][i1]];
    } else if (r1 !== r2) {
      const pkg = result[r1].splice(i1, 1)[0];
      result[r2].splice(Math.floor(Math.random() * (result[r2].length + 1)), 0, pkg);
    }
  }
  
  return result;
};

const evolve = (population, distMatrix, packages, numRobots) => {
  const fitnesses = population.map(ind => calcFitness(ind, distMatrix, packages, numRobots));
  const newPop = [];
  
  let bestIdx = 0;
  for (let i = 1; i < population.length; i++) {
    if (fitnesses[i].makespan < fitnesses[bestIdx].makespan) bestIdx = i;
  }
  newPop.push(population[bestIdx].map(t => [...t]));
  
  while (newPop.length < population.length) {
    const p1 = selectParent(population, fitnesses);
    const p2 = selectParent(population, fitnesses);
    
    let [c1, c2] = crossover(p1, p2, numRobots);
    c1 = mutate(c1, numRobots, packages);
    c2 = mutate(c2, numRobots, packages);
    c1 = repairMTMV(c1, packages, numRobots);
    c2 = repairMTMV(c2, packages, numRobots);
    
    newPop.push(c1);
    if (newPop.length < population.length) newPop.push(c2);
  }
  
  return newPop;
};

const getBest = (population, distMatrix, packages, numRobots) => {
  let best = null, bestMakespan = Infinity;
  for (const ind of population) {
    const { makespan } = calcFitness(ind, distMatrix, packages, numRobots);
    if (makespan < bestMakespan) {
      bestMakespan = makespan;
      best = ind;
    }
  }
  return { best, makespan: bestMakespan };
};

const createInitialState = () => ({
  shelves: SHELVES,
  packages: INIT_PACKAGES.map(p => ({...p, visitsDone: 0})),
  deposit: DEPOSIT,
  robots: INIT_ROBOTS.map(r => ({...r, tour: [], tourIndex: 0, path: [], carrying: [], waiting: 0, phase: 'idle'})),
  solved: false,
  makespan: null,
  generation: 0,
  eaHistory: []
});

export default function WarehouseEA() {
  const [state, setState] = useState(createInitialState);
  const [running, setRunning] = useState(false);
  const [evolving, setEvolving] = useState(false);
  const [speed, setSpeed] = useState(300);
  const [log, setLog] = useState([]);
  const [population, setPopulation] = useState(null);
  const [distMatrix, setDistMatrix] = useState(null);
  const [targetGens, setTargetGens] = useState(100);
  
  const numRobots = INIT_ROBOTS.length;
  const packages = INIT_PACKAGES;
  
  const addLog = (msg) => setLog(prev => [...prev.slice(-20), msg]);
  
  const reset = () => {
    setState(createInitialState());
    setLog([]);
    setRunning(false);
    setEvolving(false);
    setPopulation(null);
  };
  
  const initEA = () => {
    const locations = [
      DEPOSIT,
      ...packages.map(p => ({x: p.x, y: p.y})),
      ...INIT_ROBOTS.map(r => ({x: r.x, y: r.y}))
    ];
    const dm = buildDistanceMatrix(locations, SHELVES);
    setDistMatrix(dm);
    
    const pop = initPopulation(POP_SIZE, packages, numRobots);
    setPopulation(pop);
    
    const { makespan } = getBest(pop, dm, packages, numRobots);
    setState(prev => ({...prev, generation: 0, eaHistory: [makespan]}));
    
    const totalVisits = packages.reduce((s, p) => s + p.visits, 0);
    addLog(`MTMV EA init. Total visits needed: ${totalVisits}`);
    addLog(`Initial makespan: ${makespan}`);
  };
  
  useEffect(() => {
    if (!evolving || !population || !distMatrix) return;
    
    const interval = setInterval(() => {
      setPopulation(prev => {
        const newPop = evolve(prev, distMatrix, packages, numRobots);
        const { makespan } = getBest(newPop, distMatrix, packages, numRobots);
        
        setState(s => {
          const newGen = s.generation + 1;
          if (newGen >= targetGens) {
            setEvolving(false);
            addLog(`EA complete! Final makespan: ${makespan}`);
          } else if (newGen % 10 === 0) {
            addLog(`Gen ${newGen}: makespan = ${makespan}`);
          }
          return {...s, generation: newGen, eaHistory: [...s.eaHistory, makespan]};
        });
        
        return newPop;
      });
    }, 20);
    
    return () => clearInterval(interval);
  }, [evolving, population, distMatrix, numRobots, targetGens]);
  
  const applySolution = () => {
    if (!population || !distMatrix) return;
    
    const { best, makespan } = getBest(population, distMatrix, packages, numRobots);
    const tours = best.map(pkgIndices => pkgIndices.map(i => packages[i].id));
    
    setState(prev => ({
      ...prev,
      robots: prev.robots.map((r, i) => ({
        ...r,
        tour: tours[i],
        tourIndex: 0,
        phase: tours[i].length > 0 ? 'toPackage' : 'idle'
      })),
      solved: true,
      makespan
    }));
    
    addLog(`Solution applied:`);
    tours.forEach((t, i) => addLog(`  R${i+1}: [${t.join('→')}]`));
  };
  
  const step = useCallback(() => {
    setState(prev => {
      if (!prev.solved) return prev;
      
      let { robots, packages: pkgs, deposit, shelves } = prev;
      robots = robots.map(r => ({...r, path: [...r.path], carrying: [...r.carrying]}));
      
      robots = robots.map(r => {
        if (r.path.length > 0 || r.phase === 'idle') return r;
        if (r.phase === 'toPackage' && r.tourIndex < r.tour.length) {
          const pkg = pkgs.find(p => p.id === r.tour[r.tourIndex]);
          if (pkg) {
            const path = aStar(r, pkg, shelves);
            return {...r, path: path.slice(1)};
          }
        }
        if (r.phase === 'toDeposit') {
          const path = aStar(r, deposit, shelves);
          return {...r, path: path.slice(1)};
        }
        return r;
      });
      
      const nextPos = robots.map(r => (r.waiting > 0 || !r.path.length) ? {x: r.x, y: r.y} : r.path[0]);
      
      const newRobots = robots.map((robot, i) => {
        let r = {...robot, carrying: [...robot.carrying]};
        if (r.waiting > 0) { r.waiting--; return r; }
        if (!r.path.length) return r;
        
        const next = r.path[0];
        let collision = false;
        
        for (let j = 0; j < robots.length; j++) {
          if (i === j) continue;
          const other = robots[j], otherNext = nextPos[j];
          if ((next.x === otherNext.x && next.y === otherNext.y) ||
              (next.x === other.x && next.y === other.y)) {
            collision = true; break;
          }
        }
        
        if (collision && robots.some((o, j) => i !== j && o.id < r.id && 
            (next.x === nextPos[j].x && next.y === nextPos[j].y || next.x === o.x && next.y === o.y))) {
          const sides = [{x:1,y:0},{x:-1,y:0},{x:0,y:1},{x:0,y:-1}];
          for (const s of sides) {
            const side = {x: r.x + s.x, y: r.y + s.y};
            if (side.x < 0 || side.x >= GRID_SIZE || side.y < 0 || side.y >= GRID_SIZE) continue;
            if (shelves.some(sh => sh[0] === side.x && sh[1] === side.y)) continue;
            if (robots.some((o, j) => i !== j && (o.x === side.x && o.y === side.y || nextPos[j].x === side.x && nextPos[j].y === side.y))) continue;
            
            const target = r.phase === 'toDeposit' ? deposit : pkgs.find(p => p.id === r.tour[r.tourIndex]);
            if (target) {
              const newPath = aStar(side, target, shelves);
              if (newPath.length > 0) {
                r.x = side.x; r.y = side.y; r.path = newPath.slice(1);
                addLog(`R${r.id} sidestepped`);
                return r;
              }
            }
          }
          r.waiting = 2;
          return r;
        }
        
        r.x = next.x; r.y = next.y; r.path = r.path.slice(1);
        return r;
      });
      
      let newPkgs = [...pkgs];
      const finalRobots = newRobots.map(r => {
        if (r.phase === 'toPackage' && r.path.length === 0) {
          const pkg = newPkgs.find(p => p.id === r.tour[r.tourIndex]);
          if (pkg && r.x === pkg.x && r.y === pkg.y) {
            newPkgs = newPkgs.map(p => p.id === pkg.id ? {...p, visitsDone: p.visitsDone + 1} : p);
            const newCarrying = [...r.carrying, pkg.id];
            addLog(`R${r.id} visited #${pkg.id} (${newCarrying.length}/${ROBOT_CAPACITY})`);
            
            const nextIdx = r.tourIndex + 1;
            const atCapacity = newCarrying.length >= ROBOT_CAPACITY;
            const tourDone = nextIdx >= r.tour.length;
            
            if (atCapacity || tourDone) {
              return {...r, carrying: newCarrying, phase: 'toDeposit', path: [], tourIndex: nextIdx};
            } else {
              return {...r, carrying: newCarrying, tourIndex: nextIdx, path: []};
            }
          }
        }
        
        if (r.phase === 'toDeposit' && r.path.length === 0 && r.x === deposit.x && r.y === deposit.y) {
          addLog(`R${r.id} delivered [${r.carrying.join(',')}]`);
          const tourDone = r.tourIndex >= r.tour.length;
          return {...r, carrying: [], phase: tourDone ? 'idle' : 'toPackage', path: []};
        }
        
        return r;
      });
      
      return {...prev, robots: finalRobots, packages: newPkgs};
    });
  }, []);
  
  useEffect(() => {
    if (!running) return;
    const interval = setInterval(step, speed);
    return () => clearInterval(interval);
  }, [running, speed, step]);
  
  const totalVisits = state.packages.reduce((s, p) => s + p.visits, 0);
  const visitsDone = state.packages.reduce((s, p) => s + p.visitsDone, 0);
  const allDone = visitsDone >= totalVisits && state.robots.every(r => r.carrying.length === 0);
  
  return (
    <div className="p-3 bg-gray-900 min-h-screen text-white">
      <h1 className="text-xl font-bold mb-2">Multi-Robot Warehouse (MTMV EA)</h1>
      
      <div className="flex gap-2 mb-2 flex-wrap items-center">
        {!population && <button onClick={initEA} className="px-3 py-1 bg-purple-600 rounded hover:bg-purple-700">Init EA</button>}
        {population && !state.solved && (
          <>
            <button onClick={() => setEvolving(!evolving)} className="px-3 py-1 bg-yellow-600 rounded hover:bg-yellow-700">
              {evolving ? 'Pause EA' : 'Run EA'}
            </button>
            <span className="text-sm">Gen: {state.generation}/{targetGens}</span>
            <input type="range" min="50" max="500" value={targetGens} onChange={e => setTargetGens(Number(e.target.value))} className="w-24"/>
            <button onClick={applySolution} className="px-3 py-1 bg-green-600 rounded hover:bg-green-700">Apply Solution</button>
          </>
        )}
        {state.solved && (
          <>
            <button onClick={() => setRunning(!running)} className="px-3 py-1 bg-blue-600 rounded">{running ? 'Pause' : 'Start'}</button>
            <button onClick={step} className="px-3 py-1 bg-green-600 rounded">Step</button>
          </>
        )}
        <button onClick={reset} className="px-3 py-1 bg-red-600 rounded">Reset</button>
        <select value={speed} onChange={e => setSpeed(Number(e.target.value))} className="px-2 py-1 bg-gray-700 rounded text-sm">
          <option value={500}>Slow</option>
          <option value={300}>Medium</option>
          <option value={100}>Fast</option>
        </select>
      </div>
      
      {allDone && <div className="mb-2 p-2 bg-green-700 rounded text-center font-bold">All visits complete!</div>}
      
      <div className="flex gap-4 flex-wrap">
        <svg width={GRID_SIZE * CELL_SIZE} height={GRID_SIZE * CELL_SIZE} className="border border-gray-600">
          {Array.from({length: GRID_SIZE}, (_, i) => Array.from({length: GRID_SIZE}, (_, j) => (
            <rect key={`${i}-${j}`} x={i*CELL_SIZE} y={j*CELL_SIZE} width={CELL_SIZE} height={CELL_SIZE} fill="#1a1a2e" stroke="#333"/>
          )))}
          {state.shelves.map(([x, y], i) => <rect key={`s${i}`} x={x*CELL_SIZE+2} y={y*CELL_SIZE+2} width={CELL_SIZE-4} height={CELL_SIZE-4} fill="#8b4513" rx="2"/>)}
          <rect x={state.deposit.x*CELL_SIZE+2} y={state.deposit.y*CELL_SIZE+2} width={CELL_SIZE-4} height={CELL_SIZE-4} fill="#f39c12" rx="4"/>
          <text x={state.deposit.x*CELL_SIZE+CELL_SIZE/2} y={state.deposit.y*CELL_SIZE+CELL_SIZE/2+4} textAnchor="middle" fontSize="10" fill="#000">DEP</text>
          
          {/* Packages with visit counts */}
          {state.packages.map(p => (
            <g key={`p${p.id}`}>
              <rect x={p.x*CELL_SIZE+6} y={p.y*CELL_SIZE+6} width={CELL_SIZE-12} height={CELL_SIZE-12} 
                fill={p.visitsDone >= p.visits ? '#2d3748' : '#9b59b6'} rx="2" stroke={p.visits > 1 ? '#f39c12' : 'none'} strokeWidth="2"/>
              <text x={p.x*CELL_SIZE+CELL_SIZE/2} y={p.y*CELL_SIZE+CELL_SIZE/2+4} textAnchor="middle" fontSize="10" fill="#fff">
                {p.id}({p.visitsDone}/{p.visits})
              </text>
            </g>
          ))}
          
          {state.robots.map(r => r.path.length > 0 && (
            <polyline key={`path${r.id}`} points={[{x:r.x,y:r.y},...r.path].map(p=>`${p.x*CELL_SIZE+CELL_SIZE/2},${p.y*CELL_SIZE+CELL_SIZE/2}`).join(' ')} fill="none" stroke={r.color} strokeWidth="2" strokeDasharray="4" opacity="0.5"/>
          ))}
          {state.robots.map(r => (
            <g key={`r${r.id}`}>
              <circle cx={r.x*CELL_SIZE+CELL_SIZE/2} cy={r.y*CELL_SIZE+CELL_SIZE/2} r={CELL_SIZE/2-4} fill={r.color} stroke="#fff" strokeWidth="2"/>
              <text x={r.x*CELL_SIZE+CELL_SIZE/2} y={r.y*CELL_SIZE+CELL_SIZE/2+4} textAnchor="middle" fontSize="12" fill="#fff" fontWeight="bold">{r.id}</text>
              {r.carrying.length > 0 && (
                <g>
                  <circle cx={r.x*CELL_SIZE+CELL_SIZE-6} cy={r.y*CELL_SIZE+6} r="8" fill="#9b59b6" stroke="#fff"/>
                  <text x={r.x*CELL_SIZE+CELL_SIZE-6} y={r.y*CELL_SIZE+10} textAnchor="middle" fontSize="10" fill="#fff">{r.carrying.length}</text>
                </g>
              )}
            </g>
          ))}
        </svg>
        
        <div className="flex-1 min-w-60">
          {state.eaHistory.length > 0 && (
            <div className="mb-3">
              <h3 className="font-bold text-sm mb-1">EA Progress</h3>
              <div className="bg-gray-800 p-2 rounded h-20 flex items-end gap-px">
                {state.eaHistory.slice(-100).map((m, i) => (
                  <div key={i} className="bg-green-500 flex-1" style={{height: `${Math.max(5, 100 - (m - Math.min(...state.eaHistory)) * 2)}%`}}/>
                ))}
              </div>
              <div className="text-xs text-gray-400">Best: {Math.min(...state.eaHistory)}</div>
            </div>
          )}
          
          <h3 className="font-bold text-sm mb-1">MTMV Package Visits</h3>
          <div className="text-xs bg-gray-800 p-2 rounded mb-2 max-h-20 overflow-y-auto">
            {packages.map(p => (
              <span key={p.id} className={`mr-2 ${state.packages.find(x=>x.id===p.id)?.visitsDone >= p.visits ? 'text-green-400' : ''}`}>
                #{p.id}:{p.visits}visits
              </span>
            ))}
          </div>
          
          <h3 className="font-bold text-sm mb-1">Robot Tours</h3>
          <div className="text-xs bg-gray-800 p-2 rounded mb-2">
            {state.robots.map(r => (
              <div key={r.id} className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full" style={{backgroundColor: r.color}}/>
                <span>R{r.id}: [{r.tour.join('→')}]</span>
                {r.carrying.length > 0 && <span className="text-purple-400">📦{r.carrying.length}</span>}
              </div>
            ))}
          </div>
          
          <h3 className="font-bold text-sm mb-1">Log</h3>
          <div className="bg-gray-800 p-2 rounded text-xs h-28 overflow-y-auto">
            {log.map((m, i) => <div key={i}>{m}</div>)}
            {!log.length && <span className="text-gray-500">Click "Init EA"</span>}
          </div>
        </div>
      </div>
    </div>
  );
}