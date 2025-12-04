import React, { useState, useEffect, useCallback, useRef } from 'react';

// ============================================================================
// ALGORITHMS
// ============================================================================

class PriorityQueue {
  constructor() { this.items = []; }
  enqueue(item, priority) {
    this.items.push({ item, priority });
    this.items.sort((a, b) => a.priority - b.priority);
  }
  dequeue() { return this.items.shift()?.item; }
  isEmpty() { return this.items.length === 0; }
}

function astar(grid, start, goal, rows, cols, blockedCells = new Set()) {
  const key = (r, c) => `${r},${c}`;
  const heuristic = (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
  if (start[0] === goal[0] && start[1] === goal[1]) return [start];
  const openSet = new PriorityQueue();
  openSet.enqueue(start, 0);
  const cameFrom = new Map(), gScore = new Map(), visited = new Set();
  gScore.set(key(...start), 0);
  const dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];
  while (!openSet.isEmpty()) {
    const current = openSet.dequeue();
    const currKey = key(...current);
    if (visited.has(currKey)) continue;
    visited.add(currKey);
    if (current[0] === goal[0] && current[1] === goal[1]) {
      const path = [current];
      let curr = current;
      while (cameFrom.has(key(...curr))) { curr = cameFrom.get(key(...curr)); path.unshift(curr); }
      return path;
    }
    for (const [dr, dc] of dirs) {
      const nr = current[0] + dr, nc = current[1] + dc, nKey = key(nr, nc);
      if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
      if (grid[nr][nc] === 1 || blockedCells.has(nKey)) continue;
      const tentG = gScore.get(currKey) + 1;
      if (tentG < (gScore.get(nKey) ?? Infinity)) {
        cameFrom.set(nKey, current);
        gScore.set(nKey, tentG);
        openSet.enqueue([nr, nc], tentG + heuristic([nr, nc], goal));
      }
    }
  }
  return null;
}

function computeDistanceMatrix(grid, locations, rows, cols) {
  const n = locations.length, matrix = Array(n).fill(null).map(() => Array(n).fill(Infinity)), paths = {};
  for (let i = 0; i < n; i++) {
    matrix[i][i] = 0;
    for (let j = i + 1; j < n; j++) {
      const path = astar(grid, locations[i], locations[j], rows, cols);
      if (path) { matrix[i][j] = matrix[j][i] = path.length - 1; paths[`${i}-${j}`] = path; paths[`${j}-${i}`] = [...path].reverse(); }
    }
  }
  return { matrix, paths };
}

function solveTSP(distMatrix, packageIndices, depotIdx, depositIdx) {
  if (packageIndices.length === 0) return { tour: [depotIdx, depositIdx, depotIdx], dist: 0 };
  const tour = [depotIdx];
  const unvisited = new Set(packageIndices);
  let current = depotIdx, totalDist = 0;
  while (unvisited.size > 0) {
    let nearest = null, nearestDist = Infinity;
    for (const pkg of unvisited) { if (distMatrix[current][pkg] < nearestDist) { nearestDist = distMatrix[current][pkg]; nearest = pkg; } }
    if (nearest === null) break;
    tour.push(nearest); totalDist += nearestDist; unvisited.delete(nearest); current = nearest;
  }
  tour.push(depositIdx); totalDist += distMatrix[current][depositIdx];
  tour.push(depotIdx); totalDist += distMatrix[depositIdx][depotIdx];
  return { tour, dist: totalDist };
}

function assignTasksToRobots(distMatrix, packageIndices, robotDepots, depositIdx, numRobots) {
  const assignments = Array(numRobots).fill(null).map(() => []);
  for (const pkg of packageIndices) {
    let bestRobot = 0, bestDist = Infinity;
    for (let r = 0; r < numRobots; r++) { const d = distMatrix[robotDepots[r]][pkg]; if (d < bestDist) { bestDist = d; bestRobot = r; } }
    assignments[bestRobot].push(pkg);
  }
  return assignments.map((pkgs, r) => ({ ...solveTSP(distMatrix, pkgs, robotDepots[r], depositIdx), packages: pkgs }));
}

function buildFullPath(tour, paths) {
  const fullPath = [];
  for (let i = 0; i < tour.length - 1; i++) {
    const segment = paths[`${tour[i]}-${tour[i + 1]}`];
    if (segment) { if (fullPath.length > 0) fullPath.pop(); fullPath.push(...segment); }
  }
  return fullPath;
}

function replanRoute(robot, allRobots, grid, rows, cols) {
  const blocked = new Set();
  allRobots.forEach(o => { if (o.id !== robot.id) blocked.add(`${o.pos[0]},${o.pos[1]}`); });
  return astar(grid, robot.pos, robot.path[robot.path.length - 1], rows, cols, blocked);
}

// ============================================================================
// RANDOM ENVIRONMENT GENERATOR
// ============================================================================

function generateRandomEnvironment(rows, cols, numShelves, numPackages, numRobots, seed = null) {
  let rng = seed !== null ? mulberry32(seed) : () => Math.random();
  
  function mulberry32(a) {
    return function() {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
  }
  
  const randInt = (min, max) => Math.floor(rng() * (max - min + 1)) + min;
  const randChoice = (arr) => arr[Math.floor(rng() * arr.length)];
  
  const grid = Array(rows).fill(null).map(() => Array(cols).fill(0));
  const occupied = new Set();
  const key = (r, c) => `${r},${c}`;
  
  const buffer = 3;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (r < buffer || r >= rows - buffer || c < buffer || c >= cols - buffer) {
        occupied.add(key(r, c));
      }
    }
  }
  
  const shelves = [];
  const shelfWidth = randInt(4, 7);
  const shelfHeight = 2;
  const corridorWidth = 3;
  
  let attempts = 0;
  while (shelves.length < numShelves && attempts < 500) {
    attempts++;
    const sr = randInt(buffer + 1, rows - buffer - shelfHeight - 1);
    const sc = randInt(buffer + 1, cols - buffer - shelfWidth - 1);
    
    let canPlace = true;
    for (let r = sr - corridorWidth; r < sr + shelfHeight + corridorWidth && canPlace; r++) {
      for (let c = sc - corridorWidth; c < sc + shelfWidth + corridorWidth && canPlace; c++) {
        if (r >= 0 && r < rows && c >= 0 && c < cols) {
          if (grid[r]?.[c] === 1) canPlace = false;
        }
      }
    }
    
    if (canPlace) {
      for (let r = sr; r < sr + shelfHeight; r++) {
        for (let c = sc; c < sc + shelfWidth; c++) {
          grid[r][c] = 1;
          occupied.add(key(r, c));
        }
      }
      shelves.push({ r: sr, c: sc, h: shelfHeight, w: shelfWidth });
    }
  }
  
  const walkable = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c] === 0) walkable.push([r, c]);
    }
  }
  
  const depositCandidates = walkable.filter(([r, c]) => r < rows / 3 && c > cols * 2 / 3);
  const deposit = depositCandidates.length > 0 ? randChoice(depositCandidates) : [buffer, cols - buffer - 1];
  occupied.add(key(...deposit));
  
  const depots = [];
  const depotRegions = [];
  for (let i = 0; i < numRobots; i++) {
    const regionStart = Math.floor((rows - 2 * buffer) / numRobots) * i + buffer;
    const regionEnd = Math.floor((rows - 2 * buffer) / numRobots) * (i + 1) + buffer;
    depotRegions.push([regionStart, regionEnd]);
  }
  
  for (let i = 0; i < numRobots; i++) {
    const [rStart, rEnd] = depotRegions[i];
    const candidates = walkable.filter(([r, c]) => 
      r >= rStart && r < rEnd && c < cols / 4 && !occupied.has(key(r, c))
    );
    if (candidates.length > 0) {
      const depot = randChoice(candidates);
      depots.push(depot);
      occupied.add(key(...depot));
    } else {
      depots.push([rStart + 1, buffer]);
    }
  }
  
  const packages = [];
  const packageCandidates = [];
  
  for (const shelf of shelves) {
    for (let r = shelf.r - 1; r <= shelf.r + shelf.h; r++) {
      for (let c = shelf.c - 1; c <= shelf.c + shelf.w; c++) {
        if (r >= 0 && r < rows && c >= 0 && c < cols && 
            grid[r][c] === 0 && !occupied.has(key(r, c))) {
          packageCandidates.push([r, c]);
        }
      }
    }
  }
  
  walkable.forEach(pos => {
    if (!occupied.has(key(...pos)) && rng() < 0.1) {
      packageCandidates.push(pos);
    }
  });
  
  const shuffled = packageCandidates.sort(() => rng() - 0.5);
  for (let i = 0; i < Math.min(numPackages, shuffled.length); i++) {
    if (!occupied.has(key(...shuffled[i]))) {
      packages.push({ pos: shuffled[i], collected: false });
      occupied.add(key(...shuffled[i]));
    }
  }
  
  const testStart = depots[0];
  const allLocations = [deposit, ...depots, ...packages.map(p => p.pos)];
  let valid = true;
  
  for (const loc of allLocations) {
    const path = astar(grid, testStart, loc, rows, cols);
    if (!path) {
      valid = false;
      break;
    }
  }
  
  return { grid, deposit, depots, packages, valid, seed };
}

// ============================================================================
// UI COMPONENT
// ============================================================================

const CELL = 18;
const COLORS = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6'];

export default function WarehouseSimulation() {
  const [rows, setRows] = useState(30);
  const [cols, setCols] = useState(40);
  const [numShelves, setNumShelves] = useState(8);
  const [numPackages, setNumPackages] = useState(25);
  const [seed, setSeed] = useState('');
  
  const [grid, setGrid] = useState([]);
  const [packages, setPackages] = useState([]);
  const [deposit, setDeposit] = useState([4, 36]);
  const [robots, setRobots] = useState([]);
  const [mode, setMode] = useState('multi');
  const [numRobots, setNumRobots] = useState(3);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState({ totalDist: 0, steps: 0, collected: 0, waits: 0, replans: 0 });
  const [speed, setSpeed] = useState(60);
  const [collisionLog, setCollisionLog] = useState([]);
  const [envValid, setEnvValid] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const animRef = useRef(null);
  const gridRef = useRef([]);

  const generateEnvironment = useCallback((useSeed = false) => {
    const seedVal = useSeed && seed ? parseInt(seed) : Math.floor(Math.random() * 1000000);
    if (!useSeed) setSeed(seedVal.toString());
    
    let env;
    let attempts = 0;
    
    do {
      env = generateRandomEnvironment(rows, cols, numShelves, numPackages, numRobots, seedVal + attempts);
      attempts++;
    } while (!env.valid && attempts < 10);
    
    setGrid(env.grid);
    gridRef.current = env.grid;
    setDeposit(env.deposit);
    setPackages(env.packages);
    setEnvValid(env.valid);
    
    setRobots(env.depots.map((pos, id) => ({
      id, pos: [...pos], depot: [...pos], path: [], pathIdx: 0, totalDist: 0,
      waiting: false, waitCount: 0, status: 'idle', finished: false
    })));
    
    setStats({ totalDist: 0, steps: 0, collected: 0, waits: 0, replans: 0 });
    setCollisionLog([]);
    setRunning(false);
  }, [rows, cols, numShelves, numPackages, numRobots, seed]);

  const initDefaultWarehouse = useCallback(() => {
    const g = Array(rows).fill(null).map(() => Array(cols).fill(0));
    const shelfRows = [6, 7, 11, 12, 16, 17, 21, 22, 26, 27];
    const shelfCols = [[6, 12], [16, 22], [26, 32]];
    for (const r of shelfRows) for (const [c1, c2] of shelfCols) for (let c = c1; c <= c2; c++) if (r < rows && c < cols) g[r][c] = 1;
    setGrid(g);
    gridRef.current = g;

    const pkgPositions = [
      [5, 7], [5, 11], [5, 17], [5, 21], [5, 27], [5, 31], [8, 6], [8, 12], [8, 16], [8, 22], [8, 26], [8, 32],
      [13, 7], [13, 11], [13, 17], [13, 21], [13, 27], [13, 31], [18, 6], [18, 12], [18, 16], [18, 22], [18, 26], [18, 32],
      [23, 7], [23, 11], [23, 17], [23, 21], [23, 27], [23, 31]
    ].filter(([r, c]) => r < rows && c < cols);
    setPackages(pkgPositions.filter(([r, c]) => g[r][c] === 0).map(pos => ({ pos, collected: false })));

    const dep = [4, Math.min(36, cols - 4)];
    setDeposit(dep);
    
    const depots = mode === 'single' ? [[25, 3]] : [[25, 3], [4, 3], [24, 4]].slice(0, numRobots);
    setRobots(depots.filter(([r, c]) => r < rows && c < cols).map((pos, id) => ({
      id, pos: [...pos], depot: [...pos], path: [], pathIdx: 0, totalDist: 0,
      waiting: false, waitCount: 0, status: 'idle', finished: false
    })));

    setStats({ totalDist: 0, steps: 0, collected: 0, waits: 0, replans: 0 });
    setCollisionLog([]);
    setRunning(false);
    setEnvValid(true);
  }, [rows, cols, mode, numRobots]);

  useEffect(() => { initDefaultWarehouse(); }, []);

  const startSimulation = () => {
    if (running || !envValid) return;
    const pkgIndices = packages.map((_, i) => i + robots.length + 1);
    const depositIdx = robots.length;
    const locations = [...robots.map(r => r.depot), deposit, ...packages.map(p => p.pos)];
    const { matrix, paths } = computeDistanceMatrix(grid, locations, rows, cols);

    const robotTours = mode === 'single'
      ? [{ ...solveTSP(matrix, pkgIndices, 0, depositIdx), packages: pkgIndices }]
      : assignTasksToRobots(matrix, pkgIndices, robots.map((_, i) => i), depositIdx, robots.length);

    setRobots(robots.map((r, idx) => {
      const tour = robotTours[idx]?.tour || [idx];
      const fullPath = buildFullPath(tour, paths);
      return { ...r, path: fullPath, pathIdx: 0, status: fullPath.length > 1 ? 'moving' : 'idle', finished: false };
    }));
    setRunning(true);
  };

  useEffect(() => {
    if (!running) return;
    const posKey = (p) => `${p[0]},${p[1]}`;

    const animate = () => {
      setRobots(prev => {
        let allDone = true, newWaits = 0, newReplans = 0;
        const logs = [];
        const posMap = new Map();
        prev.forEach(r => posMap.set(posKey(r.pos), r.id));

        const updated = prev.map(r => {
          if (r.finished) return r;
          if (r.pathIdx >= r.path.length - 1) return { ...r, status: 'finished', finished: true, waiting: false };
          allDone = false;

          const nextPos = r.path[r.pathIdx + 1];
          const nextKey = posKey(nextPos);
          const occupantId = posMap.get(nextKey);

          if (occupantId !== undefined && occupantId !== r.id) {
            const occupant = prev.find(x => x.id === occupantId);

            if (occupant.finished) {
              const newPath = replanRoute(r, prev, gridRef.current, rows, cols);
              if (newPath && newPath.length > 1) {
                newReplans++;
                logs.push({ step: stats.steps, robot: r.id + 1, action: 'REROUTE', reason: `around parked R${occupantId + 1}` });
                return { ...r, path: newPath, pathIdx: 0, waiting: false, waitCount: 0, status: `rerouting around R${occupantId + 1}` };
              }
            }

            if (r.id > occupantId) {
              newWaits++;
              logs.push({ step: stats.steps, robot: r.id + 1, action: 'WAIT', reason: `R${occupantId + 1} has priority` });
              return { ...r, waiting: true, waitCount: r.waitCount + 1, status: `yielding to R${occupantId + 1}` };
            }

            const otherNext = prev.find(x => x.id === occupantId)?.path?.[prev.find(x => x.id === occupantId).pathIdx + 1];
            if (otherNext && posKey(otherNext) === posKey(r.pos)) {
              if (r.id > occupantId) {
                newWaits++;
                return { ...r, waiting: true, waitCount: r.waitCount + 1, status: `head-on with R${occupantId + 1}` };
              }
            } else {
              newWaits++;
              return { ...r, waiting: true, waitCount: r.waitCount + 1, status: `waiting for R${occupantId + 1}` };
            }
          }

          const headOn = prev.find(o => {
            if (o.id === r.id || o.finished || o.pathIdx >= o.path.length - 1) return false;
            const oNext = o.path[o.pathIdx + 1];
            return posKey(oNext) === posKey(r.pos) && posKey(nextPos) === posKey(o.pos);
          });
          if (headOn && r.id > headOn.id) {
            newWaits++;
            logs.push({ step: stats.steps, robot: r.id + 1, action: 'WAIT', reason: `head-on R${headOn.id + 1}` });
            return { ...r, waiting: true, waitCount: r.waitCount + 1, status: `head-on yield R${headOn.id + 1}` };
          }

          return { ...r, pos: [...nextPos], pathIdx: r.pathIdx + 1, totalDist: r.totalDist + 1, waiting: false, waitCount: 0, status: 'moving' };
        });

        if (logs.length > 0) setCollisionLog(p => [...p.slice(-50), ...logs]);
        setStats(s => ({ ...s, waits: s.waits + newWaits, replans: s.replans + newReplans, steps: s.steps + 1 }));
        if (allDone) setRunning(false);
        return updated;
      });
    };

    animRef.current = setTimeout(animate, speed);
    return () => clearTimeout(animRef.current);
  }, [running, speed, stats.steps, rows, cols]);

  useEffect(() => {
    if (!running) return;
    setPackages(prev => prev.map(p => {
      if (p.collected) return p;
      return robots.find(r => r.pos[0] === p.pos[0] && r.pos[1] === p.pos[1]) ? { ...p, collected: true } : p;
    }));
  }, [robots, running]);

  useEffect(() => {
    setStats(s => ({
      ...s,
      totalDist: robots.reduce((sum, r) => sum + r.totalDist, 0),
      collected: packages.filter(p => p.collected).length
    }));
  }, [robots, packages]);

  return (
    <div className="p-2 bg-gray-900 min-h-screen text-white">
      <h1 className="text-lg font-bold mb-1 text-center">Multi-Robot Warehouse Simulation</h1>
      <p className="text-center text-green-400 text-xs mb-2">✓ A* Pathfinding | TSP/mTSP | Collision Avoidance | Random Generation</p>

      <div className="flex flex-wrap gap-2 mb-2 justify-center items-center">
        <div className="flex gap-1 items-center">
          <label className="text-xs">Mode:</label>
          <select value={mode} onChange={e => setMode(e.target.value)} className="bg-gray-700 px-2 py-1 rounded text-xs" disabled={running}>
            <option value="single">Single Robot TSP</option>
            <option value="multi">Multi-Robot mTSP</option>
          </select>
        </div>
        {mode === 'multi' && (
          <div className="flex gap-1 items-center">
            <label className="text-xs">Robots:</label>
            <select value={numRobots} onChange={e => setNumRobots(+e.target.value)} className="bg-gray-700 px-2 py-1 rounded text-xs" disabled={running}>
              <option value={2}>2</option>
              <option value={3}>3</option>
              <option value={4}>4</option>
              <option value={5}>5</option>
            </select>
          </div>
        )}
        <div className="flex gap-1 items-center">
          <label className="text-xs">Speed:</label>
          <input type="range" min="20" max="150" value={150 - speed} onChange={e => setSpeed(150 - +e.target.value)} className="w-16" />
        </div>
        <button onClick={startSimulation} disabled={running || !envValid} className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-3 py-1 rounded text-xs">
          {running ? 'Running...' : 'Start'}
        </button>
        <button onClick={() => setShowSettings(!showSettings)} className="bg-purple-600 hover:bg-purple-700 px-3 py-1 rounded text-xs">
          {showSettings ? 'Hide Settings' : '⚙ Settings'}
        </button>
      </div>

      {showSettings && (
        <div className="bg-gray-800 border border-gray-600 rounded p-3 mb-2 max-w-2xl mx-auto">
          <h3 className="text-yellow-400 font-bold text-sm mb-2">Environment Settings</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <div>
              <label className="text-xs text-gray-400">Rows</label>
              <input type="number" value={rows} onChange={e => setRows(Math.max(15, Math.min(50, +e.target.value)))} 
                className="w-full bg-gray-700 px-2 py-1 rounded text-xs" disabled={running} />
            </div>
            <div>
              <label className="text-xs text-gray-400">Columns</label>
              <input type="number" value={cols} onChange={e => setCols(Math.max(20, Math.min(60, +e.target.value)))} 
                className="w-full bg-gray-700 px-2 py-1 rounded text-xs" disabled={running} />
            </div>
            <div>
              <label className="text-xs text-gray-400">Shelves</label>
              <input type="number" value={numShelves} onChange={e => setNumShelves(Math.max(1, Math.min(20, +e.target.value)))} 
                className="w-full bg-gray-700 px-2 py-1 rounded text-xs" disabled={running} />
            </div>
            <div>
              <label className="text-xs text-gray-400">Packages</label>
              <input type="number" value={numPackages} onChange={e => setNumPackages(Math.max(5, Math.min(50, +e.target.value)))} 
                className="w-full bg-gray-700 px-2 py-1 rounded text-xs" disabled={running} />
            </div>
          </div>
          <div className="flex gap-2 items-center mb-2">
            <label className="text-xs text-gray-400">Seed:</label>
            <input type="text" value={seed} onChange={e => setSeed(e.target.value)} placeholder="Random"
              className="bg-gray-700 px-2 py-1 rounded text-xs w-24" disabled={running} />
            <span className="text-xs text-gray-500">(Use same seed to reproduce environment)</span>
          </div>
          <div className="flex gap-2">
            <button onClick={() => generateEnvironment(false)} disabled={running}
              className="bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 px-3 py-1 rounded text-xs">
              🎲 Generate Random
            </button>
            <button onClick={() => generateEnvironment(true)} disabled={running || !seed}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-3 py-1 rounded text-xs">
              🔄 Regenerate with Seed
            </button>
            <button onClick={initDefaultWarehouse} disabled={running}
              className="bg-gray-600 hover:bg-gray-700 px-3 py-1 rounded text-xs">
              Reset Default
            </button>
          </div>
          {!envValid && <p className="text-red-400 text-xs mt-2">⚠ Environment invalid - some locations unreachable. Try regenerating.</p>}
        </div>
      )}

      <div className="flex gap-2 justify-center mb-2 text-xs flex-wrap">
        <span className="bg-gray-700 px-2 py-1 rounded">Steps: {stats.steps}</span>
        <span className="bg-gray-700 px-2 py-1 rounded">Distance: {stats.totalDist}</span>
        <span className="bg-gray-700 px-2 py-1 rounded">Collected: {stats.collected}/{packages.length}</span>
        <span className="bg-yellow-700 px-2 py-1 rounded">Waits: {stats.waits}</span>
        <span className="bg-purple-700 px-2 py-1 rounded">Replans: {stats.replans}</span>
        {seed && <span className="bg-gray-600 px-2 py-1 rounded">Seed: {seed}</span>}
      </div>

      <div className="flex justify-center gap-2 overflow-auto">
        <div className="relative border-2 border-gray-600 bg-gray-800 flex-shrink-0" style={{ width: cols * CELL, height: rows * CELL }}>
          {grid.map((row, r) => row.map((cell, c) => (
            <div key={`${r}-${c}`} className="absolute" style={{
              left: c * CELL, top: r * CELL, width: CELL, height: CELL,
              backgroundColor: cell === 1 ? '#4a5568' : 'transparent', border: '1px solid #2d3748'
            }} />
          )))}

          <div className="absolute flex items-center justify-center bg-emerald-800 rounded"
            style={{ left: deposit[1] * CELL + 1, top: deposit[0] * CELL + 1, width: CELL - 2, height: CELL - 2, fontSize: '10px' }}>🏠</div>

          {robots.map((r, i) => (
            <div key={`depot-${i}`} className="absolute" style={{
              left: r.depot[1] * CELL, top: r.depot[0] * CELL, width: CELL, height: CELL,
              border: `2px dashed ${COLORS[i % COLORS.length]}`, opacity: 0.5, borderRadius: '4px'
            }} />
          ))}

          {packages.map((p, i) => (
            <div key={i} className="absolute flex items-center justify-center" style={{
              left: p.pos[1] * CELL, top: p.pos[0] * CELL, width: CELL, height: CELL, opacity: p.collected ? 0.2 : 1
            }}>
              <div className={`w-3 h-3 rounded-sm flex items-center justify-center ${p.collected ? 'bg-gray-600' : 'bg-yellow-500'}`} style={{ fontSize: '8px' }}>
                {p.collected ? '✓' : '📦'}
              </div>
            </div>
          ))}

          {robots.map((r, idx) => (
            <React.Fragment key={`trail-${idx}`}>
              {r.path.slice(Math.max(0, r.pathIdx - 20), r.pathIdx + 1).map((pos, i, arr) => (
                <div key={i} className="absolute rounded-full" style={{
                  left: pos[1] * CELL + CELL / 2 - 2, top: pos[0] * CELL + CELL / 2 - 2,
                  width: 4, height: 4, backgroundColor: COLORS[idx % COLORS.length], opacity: 0.1 + (i / arr.length) * 0.3
                }} />
              ))}
            </React.Fragment>
          ))}

          {robots.map((r, idx) => (
            <div key={idx} className="absolute flex items-center justify-center" style={{
              left: r.pos[1] * CELL + 2, top: r.pos[0] * CELL + 2, width: CELL - 4, height: CELL - 4,
              backgroundColor: r.finished ? '#6b7280' : COLORS[idx % COLORS.length], borderRadius: '50%',
              border: r.waiting ? '2px solid #fbbf24' : '2px solid white',
              boxShadow: r.waiting ? '0 0 10px #fbbf24' : '0 0 6px rgba(255,255,255,0.3)',
              animation: r.waiting ? 'pulse 0.4s infinite' : 'none',
              transition: `left ${speed}ms linear, top ${speed}ms linear`
            }}>
              <span className="font-bold" style={{ fontSize: '9px' }}>{idx + 1}</span>
            </div>
          ))}
        </div>

        <div className="w-48 bg-gray-800 border border-gray-600 rounded p-2 text-xs flex-shrink-0">
          <h3 className="font-bold mb-1 text-yellow-400">Robot Status</h3>
          {robots.map((r, i) => (
            <div key={i} className="mb-1.5 p-1 bg-gray-700 rounded">
              <div className="flex items-center gap-1">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: r.finished ? '#6b7280' : COLORS[i % COLORS.length] }} />
                <span className="font-bold">R{i + 1}</span>
                {r.waiting && <span className="text-yellow-400 ml-auto">⏸</span>}
                {r.finished && <span className="text-green-400 ml-auto">✓</span>}
              </div>
              <div className={`truncate text-xs ${r.waiting ? 'text-yellow-400' : r.finished ? 'text-gray-500' : 'text-green-400'}`}>{r.status}</div>
              <div className="text-gray-500 text-xs">D:{r.totalDist} S:{r.pathIdx}/{r.path.length - 1 || 0}</div>
            </div>
          ))}

          <h3 className="font-bold mt-2 mb-1 text-yellow-400">Event Log</h3>
          <div className="h-24 overflow-y-auto bg-gray-900 rounded p-1">
            {collisionLog.slice(-10).reverse().map((log, i) => (
              <div key={i} className={`py-0.5 text-xs ${log.action === 'REROUTE' ? 'text-purple-400' : 'text-yellow-400'}`}>
                #{log.step} R{log.robot}: {log.action}
              </div>
            ))}
            {collisionLog.length === 0 && <div className="text-gray-600 text-xs">No events</div>}
          </div>
        </div>
      </div>

      <div className="mt-2 flex justify-center gap-3 text-xs flex-wrap">
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-gray-600" /> Shelves</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-yellow-500 rounded-sm" /> Package</div>
        <div className="flex items-center gap-1"><span>🏠</span> Deposit</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-full border-2 border-yellow-400" /> Waiting</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-gray-500" /> Finished</div>
      </div>

      <style>{`@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(0.9); opacity: 0.7; } }`}</style>
    </div>
  );
}
