// Coordinate-descent search over the algorithm's parameters.
// Tunes on even-numbered samples and reports held-out odd-numbered samples.
// Usage: bun eval/tune.ts [--peak 0.5] [--head 0] [--guard 0.015]
//   [--maps eval/predictions/<model>] <dataset-dir>...
// The objective blends peak visibility (weight --peak) and whole-head
// visibility (weight --head) with mean importance kept. With --guard, a
// candidate whose kept falls more than that below the heuristic's on any
// dataset is rejected. With --maps, only subject selection and placement are
// tuned, for those maps.
import {parseArgs} from 'node:util'
import {defaults, focusPoint, type Params} from '../src/index.ts'
import {score} from './metrics.ts'
import {evaluate, format, header, loadDataset, withMaps, withOptions, type Result, type Sample} from './run.ts'

const grid: {[K in keyof Params]?: Array<Params[K]>} = {
  size: [48, 64, 80, 96, 128],
  mbd: [0, 0.5, 1, 1.5, 2],
  border: [0, 0.5, 1, 1.5, 2],
  skin: [0, 0.25, 0.5, 1],
  center: [0, 0.25, 0.5, 0.75, 1],
  blur: [0, 0.02, 0.04, 0.08],
  gamma: [1, 2, 3, 4],
  intensity: [0, 0.25, 0.5, 0.75, 1],
  threshold: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  radius: [0, 0.1, 0.15, 0.2, 0.3, 0.4],
  fit: [0, 0.25, 0.5, 0.75, 1],
  tolerance: [0, 0.005, 0.01, 0.02, 0.05],
  emphasis: [1, 2, 3, 4],
  focus: [0, 0.1, 0.2, 0.3, 0.5],
  protect: [0, 0.3, 0.5, 0.7]
}

const {values, positionals: dirs} = parseArgs({
  allowPositionals: true,
  options: {
    peak: {type: 'string', default: '0.5'},
    head: {type: 'string', default: '0'},
    guard: {type: 'string'},
    maps: {type: 'string'}
  }
})
// Signals that only shape the heuristic's map
const signals: Array<keyof Params> = ['mbd', 'border', 'skin', 'center', 'blur']
if (values.maps) for (const key of signals) delete grid[key]
const method = (p: Params) => (values.maps ? withMaps(values.maps, p) : withOptions(p))
const peakWeight = Number(values.peak)
const headWeight = Number(values.head)
const objective = (r: Result) => r.peak * peakWeight + r.head * headWeight + r.keptAvg * (1 - peakWeight - headWeight)

const sets: Array<{train: Array<Sample>; test: Array<Sample>; floor: number}> = []
for (const dir of dirs) {
  const samples = await loadDataset(dir)
  const train = samples.filter((_, i) => i % 2 === 0)
  // The heuristic's kept on this set, less the allowed drop
  const floor = values.guard ? (await evaluate(train, withOptions(defaults))).keptAvg - Number(values.guard) : 0
  sets.push({train, test: samples.filter((_, i) => i % 2 === 1), floor})
}
const train = sets.flatMap(s => s.train)
const test = sets.flatMap(s => s.test)
console.log(`train ${train.length}, test ${test.length}`)

/** Objective on the training halves, or -Infinity when a guard fails. */
async function fitness(p: Params): Promise<number> {
  for (const s of sets) if (values.guard && (await evaluate(s.train, method(p))).keptAvg < s.floor) return -Infinity
  return objective(await evaluate(train, method(p)))
}

let best: Params = {...defaults}
let bestScore = await fitness(best)
console.log('start', bestScore.toFixed(4), best)
for (let round = 0; round < 3; round++) {
  let improved = false
  for (const key of Object.keys(grid) as Array<keyof Params>) {
    for (const value of grid[key]!) {
      if (value === best[key]) continue
      const candidate = {...best, [key]: value}
      const s = await fitness(candidate)
      if (s > bestScore + 1e-4) {
        best = candidate
        bestScore = s
        improved = true
        console.log(`  ${key}=${value} -> ${s.toFixed(4)}`)
      }
    }
  }
  if (!improved) break
}
console.log('best', best)
console.log(header)
console.log(format('heuristic', await evaluate(test, s => score(s.mask, focusPoint(s.image), s.peak, s.region))))
console.log(format('defaults', await evaluate(test, method(defaults))))
console.log(format('tuned', await evaluate(test, method(best))))
console.log(JSON.stringify(best))
