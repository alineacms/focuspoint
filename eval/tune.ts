// Coordinate-descent search over the algorithm's parameters.
// Tunes on even-numbered samples and reports held-out odd-numbered samples.
// Usage: bun eval/tune.ts [--peak 0.5] <dataset-dir>...
// The objective blends peak visibility (weight --peak) with mean importance kept.
import {parseArgs} from 'node:util'
import {defaults, type Params} from '../src/index.ts'
import {evaluate, format, header, loadDataset, withOptions, type Result, type Sample} from './run.ts'

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
  emphasis: [1, 2, 3, 4],
  focus: [0, 0.1, 0.2, 0.3, 0.5]
}

const {values, positionals: dirs} = parseArgs({
  allowPositionals: true,
  options: {peak: {type: 'string', default: '0.5'}}
})
const peakWeight = Number(values.peak)
const objective = (r: Result) => r.peak * peakWeight + r.keptAvg * (1 - peakWeight)

const all: Array<Sample> = []
for (const dir of dirs) all.push(...(await loadDataset(dir)))
const train = all.filter((_, i) => i % 2 === 0)
const test = all.filter((_, i) => i % 2 === 1)
console.log(`train ${train.length}, test ${test.length}`)

let best: Params = {...defaults}
let bestScore = objective(await evaluate(train, withOptions(best)))
console.log('start', bestScore.toFixed(4), best)
for (let round = 0; round < 3; round++) {
  let improved = false
  for (const key of Object.keys(grid) as Array<keyof Params>) {
    for (const value of grid[key]!) {
      if (value === best[key]) continue
      const candidate = {...best, [key]: value}
      const s = objective(await evaluate(train, withOptions(candidate)))
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
console.log(format('defaults', await evaluate(test, withOptions(defaults))))
console.log(format('tuned', await evaluate(test, withOptions(best))))
console.log(JSON.stringify(best))
