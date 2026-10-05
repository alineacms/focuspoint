// Time focusPoint, heuristic and model, on a set of images at a few input sizes.
// Usage: bun scripts/bench.ts image...
import {focusPoint} from '../src/index.ts'
import * as heuristic from '../eval/heuristic.ts'
import {load} from '../eval/load.ts'

const files = process.argv.slice(2)
for (const [name, f] of [['heuristic', heuristic.focusPoint], ['model', focusPoint]] as const)
  for (const max of [160, 640, 2048]) {
    const images = await Promise.all(files.map(file => load(file, max)))
    for (const img of images) f(img) // warm up
    const t = performance.now()
    const rounds = 5
    for (let r = 0; r < rounds; r++) for (const img of images) f(img)
    const ms = (performance.now() - t) / (rounds * images.length)
    console.log(`${name.padEnd(10)} ${String(max).padStart(4)} px input: ${ms.toFixed(2)} ms/image`)
  }
