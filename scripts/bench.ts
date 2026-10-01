// Time focusPoint on a set of images at a few input sizes.
// Usage: bun scripts/bench.ts image...
import {focusPoint} from '../src/index.ts'
import {load} from '../eval/load.ts'

const files = process.argv.slice(2)
for (const max of [160, 640, 2048]) {
  const images = await Promise.all(files.map(f => load(f, max)))
  for (const img of images) focusPoint(img) // warm up
  const t = performance.now()
  const rounds = 5
  for (let r = 0; r < rounds; r++) for (const img of images) focusPoint(img)
  const ms = (performance.now() - t) / (rounds * images.length)
  console.log(`${String(max).padStart(4)} px input: ${ms.toFixed(2)} ms/image`)
}
