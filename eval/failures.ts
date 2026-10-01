// Render the images where the focus point lands furthest from the subject.
// Usage: bun eval/failures.ts <dataset-dir> <out-dir> [count]
import {mkdir} from 'node:fs/promises'
import {join} from 'node:path'
import {focusPoint} from '../src/index.ts'
import {render} from './debug.ts'
import {centroid, score} from './metrics.ts'
import {loadDataset} from './run.ts'

const [dir, out, count = '24'] = process.argv.slice(2)
if (!dir || !out) throw new Error('usage: bun eval/failures.ts <dataset-dir> <out-dir> [count]')
await mkdir(out, {recursive: true})
const samples = await loadDataset(dir)
const ranked = samples
  .map(s => ({s, r: score(s.mask, focusPoint(s.image))}))
  .filter(x => !x.r.hit)
  .sort((a, b) => b.r.dist - a.r.dist)
  .slice(0, Number(count))
for (const {s} of ranked) {
  const file = (await Array.fromAsync(new Bun.Glob(`${s.name}.*`).scan(join(dir, 'images'))))[0]!
  await render(join(dir, 'images', file), out, {}, centroid(s.mask))
}
console.log(ranked.map(x => `${x.s.name} ${x.r.dist.toFixed(3)}`).join('\n'))
