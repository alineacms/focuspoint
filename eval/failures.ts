// Render the images where the focus point loses the most importance
// compared to the best possible point.
// Usage: bun eval/failures.ts <dataset-dir> <out-dir> [count]
import {mkdir} from 'node:fs/promises'
import {join} from 'node:path'
import {focusPoint} from '../src/index.ts'
import {render} from './debug.ts'
import {oracle, score} from './metrics.ts'
import {loadDataset} from './run.ts'

const [dir, out, count = '24'] = process.argv.slice(2)
if (!dir || !out) throw new Error('usage: bun eval/failures.ts <dataset-dir> <out-dir> [count]')
await mkdir(out, {recursive: true})
const samples = await loadDataset(dir)
const avg = (r: ReturnType<typeof score>) => {
  const v = Object.values(r.kept)
  return v.reduce((a, b) => a + b, 0) / v.length
}
const ranked = samples
  .map(s => ({s, loss: avg(score(s.mask, oracle(s.mask), s.peak)) - avg(score(s.mask, focusPoint(s.image), s.peak))}))
  .sort((a, b) => b.loss - a.loss)
  .slice(0, Number(count))
for (const {s} of ranked) {
  const file = (await Array.fromAsync(new Bun.Glob(`${s.name}.*`).scan(join(dir, 'images'))))[0]!
  await render(join(dir, 'images', file), out, {}, oracle(s.mask))
}
console.log(ranked.map(x => `${x.s.name} ${x.loss.toFixed(3)}`).join('\n'))
