// List images where LLM labels lose most against the centre point.
// Usage: bun eval/llm/worst.ts <dataset-dir> <labels.jsonl...>
import {place} from '../../src/place.ts'
import {mean, score} from '../metrics.ts'
import {loadDataset} from '../run.ts'
import {labelMap, readLabels} from './labels.ts'

const [dir, ...files] = process.argv.slice(2)
if (!dir || !files.length) throw new Error('Usage: worst.ts <dataset-dir> <labels.jsonl...>')
const labels = await readLabels(...files)
const rows = (await loadDataset(dir)).filter(s => labels.has(s.name)).map(s => {
  const label = labels.get(s.name)!
  const {width, height} = s.mask
  const at = place(labelMap(label, width, height), width, height, {x: label.point[0] / 100, y: label.point[1] / 100}, 0.01)
  const k = (p: {x: number; y: number}) => mean([score(s.mask, p, s.peak)]).keptAvg
  return {name: s.name, subject: label.subject, llm: k(at), center: k({x: 0.5, y: 0.5}), at}
})
rows.sort((a, b) => a.llm - a.center - (b.llm - b.center))
const pct = (v: number) => (v * 100).toFixed(1)
for (const r of rows.slice(0, 16)) console.log(r.name, pct(r.llm), pct(r.center), r.subject)
const wins = rows.filter(r => r.llm > r.center + 0.01).length
const losses = rows.filter(r => r.llm < r.center - 0.01).length
console.log(`wins ${wins}, losses ${losses}, ties ${rows.length - wins - losses}`)
