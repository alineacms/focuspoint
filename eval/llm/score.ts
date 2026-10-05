// Score LLM labels against human ground truth, next to the usual baselines.
// Usage: bun eval/llm/score.ts <dataset-dir> <labels.jsonl...>
import {basename} from 'node:path'
import {saliency} from '../../src/index.ts'
import {place} from '../../src/place.ts'
import {score} from '../metrics.ts'
import {evaluate, format, header, loadDataset, methods, type Method, type Sample} from '../run.ts'
import {idealLabel, labelMap, readLabels, type Label} from './labels.ts'

const [dir, ...files] = process.argv.slice(2)
if (!dir || !files.length) throw new Error('Usage: score.ts <dataset-dir> <labels.jsonl...>')
const labels = await readLabels(...files)
const samples = (await loadDataset(dir)).filter(s => labels.has(s.name))
const label = (name: string): Label => labels.get(name)!

function placed(s: Sample, label: Label) {
  const {width, height} = s.mask
  const [x, y] = label.point
  return score(s.mask, place(labelMap(label, width, height), width, height, {x: x / 100, y: y / 100}, 0.01), s.peak)
}

const llm: Record<string, Method> = {
  // The point as the model gave it
  'llm-point': s => {
    const [x, y] = label(s.name).point
    return score(s.mask, {x: x / 100, y: y / 100}, s.peak)
  },
  // The labelled regions as a map, through the library's crop placement
  'llm-map': s => placed(s, label(s.name)),
  // Half LLM regions, half the heuristic's map (cubed, as focusPoint places it)
  'llm-blend': s => {
    const {width, height, data} = saliency(s.image)
    const llm = labelMap(label(s.name), width, height)
    const map = data.map((v, i) => (v ** 3 + llm[i]!) / 2)
    const [x, y] = label(s.name).point
    return score(s.mask, place(map, width, height, {x: x / 100, y: y / 100}, 0.01), s.peak)
  },
  // Upper bound for the label format: cells taken from the ground truth
  'ideal-cells': s => placed(s, idealLabel(s.name, s.mask))
}

console.log(`\n${basename(dir)}: ${samples.length} labelled images`)
console.log(header)
for (const [name, method] of Object.entries({...methods, ...llm})) {
  if (name === 'smartcrop') continue
  console.log(format(name, await evaluate(samples, method)))
}
