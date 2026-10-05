// How often crops keep heads whole, on the images with detected faces
// (<dataset>/faces.json from train/faces.py). A crop that cuts through a
// head is the worst crop there is, so this is scored strictly: the whole
// head box must be inside.
// Usage: bun eval/faces.ts <dataset-dir>... [--methods heuristic,model] [--maps eval/predictions/<model>]
import {readFile} from 'node:fs/promises'
import {basename, join} from 'node:path'
import {parseArgs} from 'node:util'
import {focusPoint, locate, type Options} from '../src/index.ts'
import * as heuristic from './heuristic.ts'
import {containers, cover, type Point} from './metrics.ts'
import {fit, loadDataset, readMap, type Sample} from './run.ts'

type Head = [number, number, number, number, number]

const {values, positionals} = parseArgs({
  allowPositionals: true,
  options: {
    methods: {type: 'string', default: 'ideal,center,heuristic,model'},
    maps: {type: 'string', multiple: true, default: []},
    options: {type: 'string', default: '{}'}
  }
})
const options = JSON.parse(values.options) as Options

const methods: Record<string, (s: Sample) => Promise<Point> | Point> = {
  center: () => ({x: 0.5, y: 0.5}),
  heuristic: s => heuristic.focusPoint(s.image),
  model: s => focusPoint(s.image, options)
}
let current: Record<string, Array<Head>> = {}
// Upper bound for the main head: centring on it keeps it whenever it fits
methods.ideal = s => {
  const [x0, y0, x1, y1] = current[s.name]!.reduce((a, b) => (area(b) > area(a) ? b : a))
  return {x: (x0 + x1) / 2, y: (y0 + y1) / 2}
}
for (const dir of values.maps)
  methods[basename(dir)] = async s => locate(fit(await readMap(join(dir, s.set, `${s.name}.f32`)), s.image), options)

/** Share of the five crops around p that contain the head whole. */
function keeps(s: Sample, p: Point, [x0, y0, x1, y1]: Head): number {
  const {width: w, height: h} = s.image
  let n = 0
  for (const aspect of Object.values(containers)) {
    const r = cover(p, w, h, aspect, 'center')
    if (x0 * w >= r.x - 1 && x1 * w <= r.x + r.width + 1 && y0 * h >= r.y - 1 && y1 * h <= r.y + r.height + 1) n++
  }
  return n / 5
}

const area = ([x0, y0, x1, y1]: Head) => (x1 - x0) * (y1 - y0)
const pct = (v: number) => `${(v * 100).toFixed(1)}%`.padStart(8)

for (const dir of positionals) {
  const faces = (current = JSON.parse(await readFile(join(dir, 'faces.json'), 'utf8')) as Record<string, Array<Head>>)
  const samples = (await loadDataset(dir)).filter(s => faces[s.name])
  console.log(`\n${basename(dir)}: ${samples.length} images with faces`)
  console.log(`${'method'.padEnd(14)}${'main head'.padStart(10)}${'all heads'.padStart(10)}`)
  for (const name of values.methods.split(',').concat(values.maps.map(d => basename(d)))) {
    const method = methods[name]
    if (!method) continue
    let main = 0, all = 0
    for (const s of samples) {
      const p = await method(s)
      const heads = faces[s.name]!
      const biggest = heads.reduce((a, b) => (area(b) > area(a) ? b : a))
      main += keeps(s, p, biggest)
      all += Math.min(...heads.map(h => keeps(s, p, h)))
    }
    console.log(`${name.padEnd(14)}${pct(main / samples.length)}  ${pct(all / samples.length)}`)
  }
}
