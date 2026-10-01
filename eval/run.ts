// Evaluate focus point methods against salient object masks.
// Usage: bun eval/run.ts <dataset-dir> [--limit N] [--methods center,smartcrop,focuspoint]
// A dataset dir holds images/*.jpg and masks/*.png with matching basenames,
// or fixations/*.png density maps (scaled so their peak is 1).
import sharp from 'sharp'
import smartcropModule from 'smartcrop'
import {basename, join} from 'node:path'
import {existsSync} from 'node:fs'
import {readdir} from 'node:fs/promises'
import {parseArgs} from 'node:util'
import {focusPoint, type Options} from '../src/index.ts'
import {load} from './load.ts'
import {containers, mean, oracle, peak, score, type Mask, type Point, type Score} from './metrics.ts'

// The published typings only cover the browser entry point
const smartcrop = smartcropModule as unknown as {
  ImgData: new (width: number, height: number, data: Uint8ClampedArray) => unknown
  crop(image: unknown, options: object): Promise<{topCrop: {x: number; y: number; width: number; height: number}}>
}

export interface Sample {
  name: string
  image: {data: Uint8ClampedArray; width: number; height: number}
  mask: Mask
  /** Most important spot of the mask, precomputed. */
  peak: Point
}

/** Load every image/mask pair at evaluation resolution (256px). */
export async function loadDataset(dir: string, limit = Infinity): Promise<Array<Sample>> {
  const masks = new Map<string, string>()
  const fixations = existsSync(join(dir, 'fixations'))
  const gt = join(dir, fixations ? 'fixations' : 'masks')
  for (const f of await readdir(gt)) masks.set(f.replace(/\.\w+$/, ''), join(gt, f))
  const images = (await readdir(join(dir, 'images')))
    .filter(f => masks.has(f.replace(/\.\w+$/, '')))
    .sort()
  // Spread a limited sample evenly over the dataset
  const step = Math.max(1, images.length / limit)
  const picked = images.length > limit ? Array.from({length: limit}, (_, i) => images[Math.floor(i * step)]!) : images
  return Promise.all(
    picked.map(async f => {
      const name = f.replace(/\.\w+$/, '')
      const image = await load(join(dir, 'images', f), 256)
      const raw = await sharp(masks.get(name)!)
        .greyscale()
        .resize(image.width, image.height, {fit: 'fill'})
        .raw()
        .toBuffer()
      const data = new Float32Array(image.width * image.height)
      let max = 255
      if (fixations) {
        max = 1
        for (let i = 0; i < data.length; i++) if (raw[i]! > max) max = raw[i]!
      }
      for (let i = 0; i < data.length; i++) data[i] = raw[i]! / max
      const mask = {width: image.width, height: image.height, data}
      return {name, image, mask, peak: peak(mask)}
    })
  )
}

function smartcropOps(image: Sample['image']) {
  const img = new smartcrop.ImgData(image.width, image.height, image.data)
  return {
    open: async () => img,
    resample: async () => img,
    getData: async () => img
  }
}

/** The focus point exactly as Alinea derives it today: the centre of smartcrop's best square crop. */
async function smartcropPoint(image: Sample['image']): Promise<Point> {
  const {topCrop} = await smartcrop.crop(image, {
    width: 100,
    height: 100,
    imageOperations: smartcropOps(image)
  })
  return {
    x: (topCrop.x + topCrop.width / 2) / image.width,
    y: (topCrop.y + topCrop.height / 2) / image.height
  }
}

export type Method = (s: Sample) => Promise<Score> | Score

export const methods: Record<string, Method> = {
  center: s => score(s.mask, {x: 0.5, y: 0.5}, s.peak),
  // Upper bound: the best single point given the ground truth
  oracle: s => score(s.mask, oracle(s.mask), s.peak),
  smartcrop: async s => score(s.mask, await smartcropPoint(s.image), s.peak),
  focuspoint: s => score(s.mask, focusPoint(s.image), s.peak)
}

export function withOptions(options: Options): Method {
  return s => score(s.mask, focusPoint(s.image, options), s.peak)
}

export async function evaluate(samples: Array<Sample>, method: Method) {
  const scores: Array<Score> = []
  for (const s of samples) scores.push(await method(s))
  return mean(scores)
}

export type Result = Awaited<ReturnType<typeof evaluate>>

export function format(name: string, r: Result) {
  const pct = (v: number) => (v * 100).toFixed(1).padStart(6) + '%'
  return [name.padEnd(12), pct(r.hit), pct(r.peak), ...Object.values(r.kept).map(pct), pct(r.keptAvg)].join(' ')
}

export const header = [
  'method'.padEnd(12),
  ...['hit', 'peak', ...Object.keys(containers), 'kept'].map(k => k.padStart(7))
].join(' ')

if (import.meta.main) {
  const {values, positionals} = parseArgs({
    allowPositionals: true,
    options: {
      limit: {type: 'string'},
      methods: {type: 'string', default: 'center,smartcrop,focuspoint'}
    }
  })
  for (const dir of positionals) {
    const t0 = performance.now()
    const samples = await loadDataset(dir, values.limit ? Number(values.limit) : Infinity)
    console.log(`\n${basename(dir)}: ${samples.length} images (loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s)`)
    console.log(header)
    for (const name of values.methods.split(',')) {
      const t = performance.now()
      const r = await evaluate(samples, methods[name]!)
      const ms = (performance.now() - t) / samples.length
      console.log(format(name, r), ` ${ms.toFixed(1)}ms/img`)
    }
  }
}
