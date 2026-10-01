// Evaluate focus point methods against salient object masks.
// Usage: bun eval/run.ts <dataset-dir> [--limit N] [--methods center,smartcrop,focuspoint]
// A dataset dir holds images/*.jpg and masks/*.png with matching basenames.
import sharp from 'sharp'
import smartcropModule from 'smartcrop'
import {basename, join} from 'node:path'
import {readdir} from 'node:fs/promises'
import {parseArgs} from 'node:util'
import {focusPoint, type Options} from '../src/index.ts'
import {load} from './load.ts'
import {centroid, crops, mean, score, type CropName, type Mask, type Point, type Score} from './metrics.ts'

// The published typings only cover the browser entry point
const smartcrop = smartcropModule as unknown as {
  ImgData: new (width: number, height: number, data: Uint8ClampedArray) => unknown
  crop(image: unknown, options: object): Promise<{topCrop: {x: number; y: number; width: number; height: number}}>
}

export interface Sample {
  name: string
  image: {data: Uint8ClampedArray; width: number; height: number}
  mask: Mask
}

/** Load every image/mask pair at evaluation resolution (256px). */
export async function loadDataset(dir: string, limit = Infinity): Promise<Array<Sample>> {
  const masks = new Map<string, string>()
  for (const f of await readdir(join(dir, 'masks')))
    masks.set(f.replace(/\.\w+$/, ''), join(dir, 'masks', f))
  const images = (await readdir(join(dir, 'images')))
    .filter(f => masks.has(f.replace(/\.\w+$/, '')))
    .sort()
    .slice(0, limit)
  return Promise.all(
    images.map(async f => {
      const name = f.replace(/\.\w+$/, '')
      const image = await load(join(dir, 'images', f), 256)
      const raw = await sharp(masks.get(name)!)
        .greyscale()
        .resize(image.width, image.height, {fit: 'fill'})
        .raw()
        .toBuffer()
      const data = new Float32Array(image.width * image.height)
      for (let i = 0; i < data.length; i++) data[i] = raw[i]! / 255
      return {name, image, mask: {width: image.width, height: image.height, data}}
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

async function smartcropCenter(image: Sample['image'], aspect: number, zoom: number): Promise<Point> {
  const w = aspect ? Math.min(image.width, image.height * aspect) : image.width
  const h = aspect ? w / aspect : image.height
  const scale = 1 / zoom
  const {topCrop} = await smartcrop.crop(image, {
    width: w,
    height: h,
    minScale: scale,
    maxScale: scale,
    imageOperations: smartcropOps(image)
  })
  return {
    x: (topCrop.x + topCrop.width / 2) / image.width,
    y: (topCrop.y + topCrop.height / 2) / image.height
  }
}

export type Method = (s: Sample) => Promise<Score> | Score

export const methods: Record<string, Method> = {
  center: s => score(s.mask, {x: 0.5, y: 0.5}),
  // Upper bound: the true centroid of the subject mask
  oracle: s => score(s.mask, centroid(s.mask)),
  smartcrop: async s => {
    // smartcrop optimises a crop per aspect ratio, so give it every crop
    const per = {} as Record<CropName, Point>
    for (const name of Object.keys(crops) as Array<CropName>)
      per[name] = await smartcropCenter(s.image, crops[name].aspect, crops[name].zoom)
    return score(s.mask, per.square, name => per[name])
  },
  focuspoint: s => score(s.mask, focusPoint(s.image))
}

export function withOptions(options: Options): Method {
  return s => score(s.mask, focusPoint(s.image, options))
}

export async function evaluate(samples: Array<Sample>, method: Method) {
  const scores: Array<Score> = []
  for (const s of samples) scores.push(await method(s))
  return mean(scores)
}

export function format(name: string, r: Awaited<ReturnType<typeof evaluate>>) {
  const pct = (v: number) => (v * 100).toFixed(1).padStart(5) + '%'
  const c = Object.values(r.crops)
  const avg = c.reduce((a, b) => a + b, 0) / c.length
  return [
    name.padEnd(14),
    pct(r.hit),
    r.dist.toFixed(3).padStart(6),
    ...c.map(pct),
    pct(avg)
  ].join('  ')
}

export const header = ['method'.padEnd(14), '   hit', '  dist', ...Object.keys(crops).map(k => k.padStart(6)), '   avg'].join('  ')

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
