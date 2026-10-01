// Render the saliency map and focus point next to each input image.
// Usage: bun eval/debug.ts out-dir image...
import sharp from 'sharp'
import {basename} from 'node:path'
import {mkdir} from 'node:fs/promises'
import {focusPoint, saliency, type Options} from '../src/index.ts'
import {load} from './load.ts'

export async function render(file: string, out: string, options: Options = {}, truth?: {x: number; y: number}) {
  const img = await load(file, 256)
  const map = saliency(img, options)
  const fp = focusPoint(img, options)
  const {width: w, height: h} = img
  const gray = Buffer.alloc(map.width * map.height)
  for (let i = 0; i < gray.length; i++) gray[i] = Math.round(map.data[i]! * 255)
  const mapPng = await sharp(gray, {raw: {width: map.width, height: map.height, channels: 1}})
    .resize(w, h, {kernel: 'nearest'})
    .png()
    .toBuffer()
  const mark = (x: number, y: number, color: string) =>
    `<circle cx="${x * w}" cy="${y * h}" r="6" fill="none" stroke="${color}" stroke-width="3"/>`
  const svg = `<svg width="${w * 2}" height="${h}" xmlns="http://www.w3.org/2000/svg">
    <rect x="${fp.box.x * w}" y="${fp.box.y * h}" width="${fp.box.width * w}" height="${fp.box.height * h}" fill="none" stroke="yellow" stroke-width="1"/>
    ${mark(fp.x, fp.y, 'red')}${truth ? mark(truth.x, truth.y, 'lime') : ''}
    <text x="4" y="14" font-size="12" fill="white" stroke="black" stroke-width="0.5">c=${fp.confidence.toFixed(2)}</text>
  </svg>`
  await sharp({create: {width: w * 2, height: h, channels: 3, background: '#000'}})
    .composite([
      {input: Buffer.from(img.data.buffer), raw: {width: w, height: h, channels: 4}, left: 0, top: 0},
      {input: mapPng, left: w, top: 0},
      {input: Buffer.from(svg), left: 0, top: 0}
    ])
    .jpeg()
    .toFile(`${out}/${basename(file).replace(/\.\w+$/, '')}.jpg`)
}

if (import.meta.main) {
  const [out, ...files] = process.argv.slice(2)
  if (!out || !files.length) throw new Error('usage: bun eval/debug.ts out-dir image...')
  await mkdir(out, {recursive: true})
  for (const file of files) await render(file, out)
}
