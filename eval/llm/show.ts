// Draw labelled regions (yellow, thicker = heavier) and the point (red) on
// the images, to check labels by eye.
// Usage: bun eval/llm/show.ts <out-dir> <dataset-dir> <labels.jsonl...>
import sharp from 'sharp'
import {mkdir} from 'node:fs/promises'
import {join} from 'node:path'
import {readLabels} from './labels.ts'

const [out, dir, ...files] = process.argv.slice(2)
if (!out || !dir || !files.length) throw new Error('Usage: show.ts <out-dir> <dataset-dir> <labels.jsonl...>')
await mkdir(out, {recursive: true})
for (const label of (await readLabels(...files)).values()) {
  const img = await sharp(join(dir, 'images', `${label.name}.jpg`)).rotate().resize(480, 480, {fit: 'inside'}).toBuffer({resolveWithObject: true})
  const {width: w, height: h} = img.info
  const boxes = label.regions.map(({box: [x0, y0, x1, y1], weight}) =>
    `<rect x="${(x0 * w) / 100}" y="${(y0 * h) / 100}" width="${((x1 - x0) * w) / 100}" height="${((y1 - y0) * h) / 100}" fill="none" stroke="#ff0" stroke-width="${1 + 3 * weight}"/>`
  )
  const [px, py] = label.point
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">${boxes.join('')}<circle cx="${(px * w) / 100}" cy="${(py * h) / 100}" r="7" fill="#f00" stroke="#fff" stroke-width="2"/></svg>`
  await sharp(img.data).composite([{input: Buffer.from(svg)}]).jpeg().toFile(join(out, `${label.name}.jpg`))
}
