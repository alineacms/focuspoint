// Turn the ideal points marked on the website (site/labels.json) into a
// dataset: the photos plus a small disc mask at each marked point. Scored
// with run.ts, peak is then how often the five crops keep the marked point.
// Usage: bun eval/marks.ts  ->  eval/data/MARKS
import sharp from 'sharp'
import {mkdir, symlink, rm} from 'node:fs/promises'
import {resolve} from 'node:path'

const labels = (await Bun.file('site/labels.json').json()) as Record<string, {x: number; y: number}>
const dir = 'eval/data/MARKS'
await rm(dir, {recursive: true, force: true})
await mkdir(`${dir}/images`, {recursive: true})
await mkdir(`${dir}/masks`, {recursive: true})
for (const [id, p] of Object.entries(labels)) {
  const photo = resolve(`site/dist/photos/${id}.jpg`)
  await symlink(photo, `${dir}/images/${id}.jpg`)
  const {width = 0, height = 0} = await sharp(photo).metadata()
  const r = Math.max(width, height) * 0.02
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="100%" fill="#000"/><circle cx="${p.x * width}" cy="${p.y * height}" r="${r}" fill="#fff"/></svg>`
  await sharp(Buffer.from(svg)).greyscale().png().toFile(`${dir}/masks/${id}.png`)
}
console.log(`${dir}: ${Object.keys(labels).length} marked photos`)
