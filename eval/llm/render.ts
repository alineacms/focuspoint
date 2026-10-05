// Render images with an 8x8 grid of named cells (A1..H8) for a vision LLM to
// label. Small models locate things far more reliably by reading a printed
// cell name than by estimating coordinates, even with a labelled ruler.
// Usage: bun eval/llm/render.ts <out-dir> <images...>
import sharp from 'sharp'
import {mkdir} from 'node:fs/promises'
import {basename, join} from 'node:path'

const size = 640
const cols = 'ABCDEFGH'

export async function render(file: string, out: string): Promise<void> {
  const img = await sharp(file).rotate().resize(size, size, {fit: 'inside'}).toBuffer({resolveWithObject: true})
  const {width, height} = img.info
  const n = cols.length
  const parts: Array<string> = []
  for (let k = 1; k < n; k++) {
    const x = (k / n) * width, y = (k / n) * height
    parts.push(
      `<line x1="${x}" y1="0" x2="${x}" y2="${height}" stroke="#000" stroke-width="3" stroke-opacity="0.5"/>`,
      `<line x1="${x}" y1="0" x2="${x}" y2="${height}" stroke="#fff" stroke-width="1"/>`,
      `<line x1="0" y1="${y}" x2="${width}" y2="${y}" stroke="#000" stroke-width="3" stroke-opacity="0.5"/>`,
      `<line x1="0" y1="${y}" x2="${width}" y2="${y}" stroke="#fff" stroke-width="1"/>`
    )
  }
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      const x = (c / n) * width + 2, y = (r / n) * height + 2
      parts.push(
        `<rect x="${x}" y="${y}" width="22" height="14" fill="#000" fill-opacity="0.6"/>`,
        `<text x="${x + 11}" y="${y + 11}" text-anchor="middle">${cols[c]}${r + 1}</text>`
      )
    }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" font-family="Helvetica, Arial" font-size="11" font-weight="bold" fill="#fff">${parts.join('')}</svg>`
  await sharp(img.data).composite([{input: Buffer.from(svg)}]).jpeg({quality: 88}).toFile(out)
}

if (import.meta.main) {
  const [dir, ...files] = process.argv.slice(2)
  if (!dir || !files.length) throw new Error('Usage: render.ts <out-dir> <images...>')
  await mkdir(dir, {recursive: true})
  for (const f of files) await render(f, join(dir, basename(f).replace(/\.\w+$/, '.jpg')))
}
