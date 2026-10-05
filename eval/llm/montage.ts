// Tile images into one sheet for quick review.
// Usage: bun eval/llm/montage.ts <out.jpg> <images...>
import sharp from 'sharp'

const [out, ...files] = process.argv.slice(2)
if (!out || !files.length) throw new Error('Usage: montage.ts <out.jpg> <images...>')
const cell = 320, cols = Number(process.env.COLS ?? 4)
const rows = Math.ceil(files.length / cols)
const tiles = await Promise.all(
  files.map(async (f, i) => ({
    input: await sharp(f).resize(cell, cell, {fit: 'contain', background: '#222'}).toBuffer(),
    left: (i % cols) * cell,
    top: Math.floor(i / cols) * cell
  }))
)
await sharp({create: {width: cols * cell, height: rows * cell, channels: 3, background: '#222'}}).composite(tiles).jpeg({quality: 80}).toFile(out)
