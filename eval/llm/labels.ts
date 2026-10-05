import {readFile} from 'node:fs/promises'

export interface Region {
  /** x0, y0, x1, y1 in 0..100 */
  box: [number, number, number, number]
  weight: number
  what?: string
}

export interface Label {
  name: string
  subject?: string
  /** x, y in 0..100 */
  point: [number, number]
  regions: Array<Region>
}

/** A label in grid cells as the model wrote it, see PROMPT.md */
export interface CellLabel {
  name: string
  subject?: string
  point: string
  cells: Record<string, number>
}

const grid = 8

/** Centre of a cell such as `C5`, in 0..100 */
function cell(id: string): [number, number] {
  const c = id.toUpperCase().charCodeAt(0) - 65
  const r = Number(id.slice(1)) - 1
  if (!(c >= 0 && c < grid && r >= 0 && r < grid)) throw new Error(`Bad cell ${id}`)
  return [((c + 0.5) * 100) / grid, ((r + 0.5) * 100) / grid]
}

/**
 * Cells become one region each, slightly larger than the cell so that
 * neighbouring cells blend into one blob rather than leaving dips between.
 */
export function fromCells(label: CellLabel): Label {
  const half = (0.75 * 100) / grid
  return {
    name: label.name,
    subject: label.subject,
    point: cell(label.point),
    regions: Object.entries(label.cells).map(([id, weight]) => {
      const [x, y] = cell(id)
      return {box: [x - half, y - half, x + half, y + half], weight, what: id}
    })
  }
}

export async function readLabels(...files: Array<string>): Promise<Map<string, Label>> {
  const labels = new Map<string, Label>()
  for (const file of files)
    for (const [i, line] of (await readFile(file, 'utf8')).split('\n').entries()) {
      if (!line.trim()) continue
      try {
        const label = JSON.parse(line) as Label | CellLabel
        labels.set(label.name, 'cells' in label ? fromCells(label) : label)
      } catch (e) {
        throw new Error(`${file}:${i + 1}: ${(e as Error).message}`)
      }
    }
  return labels
}

/**
 * Importance map from labelled regions: a soft elliptical blob per box, at
 * its weight in the middle and about 0.14 at its edge, combined by max so
 * overlapping boxes (a face inside a body) don't add up.
 */
export function labelMap(label: Label, width: number, height: number): Float32Array {
  const map = new Float32Array(width * height)
  for (const {box, weight} of label.regions) {
    const [x0, y0, x1, y1] = box.map(v => Math.min(100, Math.max(0, v)) / 100) as [number, number, number, number]
    const cx = ((x0 + x1) / 2) * width, cy = ((y0 + y1) / 2) * height
    const rx = Math.max(0.02 * width, (Math.abs(x1 - x0) / 2) * width)
    const ry = Math.max(0.02 * height, (Math.abs(y1 - y0) / 2) * height)
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const dx = (x + 0.5 - cx) / rx, dy = (y + 0.5 - cy) / ry
        const v = weight * Math.exp(-2 * (dx * dx + dy * dy))
        const i = y * width + x
        if (v > map[i]!) map[i] = v
      }
  }
  return map
}

/**
 * The best labels this format allows for a ground truth map: every cell
 * holding at least `keep` of the strongest cell's importance, weighted by it.
 */
export function idealLabel(name: string, map: {width: number; height: number; data: Float32Array}, keep = 0.3): Label {
  const {width, height, data} = map
  const sums = new Float64Array(grid * grid)
  const counts = new Float64Array(grid * grid)
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const c = Math.min(grid - 1, Math.floor((x / width) * grid)) + grid * Math.min(grid - 1, Math.floor((y / height) * grid))
      sums[c]! += data[y * width + x]!
      counts[c]!++
    }
  const means = Array.from(sums, (v, i) => v / (counts[i] || 1))
  const max = Math.max(...means)
  const id = (i: number) => String.fromCharCode(65 + (i % grid)) + (Math.floor(i / grid) + 1)
  const cells: Record<string, number> = {}
  for (const [i, m] of means.entries()) if (m >= keep * max) cells[id(i)] = m / max
  return fromCells({name, cells, point: id(means.indexOf(max))})
}
