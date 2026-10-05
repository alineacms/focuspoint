/**
 * Ground truth importance per pixel, 0..1: a subject mask, or a human
 * attention map scaled so its peak is 1.
 */
export interface Mask {
  width: number
  height: number
  data: Float32Array
}

export interface Point {
  x: number
  y: number
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

/**
 * Container aspect ratios the image is shown in. Each is filled with
 * `object-fit: cover`, so only one axis of the image gets cropped.
 */
export const containers = {
  banner: 3,
  wide: 16 / 9,
  square: 1,
  portrait: 4 / 5,
  story: 9 / 16
} as const

export type ContainerName = keyof typeof containers

/**
 * How a frontend turns a stored focus point into a crop:
 * - `center`: the window is centred on the point and clamped to the image,
 *   as image CDNs and most crop tools do. This is what we score.
 * - `css`: `object-fit: cover; object-position: x% y%`, which aligns the
 *   point proportionally. Kept for comparison only.
 */
export type Mode = 'center' | 'css'
export const modes: ReadonlyArray<Mode> = ['center']

/** The visible part of an image filling a container of `aspect`. */
export function cover(p: Point, width: number, height: number, aspect: number, mode: Mode): Rect {
  let w = width
  let h = width / aspect
  if (h > height) {
    h = height
    w = height * aspect
  }
  if (mode === 'css') return {x: p.x * (width - w), y: p.y * (height - h), width: w, height: h}
  const x = Math.min(width - w, Math.max(0, p.x * width - w / 2))
  const y = Math.min(height - h, Math.max(0, p.y * height - h / 2))
  return {x, y, width: w, height: h}
}

/** Share of the mask's total importance inside the rectangle. */
export function retained(mask: Mask, r: Rect): number {
  let total = 0
  let inside = 0
  const x0 = Math.round(r.x), x1 = Math.round(r.x + r.width)
  const y0 = Math.round(r.y), y1 = Math.round(r.y + r.height)
  for (let y = 0; y < mask.height; y++)
    for (let x = 0; x < mask.width; x++) {
      const v = mask.data[y * mask.width + x]!
      total += v
      if (x >= x0 && x < x1 && y >= y0 && y < y1) inside += v
    }
  return total ? inside / total : 1
}

/**
 * Location of the most important spot: the peak of a lightly blurred mask.
 * Subject masks are flat, so their whole interior ties for the peak; ties
 * (within 1%) go to the tied position nearest the middle of the tied area,
 * the heart of the subject, rather than to whichever is scanned first.
 */
export function peak(mask: Mask): Point {
  const {width, height, data} = mask
  const r = Math.max(1, Math.round(Math.max(width, height) / 40))
  const blurred: Array<{x: number; y: number; s: number}> = []
  let best = -1
  for (let y = 0; y < height; y += 2)
    for (let x = 0; x < width; x += 2) {
      let s = 0
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          const yy = Math.min(height - 1, Math.max(0, y + dy))
          const xx = Math.min(width - 1, Math.max(0, x + dx))
          s += data[yy * width + xx]!
        }
      blurred.push({x, y, s})
      if (s > best) best = s
    }
  const tied = blurred.filter(b => b.s >= best * 0.99)
  const cx = tied.reduce((a, b) => a + b.x, 0) / tied.length
  const cy = tied.reduce((a, b) => a + b.y, 0) / tied.length
  let at = tied[0]!
  for (const b of tied) if ((b.x - cx) ** 2 + (b.y - cy) ** 2 < (at.x - cx) ** 2 + (at.y - cy) ** 2) at = b
  return {x: (at.x + 0.5) / width, y: (at.y + 0.5) / height}
}

/**
 * The most important region: the area connected to the peak of the lightly
 * blurred mask that stays above half of it. For attention this is the face
 * or object most people looked at; for a flat subject mask, the subject.
 */
export function topRegion(mask: Mask, top: Point = peak(mask)): Rect {
  const {width, height, data} = mask
  const start = Math.min(height - 1, Math.floor(top.y * height)) * width + Math.min(width - 1, Math.floor(top.x * width))
  const t = data[start]! * 0.5
  const seen = new Uint8Array(data.length)
  const stack = [start]
  seen[start] = 1
  let x0 = width, y0 = height, x1 = 0, y1 = 0
  while (stack.length) {
    const i = stack.pop()!
    const x = i % width
    const y = (i - x) / width
    x0 = Math.min(x0, x)
    x1 = Math.max(x1, x)
    y0 = Math.min(y0, y)
    y1 = Math.max(y1, y)
    for (const j of [x > 0 ? i - 1 : -1, x < width - 1 ? i + 1 : -1, y > 0 ? i - width : -1, y < height - 1 ? i + width : -1])
      if (j >= 0 && !seen[j] && data[j]! >= t) {
        seen[j] = 1
        stack.push(j)
      }
  }
  return {x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1}
}

export interface Score {
  /** Point lies on the subject (mask >= 0.5). */
  hit: number
  /** Share of importance that stays visible, per container. */
  kept: Record<ContainerName, number>
  /** Share of crops in which the most important spot stays visible. */
  peak: number
  /** Share of crops that keep the whole most important region (a head). */
  head: number
}

export function score(mask: Mask, p: Point, top: Point = peak(mask), region: Rect = topRegion(mask, top)): Score {
  const px = Math.min(mask.width - 1, Math.floor(p.x * mask.width))
  const py = Math.min(mask.height - 1, Math.floor(p.y * mask.height))
  const kept = {} as Record<ContainerName, number>
  let peaks = 0
  let heads = 0
  const names = Object.keys(containers) as Array<ContainerName>
  const tx = top.x * mask.width
  const ty = top.y * mask.height
  for (const name of names) {
    kept[name] = 0
    for (const mode of modes) {
      const r = cover(p, mask.width, mask.height, containers[name], mode)
      kept[name] += retained(mask, r) / modes.length
      if (tx >= r.x && tx <= r.x + r.width && ty >= r.y && ty <= r.y + r.height) peaks++
      // Whole region inside, allowing a pixel for rounding
      if (region.x >= r.x - 1 && region.x + region.width <= r.x + r.width + 1 && region.y >= r.y - 1 && region.y + region.height <= r.y + r.height + 1) heads++
    }
  }
  return {
    hit: mask.data[py * mask.width + px]! >= 0.5 ? 1 : 0,
    kept,
    peak: peaks / (names.length * modes.length),
    head: heads / (names.length * modes.length)
  }
}

export function mean(scores: Array<Score>) {
  const n = scores.length
  const kept = {} as Record<ContainerName, number>
  for (const name of Object.keys(containers) as Array<ContainerName>)
    kept[name] = scores.reduce((s, x) => s + x.kept[name], 0) / n
  const all = Object.values(kept)
  return {
    n,
    hit: scores.reduce((s, x) => s + x.hit, 0) / n,
    peak: scores.reduce((s, x) => s + x.peak, 0) / n,
    head: scores.reduce((s, x) => s + x.head, 0) / n,
    kept,
    keptAvg: all.reduce((a, b) => a + b, 0) / all.length
  }
}

/**
 * The best possible point for this mask: searched exhaustively on a grid,
 * maximising average retention. Since each container crops one axis only,
 * x and y can be optimised independently.
 */
export function oracle(mask: Mask): Point {
  const {width, height, data} = mask
  const cols = new Float64Array(width + 1)
  const rows = new Float64Array(height + 1)
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const v = data[y * width + x]!
      cols[x + 1]! += v
      rows[y + 1]! += v
    }
  for (let i = 1; i <= width; i++) cols[i]! += cols[i - 1]!
  for (let i = 1; i <= height; i++) rows[i]! += rows[i - 1]!
  const best = (prefix: Float64Array, size: number, horizontal: boolean) => {
    let top = -1
    let at = 0.5
    for (let k = 0; k <= 100; k++) {
      const t = k / 100
      let s = 0
      for (const aspect of Object.values(containers))
        for (const mode of modes) {
          const r = cover(horizontal ? {x: t, y: 0.5} : {x: 0.5, y: t}, width, height, aspect, mode)
          const a = Math.round(horizontal ? r.x : r.y)
          const b = Math.round(horizontal ? r.x + r.width : r.y + r.height)
          s += prefix[Math.min(size, b)]! - prefix[Math.max(0, a)]!
        }
      // Prefer the point closest to the centre among equals
      if (s > top + 1e-9 || (Math.abs(s - top) <= 1e-9 && Math.abs(t - 0.5) < Math.abs(at - 0.5))) {
        top = s
        at = t
      }
    }
    return at
  }
  return {x: best(cols, width, true), y: best(rows, height, false)}
}
