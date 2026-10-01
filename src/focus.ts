import {half, toLab, type ImageDataLike} from './image.ts'
import {borderContrast, boxBlur, globalContrast, mbd, normalize, sharpness} from './saliency.ts'

export interface Box {
  x: number
  y: number
  width: number
  height: number
}

export interface FocusPoint {
  /** Horizontal position of the subject, 0 (left) to 1 (right). */
  x: number
  /** Vertical position of the subject, 0 (top) to 1 (bottom). */
  y: number
  /** Bounding box of the subject in the same 0..1 coordinates. */
  box: Box
  /** How clearly a subject stands out from its surroundings, 0..1. */
  confidence: number
}

export interface SaliencyMap {
  width: number
  height: number
  /** Importance per pixel, 0..1. */
  data: Float32Array
}

/** Tuning knobs; the defaults are fitted on the evaluation set. */
export interface Params {
  /** Longest side of the working image in pixels. */
  size: number
  /** Weight of the minimum barrier distance map. */
  mbd: number
  /** Weight of the border colour contrast map. */
  border: number
  /** Weight of the global colour contrast map. */
  contrast: number
  /** Weight of the detail (in-focus) map. */
  sharpness: number
  /** Strength of the centre prior, 0 disables it. */
  center: number
  /** Blur radius applied to the combined map, relative to `size`. */
  blur: number
  /** Raise the map to this power before taking the centroid. */
  gamma: number
  /**
   * Regions scoring at least this fraction of the strongest region also pull
   * the point towards them. 1 uses only the strongest region.
   */
  merge: number
}

export const defaults: Params = {
  size: 80,
  mbd: 1,
  border: 1,
  contrast: 0,
  sharpness: 0,
  center: 0.5,
  blur: 0.02,
  gamma: 2,
  merge: 1
}

export interface Options extends Partial<Params> {}

export function saliency(image: ImageDataLike, options: Options = {}): SaliencyMap {
  const p = {...defaults, ...options}
  const hi = toLab(image, p.size * 2)
  const lab = half(hi)
  const {width, height, alpha} = lab
  const n = width * height
  const map = new Float32Array(n)
  if (p.mbd) {
    const d = new Float32Array(n)
    mbd(lab.l, width, height, 3, d)
    mbd(lab.a, width, height, 3, d)
    mbd(lab.b, width, height, 3, d)
    normalize(d)
    for (let i = 0; i < n; i++) map[i]! += p.mbd * d[i]!
  }
  if (p.border) {
    const b = normalize(borderContrast(lab, Math.round(Math.max(width, height) / 20)))
    for (let i = 0; i < n; i++) map[i]! += p.border * b[i]!
  }
  if (p.contrast) {
    const c = normalize(globalContrast(lab))
    for (let i = 0; i < n; i++) map[i]! += p.contrast * c[i]!
  }
  if (p.sharpness) {
    const s = sharpness(hi, width, height)
    boxBlur(s, width, height, Math.max(1, Math.round(0.03 * Math.max(width, height))))
    normalize(s)
    for (let i = 0; i < n; i++) map[i]! += p.sharpness * s[i]!
  }
  boxBlur(map, width, height, Math.round(p.blur * Math.max(width, height)))
  normalize(map)
  if (p.center) {
    const cx = (width - 1) / 2
    const cy = (height - 1) / 2
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const dx = (x - cx) / width
        const dy = (y - cy) / height
        // 1 at the centre, 1 - center at the corners
        const c = 1 - p.center * Math.sqrt((dx * dx + dy * dy) * 2)
        map[y * width + x]! *= c
      }
  }
  if (alpha) for (let i = 0; i < n; i++) map[i]! *= alpha[i]!
  return {width, height, data: normalize(map)}
}

/** Threshold that best separates the values into two classes. */
function otsu(data: Float32Array): number {
  const bins = 64
  const hist = new Float64Array(bins)
  for (const v of data) hist[Math.min(bins - 1, Math.floor(v * bins))]!++
  const total = data.length
  let sumAll = 0
  for (let i = 0; i < bins; i++) sumAll += i * hist[i]!
  let sumB = 0
  let wB = 0
  let best = 0
  let threshold = 0.5
  for (let i = 0; i < bins; i++) {
    wB += hist[i]!
    if (!wB) continue
    const wF = total - wB
    if (!wF) break
    sumB += i * hist[i]!
    const mB = sumB / wB
    const mF = (sumAll - sumB) / wF
    const between = wB * wF * (mB - mF) * (mB - mF)
    if (between > best) {
      best = between
      threshold = (i + 1) / bins
    }
  }
  return threshold
}

/** Find the main subject in an image and return its focus point. */
export function focusPoint(image: ImageDataLike, options: Options = {}): FocusPoint {
  const p = {...defaults, ...options}
  const {width, height, data} = saliency(image, p)
  const n = width * height
  const t = otsu(data)

  // Label connected regions above the threshold and score them by mass
  const label = new Int32Array(n).fill(-1)
  const stack = new Int32Array(n)
  const scores: Array<number> = []
  for (let s = 0; s < n; s++) {
    if (label[s] !== -1 || data[s]! < t) continue
    const id = scores.length
    let score = 0
    let top = 0
    label[s] = id
    stack[top++] = s
    while (top) {
      const i = stack[--top]!
      score += data[i]!
      const x = i % width
      const neighbours = [
        x > 0 ? i - 1 : -1,
        x < width - 1 ? i + 1 : -1,
        i - width,
        i + width
      ]
      for (const j of neighbours) {
        if (j < 0 || j >= n || label[j] !== -1 || data[j]! < t) continue
        label[j] = id
        stack[top++] = j
      }
    }
    scores.push(score)
  }

  let main = 0
  for (let i = 1; i < scores.length; i++) if (scores[i]! > scores[main]!) main = i

  let sx = 0, sy = 0, sw = 0
  let x0 = width, y0 = height, x1 = 0, y1 = 0
  let inside = 0, inCount = 0, outside = 0
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      const v = data[i]!
      const id = label[i]!
      if (id === main || (id >= 0 && scores[id]! >= p.merge * scores[main]!)) {
        const w = Math.pow(v, p.gamma)
        sx += w * (x + 0.5)
        sy += w * (y + 0.5)
        sw += w
        if (x < x0) x0 = x
        if (y < y0) y0 = y
        if (x > x1) x1 = x
        if (y > y1) y1 = y
        inside += v
        inCount++
      } else {
        outside += v
      }
    }
  if (!sw) {
    return {x: 0.5, y: 0.5, box: {x: 0, y: 0, width: 1, height: 1}, confidence: 0}
  }
  const meanIn = inside / inCount
  const meanOut = n > inCount ? outside / (n - inCount) : 0
  return {
    x: sx / sw / width,
    y: sy / sw / height,
    box: {
      x: x0 / width,
      y: y0 / height,
      width: (x1 - x0 + 1) / width,
      height: (y1 - y0 + 1) / height
    },
    confidence: Math.max(0, Math.min(1, meanIn - meanOut))
  }
}
