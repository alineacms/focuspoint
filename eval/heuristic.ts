// The hand-written heuristic that came before the learned model: minimum
// barrier distance, skin tone and a centre prior. Kept as a baseline.
import type {ImageDataLike} from '../src/image.ts'
import {defaults as base, locate, type FocusPoint, type Params as Placement, type SaliencyMap} from '../src/index.ts'
import {borderContrast, boxBlur, mbd, normalize, skin, toLab} from './signals.ts'

export interface Params extends Placement {
  /** Weight of the minimum barrier distance map. */
  mbd: number
  /** Weight of the border colour contrast map. */
  border: number
  /** Weight of the skin tone map. */
  skin: number
  /** Strength of the centre prior, 0 disables it. */
  center: number
  /** Blur radius applied to the combined map, relative to `size`. */
  blur: number
}

export const defaults: Params = {...base, mbd: 2, border: 0, skin: 0.5, center: 1, blur: 0, protect: 0}

/** Compute the heuristic's importance map. */
export function saliency(image: ImageDataLike, options: Partial<Params> = {}): SaliencyMap {
  const p = {...defaults, ...options}
  const lab = toLab(image, p.size)
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
  if (p.skin) {
    const k = skin(lab)
    for (let i = 0; i < n; i++) map[i]! += p.skin * k[i]!
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

export function focusPoint(image: ImageDataLike, options: Partial<Params> = {}): FocusPoint {
  const p = {...defaults, ...options}
  return locate(saliency(image, p), p)
}
