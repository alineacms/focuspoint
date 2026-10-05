/**
 * Container shapes a stored focus point is likely to be used for. Images are
 * filled into them with `object-fit: cover`, which crops one axis only.
 */
const aspects = [3, 16 / 9, 1, 4 / 5, 9 / 16]

/** A region that every crop should keep whole, in 0..1 image coordinates. */
export interface Region {
  x0: number
  y0: number
  x1: number
  y1: number
}

/**
 * Choose the point that keeps the most importance in view across typical
 * crops, where each crop window is centred on the point and clamped to the
 * image. Because each crop cuts one axis only, x and y are searched
 * independently on the row and column profiles of the map. With a `keep`
 * region, only points whose crops all contain it count; when no point can
 * manage that, the one that cuts least of it wins.
 */
export function place(
  map: Float32Array,
  width: number,
  height: number,
  prefer: {x: number; y: number},
  tolerance: number,
  keep?: Region
): {x: number; y: number} {
  const cols = new Float64Array(width + 1)
  const rows = new Float64Array(height + 1)
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const v = map[y * width + x]!
      cols[x + 1]! += v
      rows[y + 1]! += v
    }
  for (let i = 0; i < width; i++) cols[i + 1]! += cols[i]!
  for (let i = 0; i < height; i++) rows[i + 1]! += rows[i]!
  const aspect = width / height
  return {
    x: search(cols, width, aspects.filter(a => a < aspect).map(a => a / aspect), prefer.x, tolerance, keep && [keep.x0, keep.x1]),
    y: search(rows, height, aspects.filter(a => a > aspect).map(a => aspect / a), prefer.y, tolerance, keep && [keep.y0, keep.y1])
  }
}

/** Sum of a prefix-summed profile over a fractional interval. */
function span(prefix: Float64Array, size: number, from: number, to: number): number {
  const at = (t: number) => {
    const f = Math.max(0, Math.min(size, t * size))
    const i = Math.floor(f)
    const v = prefix[i]!
    return i < size ? v + (prefix[i + 1]! - v) * (f - i) : v
  }
  return at(to) - at(from)
}

/**
 * Position along one axis for windows covering `widths` of it. Positions
 * that keep the most of the `keep` span in the worst window come first.
 * Of those, positions within `tolerance` (relative) of the best score count
 * as equally good, and the one nearest `prefer` wins: the point only leaves
 * the subject when that keeps noticeably more in view.
 */
function search(
  prefix: Float64Array,
  size: number,
  widths: Array<number>,
  prefer: number,
  tolerance: number,
  keep?: [number, number]
): number {
  if (!widths.length || !prefix[size]) return prefer
  const scores = new Float64Array(101)
  const kept = new Float64Array(101)
  let most = 0
  for (let k = 0; k <= 100; k++) {
    const t = k / 100
    let s = 0
    let worst = 1
    for (const w of widths) {
      const c = Math.min(1 - w, Math.max(0, t - w / 2))
      s += span(prefix, size, c, c + w)
      if (keep) worst = Math.min(worst, Math.max(0, Math.min(c + w, keep[1]) - Math.max(c, keep[0])) / Math.max(1e-6, keep[1] - keep[0]))
    }
    scores[k] = s
    kept[k] = worst
    if (worst > most) most = worst
  }
  let best = 0
  for (let k = 0; k <= 100; k++) if (kept[k]! >= most - 1e-6 && scores[k]! > best) best = scores[k]!
  let at = prefer
  let distance = Infinity
  for (let k = 0; k <= 100; k++) {
    if (kept[k]! < most - 1e-6 || scores[k]! < best * (1 - tolerance)) continue
    const d = Math.abs(k / 100 - prefer)
    if (d < distance) {
      distance = d
      at = k / 100
    }
  }
  return at
}
