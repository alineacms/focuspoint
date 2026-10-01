/**
 * Container shapes a stored focus point is likely to be used for. Images are
 * filled into them with `object-fit: cover`, which crops one axis only.
 */
const aspects = [3, 16 / 9, 1, 4 / 5, 9 / 16]

/**
 * Choose the point that keeps the most importance in view across typical
 * crops, where each crop window is centred on the point and clamped to the
 * image. Because each crop cuts one axis only, x and y are searched
 * independently on the row and column profiles of the map.
 */
export function place(
  map: Float32Array,
  width: number,
  height: number,
  prefer: {x: number; y: number},
  tolerance: number
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
    x: search(cols, width, aspects.filter(a => a < aspect).map(a => a / aspect), prefer.x, tolerance),
    y: search(rows, height, aspects.filter(a => a > aspect).map(a => aspect / a), prefer.y, tolerance)
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
 * within `tolerance` (relative) of the best score count as equally good, and
 * of those the one nearest `prefer` wins: the point only leaves the subject
 * when that keeps noticeably more in view.
 */
function search(
  prefix: Float64Array,
  size: number,
  widths: Array<number>,
  prefer: number,
  tolerance: number
): number {
  if (!widths.length || !prefix[size]) return prefer
  const scores = new Float64Array(101)
  let best = 0
  for (let k = 0; k <= 100; k++) {
    const t = k / 100
    let s = 0
    for (const w of widths) {
      const c = Math.min(1 - w, Math.max(0, t - w / 2))
      s += span(prefix, size, c, c + w)
    }
    scores[k] = s
    if (s > best) best = s
  }
  let at = prefer
  let distance = Infinity
  for (let k = 0; k <= 100; k++) {
    if (scores[k]! < best * (1 - tolerance)) continue
    const d = Math.abs(k / 100 - prefer)
    if (d < distance) {
      distance = d
      at = k / 100
    }
  }
  return at
}
