/**
 * Container shapes a stored focus point is likely to be used for. Images are
 * filled into them with `object-fit: cover`, which crops one axis only.
 */
const aspects = [3, 16 / 9, 1, 4 / 5, 9 / 16]

/**
 * Choose the point that keeps the most importance in view across typical
 * crops. Frontends apply a point in one of two ways, and both are covered:
 * CSS `object-position` aligns the point proportionally (a point at 10%
 * pushes the window to the edge), while crop tools centre the window on
 * the point and clamp it. Because each crop cuts one axis only, x and y are
 * searched independently on the row and column profiles of the map.
 */
export function place(map: Float32Array, width: number, height: number): {x: number; y: number} {
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
    x: search(cols, width, aspects.filter(a => a < aspect).map(a => a / aspect)),
    y: search(rows, height, aspects.filter(a => a > aspect).map(a => aspect / a))
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

/** Best position along one axis for windows covering `widths` of it. */
function search(prefix: Float64Array, size: number, widths: Array<number>): number {
  if (!widths.length || !prefix[size]) return 0.5
  let best = -1
  let at = 0.5
  for (let k = 0; k <= 100; k++) {
    const t = k / 100
    let s = 0
    for (const w of widths) {
      // CSS object-position: proportional alignment
      s += span(prefix, size, t * (1 - w), t * (1 - w) + w)
      // Centred on the point, clamped to the image
      const c = Math.min(1 - w, Math.max(0, t - w / 2))
      s += span(prefix, size, c, c + w)
    }
    // Among equally good positions prefer the one nearest the centre
    if (s > best * (1 + 1e-6) || (s >= best * (1 - 1e-6) && Math.abs(t - 0.5) < Math.abs(at - 0.5))) {
      if (s > best) best = s
      at = t
    }
  }
  return at
}
