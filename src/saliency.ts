import type {LabImage} from './image.ts'

/**
 * Minimum barrier distance from the image border (FastMBD, Zhang et al.
 * ICCV 2015). For every pixel this is the smallest "max - min" intensity
 * range along any path to the border, approximated with raster scans. Regions
 * separated from the border by a strong change score high; background that
 * flows into the border scores low.
 */
export function mbd(
  plane: Float32Array,
  width: number,
  height: number,
  passes: number,
  out: Float32Array
): void {
  const n = width * height
  const dist = new Float32Array(n).fill(Infinity)
  const hi = Float32Array.from(plane)
  const lo = Float32Array.from(plane)
  for (let x = 0; x < width; x++) {
    dist[x] = 0
    dist[n - 1 - x] = 0
  }
  for (let y = 0; y < height; y++) {
    dist[y * width] = 0
    dist[y * width + width - 1] = 0
  }
  for (let p = 0; p < passes; p++) {
    // Alternate forward and backward raster scans, relaxing each pixel
    // against its two already-visited neighbours
    const forward = p % 2 === 0
    const dx = forward ? -1 : 1
    const dy = forward ? -width : width
    for (let k = 1; k < height - 1; k++) {
      const y = forward ? k : height - 1 - k
      for (let m = 1; m < width - 1; m++) {
        const i = y * width + (forward ? m : width - 1 - m)
        const v = plane[i]!
        let d = dist[i]!
        for (let s = 0; s < 2; s++) {
          const j = i + (s ? dy : dx)
          const u = hi[j]! > v ? hi[j]! : v
          const l = lo[j]! < v ? lo[j]! : v
          if (u - l < d) {
            d = u - l
            dist[i] = d
            hi[i] = u
            lo[i] = l
          }
        }
      }
    }
  }
  for (let i = 0; i < n; i++) out[i]! += dist[i]!
}

/**
 * Colour distinctness from each image border (the "B" map of MB+). Each
 * border strip is modelled as a Gaussian in Lab; a pixel's score is its
 * Mahalanobis distance to it. The four maps are summed minus the largest one
 * so a subject that touches one border isn't treated as background.
 */
export function borderContrast(img: LabImage, thickness: number): Float32Array {
  const {width, height, l, a, b} = img
  const n = width * height
  const t = Math.max(1, Math.min(thickness, Math.floor(Math.min(width, height) / 3)))
  const sum = new Float32Array(n)
  const maps: Array<Float32Array> = []
  const regions = [
    [0, 0, width, t],
    [0, height - t, width, height],
    [0, 0, t, height],
    [width - t, 0, width, height]
  ] as const
  for (const [x0, y0, x1, y1] of regions) {
    let m0 = 0, m1 = 0, m2 = 0, count = 0
    for (let y = y0; y < y1; y++)
      for (let x = x0; x < x1; x++) {
        const i = y * width + x
        m0 += l[i]!
        m1 += a[i]!
        m2 += b[i]!
        count++
      }
    m0 /= count
    m1 /= count
    m2 /= count
    let c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0
    for (let y = y0; y < y1; y++)
      for (let x = x0; x < x1; x++) {
        const i = y * width + x
        const d0 = l[i]! - m0
        const d1 = a[i]! - m1
        const d2 = b[i]! - m2
        c00 += d0 * d0
        c01 += d0 * d1
        c02 += d0 * d2
        c11 += d1 * d1
        c12 += d1 * d2
        c22 += d2 * d2
      }
    // Regularise so flat borders don't produce a singular covariance
    const reg = 25
    c00 = c00 / count + reg
    c11 = c11 / count + reg
    c22 = c22 / count + reg
    c01 /= count
    c02 /= count
    c12 /= count
    // Inverse of the symmetric 3x3 matrix
    const i00 = c11 * c22 - c12 * c12
    const i01 = c02 * c12 - c01 * c22
    const i02 = c01 * c12 - c02 * c11
    const i11 = c00 * c22 - c02 * c02
    const i12 = c01 * c02 - c00 * c12
    const i22 = c00 * c11 - c01 * c01
    const det = c00 * i00 + c01 * i01 + c02 * i02
    const map = new Float32Array(n)
    let max = 0
    for (let i = 0; i < n; i++) {
      const d0 = l[i]! - m0
      const d1 = a[i]! - m1
      const d2 = b[i]! - m2
      const q =
        (d0 * (i00 * d0 + i01 * d1 + i02 * d2) +
          d1 * (i01 * d0 + i11 * d1 + i12 * d2) +
          d2 * (i02 * d0 + i12 * d1 + i22 * d2)) /
        det
      const v = Math.sqrt(Math.max(0, q))
      map[i] = v
      if (v > max) max = v
    }
    if (max > 0) for (let i = 0; i < n; i++) map[i]! /= max
    maps.push(map)
  }
  for (let i = 0; i < n; i++) {
    let s = 0
    let m = 0
    for (const map of maps) {
      const v = map[i]!
      s += v
      if (v > m) m = v
    }
    sum[i] = s - m
  }
  return sum
}


/** Scale values into 0..1 in place. */
export function normalize(map: Float32Array): Float32Array {
  let min = Infinity
  let max = -Infinity
  for (const v of map) {
    if (v < min) min = v
    if (v > max) max = v
  }
  const range = max - min
  if (range <= 1e-9) map.fill(0)
  else for (let i = 0; i < map.length; i++) map[i] = (map[i]! - min) / range
  return map
}

/** Separable box blur, applied in place. */
export function boxBlur(
  map: Float32Array,
  width: number,
  height: number,
  radius: number
): void {
  if (radius < 1) return
  const tmp = new Float32Array(map.length)
  for (let y = 0; y < height; y++) {
    const row = y * width
    for (let x = 0; x < width; x++) {
      let s = 0
      let c = 0
      for (let k = Math.max(0, x - radius); k <= Math.min(width - 1, x + radius); k++) {
        s += map[row + k]!
        c++
      }
      tmp[row + x] = s / c
    }
  }
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      let s = 0
      let c = 0
      for (let k = Math.max(0, y - radius); k <= Math.min(height - 1, y + radius); k++) {
        s += tmp[k * width + x]!
        c++
      }
      map[y * width + x] = s / c
    }
  }
}

/**
 * Skin likelihood from Lab hue and chroma. Skin tones across ethnicities
 * share a narrow orange hue band; lightness varies widely so it is only
 * loosely constrained.
 */
export function skin(img: LabImage): Float32Array {
  const {l, a, b} = img
  const out = new Float32Array(l.length)
  for (let i = 0; i < l.length; i++) {
    const L = l[i]!
    if (L < 15 || L > 95) continue
    const chroma = Math.hypot(a[i]!, b[i]!)
    if (chroma < 8 || chroma > 60) continue
    const hue = (Math.atan2(b[i]!, a[i]!) * 180) / Math.PI
    // Peak around 50 degrees, fading out by 25 degrees either side
    const h = Math.max(0, 1 - Math.abs(hue - 50) / 25)
    const c = Math.min(1, (chroma - 8) / 10) * Math.min(1, (60 - chroma) / 15)
    out[i] = h * c
  }
  return out
}
