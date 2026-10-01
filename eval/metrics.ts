/** A ground-truth subject mask, 0..1 per pixel. */
export interface Mask {
  width: number
  height: number
  data: Float32Array
}

export interface Point {
  x: number
  y: number
}

/** Crops applied to every image when measuring how much subject survives. */
export const crops = {
  /** Square avatar / grid tile */
  square: {aspect: 1, zoom: 1},
  /** Phone story / portrait card */
  portrait: {aspect: 9 / 16, zoom: 1},
  /** Wide banner / hero */
  banner: {aspect: 3, zoom: 1},
  /** Same aspect, zoomed in 2x */
  zoom: {aspect: 0, zoom: 2}
} as const

export type CropName = keyof typeof crops

export function centroid(mask: Mask): Point {
  let sx = 0, sy = 0, s = 0
  for (let y = 0; y < mask.height; y++)
    for (let x = 0; x < mask.width; x++) {
      const v = mask.data[y * mask.width + x]!
      sx += v * (x + 0.5)
      sy += v * (y + 0.5)
      s += v
    }
  return s ? {x: sx / s / mask.width, y: sy / s / mask.height} : {x: 0.5, y: 0.5}
}

/** Largest crop of the given aspect centred on the point, clamped to the image. */
export function cropAround(
  p: Point,
  width: number,
  height: number,
  aspect: number,
  zoom: number
) {
  const a = aspect || width / height
  let w = width
  let h = width / a
  if (h > height) {
    h = height
    w = height * a
  }
  w /= zoom
  h /= zoom
  const x = Math.min(width - w, Math.max(0, p.x * width - w / 2))
  const y = Math.min(height - h, Math.max(0, p.y * height - h / 2))
  return {x, y, width: w, height: h}
}

/** Fraction of the mask that ends up inside the crop. */
export function retained(
  mask: Mask,
  crop: {x: number; y: number; width: number; height: number}
): number {
  let total = 0, inside = 0
  const x0 = Math.round(crop.x), x1 = Math.round(crop.x + crop.width)
  const y0 = Math.round(crop.y), y1 = Math.round(crop.y + crop.height)
  for (let y = 0; y < mask.height; y++)
    for (let x = 0; x < mask.width; x++) {
      const v = mask.data[y * mask.width + x]!
      total += v
      if (x >= x0 && x < x1 && y >= y0 && y < y1) inside += v
    }
  return total ? inside / total : 1
}

export interface Score {
  /** Point lies on the subject. */
  hit: number
  /** Distance to the subject centroid in normalised image coordinates. */
  dist: number
  crops: Record<CropName, number>
}

export function score(mask: Mask, p: Point, cropFor?: (name: CropName) => Point): Score {
  const c = centroid(mask)
  const px = Math.min(mask.width - 1, Math.floor(p.x * mask.width))
  const py = Math.min(mask.height - 1, Math.floor(p.y * mask.height))
  const out = {} as Record<CropName, number>
  for (const name of Object.keys(crops) as Array<CropName>) {
    const {aspect, zoom} = crops[name]
    const at = cropFor ? cropFor(name) : p
    out[name] = retained(mask, cropAround(at, mask.width, mask.height, aspect, zoom))
  }
  return {
    hit: mask.data[py * mask.width + px]! > 0.5 ? 1 : 0,
    dist: Math.hypot(p.x - c.x, p.y - c.y),
    crops: out
  }
}

export function mean(scores: Array<Score>) {
  const n = scores.length
  const crop = {} as Record<CropName, number>
  for (const name of Object.keys(crops) as Array<CropName>)
    crop[name] = scores.reduce((s, x) => s + x.crops[name], 0) / n
  return {
    n,
    hit: scores.reduce((s, x) => s + x.hit, 0) / n,
    dist: scores.reduce((s, x) => s + x.dist, 0) / n,
    crops: crop
  }
}
