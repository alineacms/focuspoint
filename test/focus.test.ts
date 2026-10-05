import {describe, expect, test} from 'bun:test'
import {focusPoint, saliency} from '../src/index.ts'

/** Deterministic pseudo random numbers so tests are stable. */
function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32)
}

function scene(
  width: number,
  height: number,
  subject: {x: number; y: number; r: number; color: [number, number, number]},
  options: {alpha?: boolean} = {}
) {
  const random = rng(42)
  const data = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4
      const inside = Math.hypot(x - subject.x * width, y - subject.y * height) < subject.r * width
      // Textured green-ish background with a gradient, like grass or foliage
      const n = random() * 40
      const [r, g, b] = inside ? subject.color : [60 + n, 100 + n + (y / height) * 40, 50 + n]
      data[i] = r
      data[i + 1] = g
      data[i + 2] = b
      data[i + 3] = options.alpha ? (inside ? 255 : 0) : 255
    }
  return {data, width, height}
}

describe('focusPoint', () => {
  test('finds an off-centre subject', () => {
    const img = scene(400, 300, {x: 0.75, y: 0.3, r: 0.08, color: [220, 40, 40]})
    const p = focusPoint(img)
    expect(p.x).toBeCloseTo(0.75, 1)
    expect(p.y).toBeCloseTo(0.3, 1)
    expect(p.confidence).toBeGreaterThan(0.3)
    expect(p.box.x).toBeLessThan(0.75)
    expect(p.box.x + p.box.width).toBeGreaterThan(0.75)
  })

  test('works on portrait images', () => {
    const img = scene(300, 600, {x: 0.3, y: 0.7, r: 0.1, color: [240, 230, 80]})
    const p = focusPoint(img)
    expect(p.x).toBeCloseTo(0.3, 1)
    expect(p.y).toBeCloseTo(0.7, 1)
  })

  test('uses transparency as a subject mask', () => {
    const img = scene(200, 200, {x: 0.25, y: 0.6, r: 0.15, color: [90, 120, 70]}, {alpha: true})
    const p = focusPoint(img)
    expect(p.x).toBeCloseTo(0.25, 1)
    expect(p.y).toBeCloseTo(0.6, 1)
  })

  test('falls back to the centre for a flat image', () => {
    const img = {data: new Uint8ClampedArray(64 * 48 * 4).fill(128), width: 64, height: 48}
    const p = focusPoint(img)
    expect(p.x).toBeCloseTo(0.5, 1)
    expect(p.y).toBeCloseTo(0.5, 1)
    expect(p.confidence).toBe(0)
    const white = focusPoint({data: new Uint8ClampedArray(64 * 48 * 4).fill(255), width: 64, height: 48})
    expect(white.y).toBeCloseTo(0.5, 1)
    expect(white.confidence).toBe(0)
  })

  test('handles tiny and huge inputs', () => {
    expect(() => focusPoint({data: new Uint8ClampedArray(4).fill(255), width: 1, height: 1})).not.toThrow()
    const big = scene(4000, 3000, {x: 0.4, y: 0.45, r: 0.05, color: [30, 60, 220]})
    const t = performance.now()
    const p = focusPoint(big)
    expect(performance.now() - t).toBeLessThan(500)
    expect(p.x).toBeCloseTo(0.4, 1)
    expect(p.y).toBeCloseTo(0.45, 1)
  })

  test('rejects data that does not match the dimensions', () => {
    expect(() => focusPoint({data: new Uint8ClampedArray(10), width: 10, height: 10})).toThrow()
  })
})

describe('saliency', () => {
  test('respects the working size', () => {
    const img = scene(400, 200, {x: 0.5, y: 0.5, r: 0.1, color: [255, 0, 0]})
    const map = saliency(img, {size: 50})
    expect(map.width).toBe(50)
    expect(map.height).toBe(25)
    expect(map.data.length).toBe(50 * 25)
  })
})
