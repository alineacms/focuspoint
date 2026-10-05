import {expect, test} from 'bun:test'
import {peak} from '../eval/metrics.ts'

test('peak of a flat mask is its middle, not its first pixel', () => {
  const width = 100, height = 80
  const data = new Float32Array(width * height)
  for (let y = 20; y < 60; y++) for (let x = 10; x < 50; x++) data[y * width + x] = 1
  const p = peak({width, height, data})
  expect(p.x).toBeCloseTo(0.3, 1)
  expect(p.y).toBeCloseTo(0.5, 1)
})
