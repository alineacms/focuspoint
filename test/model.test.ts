import {expect, test} from 'bun:test'
import {readFileSync} from 'node:fs'
import {infer} from '../src/net.ts'

const fixture = JSON.parse(readFileSync(new URL('./fixtures/model.json', import.meta.url), 'utf8')) as {
  size: number
  inputs: Array<string>
  logits: Array<Array<number>>
}

test('matches PyTorch', () => {
  for (const [i, input] of fixture.inputs.entries()) {
    const bytes = Buffer.from(input, 'base64')
    const logits = infer(Float32Array.from(bytes, v => v / 255), fixture.size)
    const expected = fixture.logits[i]!
    expect(logits.length).toBe(expected.length)
    let worst = 0
    for (let j = 0; j < logits.length; j++) worst = Math.max(worst, Math.abs(logits[j]! - expected[j]!))
    expect(worst).toBeLessThan(1e-3)
  }
})
