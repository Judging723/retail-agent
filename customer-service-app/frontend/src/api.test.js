import assert from 'node:assert/strict'
import test from 'node:test'

import { streamChatMessage } from './api.js'

test('streamChatMessage parses SSE events across response chunks', async () => {
  const originalFetch = globalThis.fetch
  const encoder = new TextEncoder()
  const responseChunks = [
    'event: sources\ndata: {"sources":[]}\n\nevent: delta\ndata: {"content":"你',
    '好"}\n\nevent: done\ndata: {"message_id":42,"sources":[]}\n\n',
  ]

  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/v1/conversations/CONV-001/chat/stream')
    assert.equal(options.method, 'POST')
    assert.equal(options.headers['Idempotency-Key'], 'request-001')
    assert.deepEqual(JSON.parse(options.body), { content: '你好' })

    const body = new ReadableStream({
      start(controller) {
        for (const chunk of responseChunks) {
          controller.enqueue(encoder.encode(chunk))
        }
        controller.close()
      },
    })

    return new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    })
  }

  try {
    const events = []
    for await (const event of streamChatMessage(
      'CONV-001',
      '你好',
      'request-001',
    )) {
      events.push(event)
    }

    assert.deepEqual(events, [
      { event: 'sources', data: { sources: [] } },
      { event: 'delta', data: { content: '你好' } },
      { event: 'done', data: { message_id: 42, sources: [] } },
    ])
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('streamChatMessage exposes an HTTP error message', async () => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: 'Conversation not found' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' },
    })

  try {
    await assert.rejects(async () => {
      for await (const _event of streamChatMessage(
        'missing',
        '你好',
        'request-002',
      )) {
        // The HTTP error is raised before any event is produced.
      }
    }, /Conversation not found/)
  } finally {
    globalThis.fetch = originalFetch
  }
})
