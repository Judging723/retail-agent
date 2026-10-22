async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  const body = await response.json().catch(() => null)

  if (!response.ok) {
    const detail =
      body?.detail || body?.error?.message || `请求失败（HTTP ${response.status}）`
    throw new Error(detail)
  }

  return body
}

export function createConversation(customerNo, displayName) {
  return request('/api/v1/conversations', {
    method: 'POST',
    body: JSON.stringify({
      customer_no: customerNo,
      display_name: displayName || null,
    }),
  })
}

export function sendChatMessage(conversationNo, content) {
  return request(`/api/v1/conversations/${conversationNo}/chat`, {
    method: 'POST',
    body: JSON.stringify({ content }),
  })
}

export async function* streamChatMessage(conversationNo, content) {
  const response = await fetch(
    `/api/v1/conversations/${conversationNo}/chat/stream`,
    {
      method: 'POST',
      headers: {
        Accept: 'text/event-stream',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ content }),
    },
  )

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail =
      body?.detail || body?.error?.message || `请求失败（HTTP ${response.status}）`
    throw new Error(detail)
  }

  if (!response.body) {
    throw new Error('当前浏览器不支持流式响应')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })

      let separatorIndex = buffer.indexOf('\n\n')
      while (separatorIndex >= 0) {
        const block = buffer.slice(0, separatorIndex)
        buffer = buffer.slice(separatorIndex + 2)

        if (block.trim()) {
          yield parseSseBlock(block)
        }

        separatorIndex = buffer.indexOf('\n\n')
      }

      if (done) break
    }

    if (buffer.trim()) {
      yield parseSseBlock(buffer)
    }
  } finally {
    reader.releaseLock()
  }
}

function parseSseBlock(block) {
  let event = 'message'
  const dataLines = []

  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart())
    }
  }

  const serializedData = dataLines.join('\n')
  return {
    event,
    data: serializedData ? JSON.parse(serializedData) : null,
  }
}

export function getConversation(conversationNo) {
  return request(`/api/v1/conversations/${conversationNo}`)
}
