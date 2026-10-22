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
    const detail = body?.detail || `请求失败（HTTP ${response.status}）`
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

export function getConversation(conversationNo) {
  return request(`/api/v1/conversations/${conversationNo}`)
}
