import { OpenAI } from "openai";
import { getGlobalConfig } from "../utils/config";
import { ProxyAgent, fetch } from 'undici'
import { setSessionState, getSessionState } from "../utils/sessionState";
import { logEvent } from "../services/statsig";


export async function getCompletion(
  type: 'large' | 'small', 
  opts: OpenAI.ChatCompletionCreateParams,
  attempt: number = 0,
  maxAttempts: number = 3
): Promise<OpenAI.ChatCompletion | AsyncIterable<OpenAI.ChatCompletionChunk>> {
  if (attempt >= maxAttempts) {
    throw new Error('Max attempts reached')
  }
  const config = getGlobalConfig()
  const apiKey = type === 'large' ? config.largeModelApiKey : config.smallModelApiKey
  const baseURL = type === 'large' ? config.largeModelBaseURL : config.smallModelBaseURL
  const proxy = config.proxy ? new ProxyAgent(config.proxy) : undefined
  
  const toolDescriptions = {}
  
  opts = structuredClone(opts)

  if (getSessionState('modelErrors')[`${baseURL}:${opts.model}:1024`]) {
    for(const tool of opts.tools) {
      if(tool.function.description.length <= 1024) {
        continue
      }
      let str = ''
      let remainder = ''
      for(let line of tool.function.description.split('\n')) {
        if(str.length + line.length < 1024) {
          str += line + '\n'
        } else {
          remainder += line + '\n'
        }
      }
      logEvent('truncated_tool_description', {
        name: tool.function.name,
        original_length: String(tool.function.description.length),
        truncated_length: String(str.length),
        remainder_length: String(remainder.length),
      })
      tool.function.description = str
      toolDescriptions[tool.function.name] = remainder
    }
    if(Object.keys(toolDescriptions).length > 0) {
      let content = '<additional-tool-usage-instructions>\n\n'
      for(const [name, description] of Object.entries(toolDescriptions)) {
        content += `<${name}>\n${description}\n</${name}>\n\n`
      }
      content += '</additional-tool-usage-instructions>'

      for(let i = opts.messages.length - 1; i >= 0; i--) {
        if(opts.messages[i].role === 'system') {
          opts.messages.splice(i + 1, 0, {
            role: 'system',
            content
          })
          break
        }
      }
    }
  } else if (getSessionState('modelErrors')[`${baseURL}:${opts.model}:max_completion_tokens`]) {
    opts.max_completion_tokens = opts.max_tokens
    delete opts.max_tokens
  } else if (getSessionState('modelErrors')[`${baseURL}:${opts.model}:stream_options`]) {
    delete opts.stream_options
  } 
  // else if (getSessionState('modelErrors')[`${baseURL}:${opts.model}:system_role`]) {
  //   for(let m of opts.messages) {
  //     if(m.role === 'system') {
  //       const msg = m as OpenAI.ChatCompletionSystemMessageParam;
  //       (msg as any).role = 'developer';
  //     }
  //   }
  // }


  if (opts.stream) {
    const response = await fetch(`${baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ...opts, stream: true }),
      dispatcher: proxy,
    })
    
    if (!response.ok) {
      const error = await response.json() as { error?: { message: string }, message?: string }
      let errMsg = error.error?.message || error.message || error
      if (errMsg) {
        if(typeof errMsg !== 'string') {
          errMsg = JSON.stringify(errMsg)
        }
        if (errMsg?.indexOf('Expected a string with maximum length 1024') > -1) {
          setSessionState('modelErrors', {
            [`${baseURL}:${opts.model}:1024`]: errMsg
          })
          return getCompletion(type, opts, attempt + 1, maxAttempts)
        } else if (errMsg?.indexOf("Use 'max_completion_tokens'") > -1) {
          setSessionState('modelErrors', {
            [`${baseURL}:${opts.model}:max_completion_tokens`]: errMsg
          })
          return getCompletion(type, opts, attempt + 1, maxAttempts)
        } else if (errMsg?.indexOf('Extra inputs are not permitted') > -1 && errMsg?.indexOf('stream_options') > -1) {
          setSessionState('modelErrors', {
            [`${baseURL}:${opts.model}:stream_options`]: errMsg
          })
          return getCompletion(type, opts, attempt + 1, maxAttempts)
        } 
        // else if (errMsg?.indexOf("'messages[0].role' does not support 'system' with this model") > -1) {
        //   setSessionState('modelErrors', {
        //     [`${baseURL}:${opts.model}:system_role`]: errMsg
        //   })
        //   return getCompletion(type, opts, attempt + 1, maxAttempts)
        // }
      }
      throw new Error(`API request failed: ${error.error?.message || JSON.stringify(error)}`)
    }
    
    return createStreamProcessor(response.body as any)
  }
  
  const response = await fetch(`${baseURL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(opts),
    dispatcher: proxy,
  })
  return response.json() as Promise<OpenAI.ChatCompletion>
}

export function createStreamProcessor(
  stream: any
): AsyncGenerator<OpenAI.ChatCompletionChunk, void, unknown> {
  if (!stream) {
    throw new Error("Stream is null or undefined")
  }
  
  return (async function* () {
    const reader = stream.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }
        
        const chunk = decoder.decode(value, { stream: true })
        buffer += chunk
        
        let lineEnd = buffer.indexOf('\n')
        while (lineEnd !== -1) {
          const line = buffer.substring(0, lineEnd).trim()
          buffer = buffer.substring(lineEnd + 1)
          
          if (line === 'data: [DONE]') {
            continue
          }
          
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            if (!data) continue
            
            try {
              const parsed = JSON.parse(data) as OpenAI.ChatCompletionChunk
              yield parsed
            } catch (e) {
              console.error('Error parsing JSON:', data)
            }
          }
          
          lineEnd = buffer.indexOf('\n')
        }
      }
      
      // Process any remaining data in the buffer
      if (buffer.trim()) {
        const lines = buffer.trim().split('\n')
        for (const line of lines) {
          if (line.startsWith('data: ') && line !== 'data: [DONE]') {
            const data = line.slice(6).trim()
            if (!data) continue
            
            try {
              const parsed = JSON.parse(data) as OpenAI.ChatCompletionChunk
              yield parsed
            } catch (e) {
              console.error('Error parsing final JSON:', data)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  })()
}

export function streamCompletion(
  stream: any
): AsyncGenerator<OpenAI.ChatCompletionChunk, void, unknown> {
  return createStreamProcessor(stream)
}