import { OpenAI } from "openai";
import { getGlobalConfig, GlobalConfig } from "../utils/config";
import { ProxyAgent, fetch, Response } from 'undici'
import { setSessionState, getSessionState } from "../utils/sessionState";
import { logEvent } from "../services/statsig";

enum ModelErrorType {
  MaxLength = '1024',
  MaxCompletionTokens = 'max_completion_tokens',
  StreamOptions = 'stream_options',
  Citations = 'citations',
  RateLimit = 'rate_limit'
}

function getModelErrorKey(baseURL: string, model: string, type: ModelErrorType): string {
  return `${baseURL}:${model}:${type}`
}

function hasModelError(baseURL: string, model: string, type: ModelErrorType): boolean {
  return !!getSessionState('modelErrors')[getModelErrorKey(baseURL, model, type)]
}

function setModelError(baseURL: string, model: string, type: ModelErrorType, error: string) {
  setSessionState('modelErrors', {
    [getModelErrorKey(baseURL, model, type)]: error
  })
}

// More flexible error detection system
type ErrorDetector = (errMsg: string) => boolean;
type ErrorFixer = (opts: OpenAI.ChatCompletionCreateParams) => Promise<void> | void;
type RateLimitHandler = (
  opts: OpenAI.ChatCompletionCreateParams, 
  response: Response, 
  type: 'large' | 'small',
  config: GlobalConfig,
  attempt: number, 
  maxAttempts: number
) => Promise<OpenAI.ChatCompletion | AsyncIterable<OpenAI.ChatCompletionChunk>>;

interface ErrorHandler {
  type: ModelErrorType;
  detect: ErrorDetector;
  fix: ErrorFixer;
}

// Specialized handler for rate limiting
const handleRateLimit: RateLimitHandler = async (opts, response, type, config, attempt, maxAttempts) => {
  const retryAfter = response?.headers.get('retry-after');
  const delay = retryAfter && !isNaN(parseInt(retryAfter)) ? parseInt(retryAfter) * 1000 : Math.pow(2, attempt) * 1000;
  logEvent('rate_limited', {
    delay,
    attempt,
    maxAttempts
  })
  await new Promise(resolve => setTimeout(resolve, delay));
  return getCompletion(type, opts, attempt + 1, maxAttempts);
};

// Standard error handlers
const ERROR_HANDLERS: ErrorHandler[] = [
  { 
    type: ModelErrorType.MaxLength, 
    detect: (errMsg) => errMsg.includes('Expected a string with maximum length 1024'),
    fix: async (opts) => {
      const toolDescriptions = {}
      for(const tool of opts.tools || []) {
        if(tool.function.description.length <= 1024) continue
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
    }
  },
  { 
    type: ModelErrorType.MaxCompletionTokens, 
    detect: (errMsg) => errMsg.includes("Use 'max_completion_tokens'"),
    fix: async (opts) => {
      opts.max_completion_tokens = opts.max_tokens
      delete opts.max_tokens
    }
  },
  { 
    type: ModelErrorType.StreamOptions, 
    detect: (errMsg) => errMsg.includes('Extra inputs are not permitted') && errMsg.includes('stream_options'),
    fix: async (opts) => {
      delete opts.stream_options
    }
  },
  { 
    type: ModelErrorType.Citations, 
    detect: (errMsg) => errMsg.includes('Extra inputs are not permitted') && errMsg.includes('citations'),
    fix: async (opts) => {
      if (!opts.messages) return;
      
      for (const message of opts.messages) {
        if (!message) continue;
        
        if (Array.isArray(message.content)) {
          for (const item of message.content) {
            // Convert to unknown first to safely access properties
            if (item && typeof item === 'object') {
              const itemObj = item as unknown as Record<string, unknown>;
              if ('citations' in itemObj) {
                delete itemObj.citations;
              }
            }
          }
        } else if (message.content && typeof message.content === 'object') {
          // Convert to unknown first to safely access properties
          const contentObj = message.content as unknown as Record<string, unknown>;
          if ('citations' in contentObj) {
            delete contentObj.citations;
          }
        }
      }
    }
  }
];

// Rate limit specific detection
function isRateLimitError(errMsg: string): boolean {
  if (!errMsg) return false;
  const lowerMsg = errMsg.toLowerCase();
  return lowerMsg.includes('rate limit') || lowerMsg.includes('too many requests');
}

async function applyModelErrorFixes(opts: OpenAI.ChatCompletionCreateParams, baseURL: string) {
  for (const handler of ERROR_HANDLERS) {
    if (hasModelError(baseURL, opts.model, handler.type)) {
      await handler.fix(opts);
      return;
    }
  }
}

async function handleApiError(
  response: Response,
  error: any,
  type: 'large' | 'small',
  opts: OpenAI.ChatCompletionCreateParams,
  config: GlobalConfig,
  attempt: number,
  maxAttempts: number
): Promise<OpenAI.ChatCompletion | AsyncIterable<OpenAI.ChatCompletionChunk>> {
  let errMsg = error.error?.message || error.message || error;
  if (errMsg) {
    if (typeof errMsg !== 'string') {
      errMsg = JSON.stringify(errMsg);
    }

    // Check for rate limiting first
    if (isRateLimitError(errMsg)) {
      logEvent('rate_limit_error', {
        error_message: errMsg
      })
      return handleRateLimit(opts, response, type, config, attempt, maxAttempts);
    }

    // Handle other errors
    const baseURL = type === 'large' ? config.largeModelBaseURL : config.smallModelBaseURL;

    for (const handler of ERROR_HANDLERS) {
      if (handler.detect(errMsg)) {
        logEvent('model_error', {
          model: opts.model,
          error: handler.type,
          error_message: errMsg
        })
        setModelError(baseURL, opts.model, handler.type, errMsg);
        return getCompletion(type, opts, attempt + 1, maxAttempts);
      }
    }
  }
  
  // If we get here, it's an unhandled error
  logEvent('unhandled_api_error', {
    model: opts.model,
    error: errMsg,
    error_message: errMsg
  })
  
  throw new Error(`API request failed: ${error.error?.message || JSON.stringify(error)}`);
}

export async function getCompletion(
  type: 'large' | 'small', 
  opts: OpenAI.ChatCompletionCreateParams,
  attempt: number = 0,
  maxAttempts: number = 5
): Promise<OpenAI.ChatCompletion | AsyncIterable<OpenAI.ChatCompletionChunk>> {
  if (attempt >= maxAttempts) {
    throw new Error('Max attempts reached')
  }
  const config = getGlobalConfig()
  const apiKey = type === 'large' ? config.largeModelApiKey : config.smallModelApiKey
  const baseURL = type === 'large' ? config.largeModelBaseURL : config.smallModelBaseURL
  const proxy = config.proxy ? new ProxyAgent(config.proxy) : undefined
  logEvent('get_completion', {
    messages: opts.messages.map(m => ({
      role: m.role,
      content: typeof m.content === 'string' ? m.content.slice(0, 100) : m.content?.map(c => ({
        type: c.type,
        text: c.text.slice(0, 100)
      }))
    }))
  })
  opts = structuredClone(opts)

  await applyModelErrorFixes(opts, baseURL)

  if (config.primaryProvider === 'custom') {
    opts.messages = opts.messages.map(msg => {
      if (msg.role === 'tool' && 
          Array.isArray(msg.content)
        ) {
        return {
          ...msg,
          content: msg.content.map(c => c.text).join('\n\n')
        };
      }
      return msg;
    });
  }

  try {
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
        try {
          const error = await response.json() as { error?: { message: string }, message?: string }
          return handleApiError(response, error, type, opts, config, attempt, maxAttempts)
        } catch (jsonError) {
          // If we can't parse the error as JSON, use the status text
          return handleApiError(
            response, 
            { error: { message: `HTTP error ${response.status}: ${response.statusText}` }}, 
            type, opts, config, attempt, maxAttempts
          )
        }
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

    if (!response.ok) {
      try {
        const error = await response.json() as { error?: { message: string }, message?: string }
        return handleApiError(response, error, type, opts, config, attempt, maxAttempts)
      } catch (jsonError) {
        // If we can't parse the error as JSON, use the status text
        return handleApiError(
          response, 
          { error: { message: `HTTP error ${response.status}: ${response.statusText}` }}, 
          type, opts, config, attempt, maxAttempts
        )
      }
    }

    return response.json() as Promise<OpenAI.ChatCompletion>
  } catch (error) {
    // Handle network errors or other exceptions
    if (attempt < maxAttempts - 1) {
      const delay = Math.pow(2, attempt) * 1000
      await new Promise(resolve => setTimeout(resolve, delay))
      return getCompletion(type, opts, attempt + 1, maxAttempts)
    }
    throw new Error(`Network error: ${error.message || 'Unknown error'}`)
  }
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
        let readResult;
        try {
          readResult = await reader.read();
        } catch (e) {
          console.error('Error reading from stream:', e);
          break;
        }
        
        const { done, value } = readResult;
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
              console.error('Error parsing JSON:', data, e)
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
              console.error('Error parsing final JSON:', data, e)
            }
          }
        }
      }
    } catch (e) {
      console.error('Unexpected error in stream processing:', e);
    } finally {
      try {
        reader.releaseLock()
      } catch (e) {
        console.error('Error releasing reader lock:', e);
      }
    }
  })()
}

export function streamCompletion(
  stream: any
): AsyncGenerator<OpenAI.ChatCompletionChunk, void, unknown> {
  return createStreamProcessor(stream)
}