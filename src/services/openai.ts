import OpenAI from 'openai'
import { randomUUID } from 'crypto'
import { createHash } from 'crypto'
import chalk from 'chalk'
import 'dotenv/config'
import { z } from 'zod'

import { addToTotalCost } from '../cost-tracker'
import type { AssistantMessage, UserMessage } from '../query'
import { getGlobalConfig, getOpenAIApiKey, getOrCreateUserID } from '../utils/config'
import { logError, SESSION_ID } from '../utils/log'
import { USER_AGENT } from '../utils/http'
import {
  createAssistantAPIErrorMessage,
  normalizeContentFromAPI,
} from '../utils/messages.js'
import { countTokens } from '../utils/tokens'
import { logEvent } from './statsig'
import { withVCR } from './vcr'
import { zodToJsonSchema } from 'zod-to-json-schema'

// Cost per million tokens for GPT-4 Turbo
const GPT4_TURBO_COST_PER_MILLION_INPUT_TOKENS = 10
const GPT4_TURBO_COST_PER_MILLION_OUTPUT_TOKENS = 30

// Cost per million tokens for GPT-3.5 Turbo
const GPT35_TURBO_COST_PER_MILLION_INPUT_TOKENS = 0.5
const GPT35_TURBO_COST_PER_MILLION_OUTPUT_TOKENS = 1.5

export const API_ERROR_MESSAGE_PREFIX = 'API Error'
export const PROMPT_TOO_LONG_ERROR_MESSAGE = 'Prompt is too long'
export const CREDIT_BALANCE_TOO_LOW_ERROR_MESSAGE = 'Credit balance is too low'
export const INVALID_API_KEY_ERROR_MESSAGE = 'Invalid API key · Please run /login'
export const NO_CONTENT_MESSAGE = '(no content)'

export const MAIN_QUERY_TEMPERATURE = 1 // to get more variation for binary feedback

const MAX_RETRIES = process.env.USER_TYPE === 'SWE_BENCH' ? 100 : 10
const BASE_DELAY_MS = 500

interface RetryOptions {
  maxRetries?: number
}

function getMetadata() {
  return {
    user_id: `${getOrCreateUserID()}_${SESSION_ID}`,
  }
}

function getRetryDelay(
  attempt: number,
  retryAfterHeader?: string | null,
): number {
  if (retryAfterHeader) {
    const seconds = parseInt(retryAfterHeader, 10)
    if (!isNaN(seconds)) {
      return seconds * 1000
    }
  }
  return Math.min(BASE_DELAY_MS * Math.pow(2, attempt - 1), 32000) // Max 32s delay
}

// Define the APIError type to match OpenAI's error structure
type APIError = {
  message?: string;
  status?: number;
  headers?: Record<string, string>;
  name: string;
};

function shouldRetry(error: APIError): boolean {
  // Check for overloaded errors first and only retry for SWE_BENCH
  if (error.message?.includes('overloaded')) {
    return process.env.USER_TYPE === 'SWE_BENCH'
  }

  // If the server explicitly says whether or not to retry, obey.
  const shouldRetryHeader = error.headers?.['x-should-retry']
  if (shouldRetryHeader === 'true') return true
  if (shouldRetryHeader === 'false') return false

  // Retry on request timeouts.
  if (error.status === 408) return true

  // Retry on rate limits.
  if (error.status === 429) return true

  // Retry internal errors.
  if (error.status && error.status >= 500) return true

  return false
}

async function withRetry<T>(
  operation: (attempt: number) => Promise<T>,
  options: RetryOptions = {},
): Promise<T> {
  const maxRetries = options.maxRetries ?? MAX_RETRIES
  let lastError: unknown

  for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
    try {
      return await operation(attempt)
    } catch (error) {
      lastError = error

      // Only retry if the error indicates we should
      if (
        attempt > maxRetries ||
        !(error instanceof Error) ||
        !('status' in error) ||
        !shouldRetry(error as APIError)
      ) {
        throw error
      }
      // Get retry-after header if available
      const apiError = error as APIError
      const retryAfter = apiError.headers?.['retry-after'] ?? null
      const delayMs = getRetryDelay(attempt, retryAfter)

      console.log(
        `  ⎿  ${chalk.red(`API ${apiError.name} (${apiError.message}) · Retrying in ${Math.round(delayMs / 1000)} seconds… (attempt ${attempt}/${maxRetries})`)}`,
      )

      logEvent('tengu_api_retry', {
        attempt: String(attempt),
        delayMs: String(delayMs),
        error: apiError.message || '',
        status: String(apiError.status || ''),
        provider: '1p',
      })

      await new Promise(resolve => setTimeout(resolve, delayMs))
    }
  }

  throw lastError
}

let openaiClients: { [key: string]: OpenAI } = {}

/**
 * Get the OpenAI client, creating it if it doesn't exist
 */
export function getOpenAIClient(type: 'large' | 'small'): OpenAI {

  if (openaiClients[type]) {
    return openaiClients[type]
  }
  const config = getGlobalConfig()
  const apiKey = type === 'large' ? config.largeModelApiKey : config.smallModelApiKey
  if (!apiKey) {
    console.error(
      chalk.red(
        'Go to /config and set your API keys',
      ),
    )
  }

  openaiClients[type] = new OpenAI({
    apiKey,
    maxRetries: 0, // Disabled auto-retry in favor of manual implementation
    timeout: parseInt(process.env.API_TIMEOUT_MS || String(60 * 1000), 10),
    dangerouslyAllowBrowser: true,
    baseURL: type === 'large' ? config.largeModelBaseURL : config.smallModelBaseURL,
  })

  return openaiClients[type]
}

/**
 * Reset the OpenAI client to null, forcing a new client to be created on next use
 */
export function resetOpenAIClient(): void {
  openaiClients = {}
}

export async function verifyApiKey(apiKey: string): Promise<boolean> {
  const openai = new OpenAI({
    apiKey,
    maxRetries: 3,
  })

  try {
    await withRetry(
      async () => {
        await openai.chat.completions.create({
          model: 'gpt-3.5-turbo',
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1,
          temperature: 0,
        })
        return true
      },
      { maxRetries: 2 }, // Use fewer retries for API key verification
    )
    return true
  } catch (error) {
    logError(error)
    // Check for authentication error
    if (
      error instanceof Error &&
      error.message.includes('Incorrect API key provided')
    ) {
      return false
    }
    throw error
  }
}

function getAssistantMessageFromError(error: unknown): AssistantMessage {
  if (error instanceof Error && error.message.includes('maximum context length')) {
    return createAssistantAPIErrorMessage(PROMPT_TOO_LONG_ERROR_MESSAGE)
  }
  if (
    error instanceof Error &&
    error.message.includes('insufficient_quota')
  ) {
    return createAssistantAPIErrorMessage(CREDIT_BALANCE_TOO_LOW_ERROR_MESSAGE)
  }
  if (
    error instanceof Error &&
    error.message.toLowerCase().includes('invalid api key')
  ) {
    return createAssistantAPIErrorMessage(INVALID_API_KEY_ERROR_MESSAGE)
  }
  if (error instanceof Error) {
    return createAssistantAPIErrorMessage(
      `${API_ERROR_MESSAGE_PREFIX}: ${error.message}`,
    )
  }
  return createAssistantAPIErrorMessage(API_ERROR_MESSAGE_PREFIX)
}

function getMaxTokensForModel(model: string): number {
  return 128000
  if (model.includes('gpt-4-turbo')) {
    return 128000
  }
  if (model.includes('gpt-4')) {
    return 8192
  }
  if (model.includes('gpt-3.5-turbo')) {
    return 16384
  }
  return 8192 // Default for unknown models
}

export function userMessageToOpenAIMessage(message: UserMessage) {
  return {
    role: 'user' as const,
    content: typeof message.message.content === 'string'
      ? message.message.content
      : message.message.content.map(c => {
          if (c.type === 'text') return c.text
          return '' // Handle other content types if needed
        }).join('\n'),
  }
}

export function assistantMessageToOpenAIMessage(message: AssistantMessage) {
  return {
    role: 'assistant' as const,
    content: typeof message.message.content === 'string'
      ? message.message.content
      : message.message.content.map(c => {
          if (c.type === 'text') return c.text
          return '' // Handle other content types if needed
        }).join('\n'),
  }
}

export async function querySonnet(
  messages: (UserMessage | AssistantMessage)[],
  systemPrompt: string[],
  maxThinkingTokens: number,
  tools: Tool[],
  signal: AbortSignal,
  options: {
    dangerouslySkipPermissions: boolean
    model: string
    prependCLISysprompt: boolean
  },
): Promise<AssistantMessage> {
  return await withVCR(messages, () =>
    queryLarge(
      messages,
      systemPrompt,
      maxThinkingTokens,
      tools,
      signal,
      options,
    ),
  )
}

async function queryLarge(
  messages: (UserMessage | AssistantMessage)[],
  systemPrompt: string[],
  maxThinkingTokens: number,
  tools: Tool[],
  signal: AbortSignal,
  options: {
    dangerouslySkipPermissions: boolean
    model: string
    prependCLISysprompt: boolean
  },
): Promise<AssistantMessage> {
  const openai = getOpenAIClient('large')

  const toolSchemas = await Promise.all(
    tools.map(async _ => ({
      type: 'function' as const,
      function: {
        name: _.name,
        description: await _.prompt({
          dangerouslySkipPermissions: options.dangerouslySkipPermissions,
        }),
        parameters: ('inputJSONSchema' in _ && _.inputJSONSchema
          ? _.inputJSONSchema
          : zodToJsonSchema(_.inputSchema)) as Record<string, unknown>,
      },
    })),
  )
  const model = getGlobalConfig().largeModelName
  const startIncludingRetries = Date.now()
  let start = Date.now()
  let attemptNumber = 0
  let response
  try {
    for(const tool of toolSchemas) {
   //   console.log(tool.function.name, tool.function.description.length)
      if(tool.function.description.length > 1024) {
//        tool.function.description = tool.function.description.slice(0, 1024)
      }
    }
    response = await withRetry(async attempt => {
      attemptNumber = attempt
      start = Date.now()
      // console.log('querying large model', await openai.models.list())
      // console.log('model', model)
      // console.log('baseURL', openai.baseURL)
      const stream = await openai.chat.completions.create({
        model,
        messages: [
          {
            role: 'system',
            content: systemPrompt.join('\n'),
          },
          ...messages.map(msg =>
            msg.type === 'user'
              ? userMessageToOpenAIMessage(msg)
              : assistantMessageToOpenAIMessage(msg),
          ),
        ],
        tools: toolSchemas,
        max_tokens: Math.max(
          maxThinkingTokens + 1,
          getMaxTokensForModel(options.model),
        ),
        temperature: MAIN_QUERY_TEMPERATURE,
        stream: true,
      })

      let content = ''
      let ttftMs: number | undefined
      const streamStartTime = Date.now()

      for await (const part of stream) {
        if (!ttftMs && part.choices[0]?.delta?.content) {
          ttftMs = Date.now() - streamStartTime
        }
        if (part.choices[0]?.delta?.content) {
          content += part.choices[0].delta.content
        }
      }

      return {
        content,
        ttftMs,
        usage: {
          prompt_tokens: 0, // Will be set later
          completion_tokens: 0, // Will be set later
        },
      }
    })
  } catch (error) {
    logError(error)
    logEvent('tengu_api_error', {
      model: options.model,
      error: error instanceof Error ? error.message : String(error),
      status:
        error instanceof OpenAI.APIError ? String(error.status) : undefined,
      messageCount: String(messages.length),
      messageTokens: String(countTokens(messages)),
      durationMs: String(Date.now() - start),
      durationMsIncludingRetries: String(Date.now() - startIncludingRetries),
      attempt: String(attemptNumber),
      provider: '1p',
    })
    return getAssistantMessageFromError(error)
  }

  const durationMs = Date.now() - start
  const durationMsIncludingRetries = Date.now() - startIncludingRetries

  // Calculate tokens and cost
  const inputTokens = countTokens([
    systemPrompt.join('\n'),
    ...messages.map(m =>
      typeof m.message.content === 'string'
        ? m.message.content
        : m.message.content.map(c => (c.type === 'text' ? c.text : '')).join('\n'),
    ),
  ])
  const outputTokens = countTokens([response.content])

  const isGPT4 = options.model.includes('gpt-4')
  const costUSD =
    (inputTokens / 1_000_000) *
      (isGPT4
        ? GPT4_TURBO_COST_PER_MILLION_INPUT_TOKENS
        : GPT35_TURBO_COST_PER_MILLION_INPUT_TOKENS) +
    (outputTokens / 1_000_000) *
      (isGPT4
        ? GPT4_TURBO_COST_PER_MILLION_OUTPUT_TOKENS
        : GPT35_TURBO_COST_PER_MILLION_OUTPUT_TOKENS)

  addToTotalCost(costUSD, durationMsIncludingRetries)

  logEvent('tengu_api_success', {
    model: options.model,
    messageCount: String(messages.length),
    messageTokens: String(countTokens(messages)),
    inputTokens: String(inputTokens),
    outputTokens: String(outputTokens),
    durationMs: String(durationMs),
    durationMsIncludingRetries: String(durationMsIncludingRetries),
    attempt: String(attemptNumber),
    ttftMs: String(response.ttftMs),
    provider: '1p',
  })

  return {
    message: {
      content: [
        {
          type: 'text',
          text: response.content,
        },
      ],
      role: 'assistant',
      usage: {
        prompt_tokens: inputTokens,
        completion_tokens: outputTokens,
      },
    },
    costUSD,
    durationMs,
    type: 'assistant',
    uuid: randomUUID(),
  }
}

// Define Tool interface to match what's used in the codebase
interface Tool {
  name: string
  description?: string
  inputSchema: z.ZodObject<any>
  inputJSONSchema?: Record<string, unknown>
  prompt: (options: { dangerouslySkipPermissions: boolean }) => Promise<string>
}

// Simple token counter for OpenAI - approximates tokens as words/4
function estimateTokens(text: string): number {
  return Math.ceil(text.split(/\s+/).length / 4) * 4;
}

export async function querySmall({
  systemPrompt = [],
  userPrompt,
  assistantPrompt,
  signal,
}: {
  systemPrompt: string[]
  userPrompt: string
  assistantPrompt?: string
  signal?: AbortSignal
}): Promise<AssistantMessage> {
  const openai = getOpenAIClient('small')

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    {
      role: 'system',
      content: systemPrompt.join('\n'),
    },
    {
      role: 'user',
      content: userPrompt,
    },
  ]

  if (assistantPrompt) {
    messages.push({
      role: 'assistant',
      content: assistantPrompt,
    })
  }

  const startIncludingRetries = Date.now()
  let start = Date.now()
  let attemptNumber = 0
  let stream
  let response
  const model = getGlobalConfig().smallModelName

  try {
    response = await withRetry(async attempt => {
      attemptNumber = attempt
      start = Date.now()
      const s = await openai.chat.completions.create({
        model,
        messages,
        temperature: 0,
        stream: true,
      })
      stream = s

      let content = ''
      let ttftMs: number | undefined
      const streamStartTime = Date.now()

      for await (const part of s) {
        if (!ttftMs && part.choices[0]?.delta?.content) {
          ttftMs = Date.now() - streamStartTime
        }
        if (part.choices[0]?.delta?.content) {
          content += part.choices[0].delta.content
        }
      }

      return {
        content,
        ttftMs,
        usage: {
          prompt_tokens: 0, // Will be set later
          completion_tokens: 0, // Will be set later
        },
      }
    })
  } catch (error) {
    logError(error)
    logEvent('tengu_api_error', {
      model,
      error: error instanceof Error ? error.message : String(error),
      status:
        error instanceof Error && 'status' in error
          ? String((error as any).status)
          : undefined,
      messageCount: String(assistantPrompt ? 2 : 1),
      durationMs: String(Date.now() - start),
      durationMsIncludingRetries: String(Date.now() - startIncludingRetries),
      attempt: String(attemptNumber),
      provider: '1p',
    })
    return getAssistantMessageFromError(error)
  }

  const durationMs = Date.now() - start
  const durationMsIncludingRetries = Date.now() - startIncludingRetries

  // Calculate tokens and cost
  const systemText = systemPrompt.join('\n')
  const inputTokens = estimateTokens(systemText) + 
                      estimateTokens(userPrompt) + 
                      (assistantPrompt ? estimateTokens(assistantPrompt) : 0)
  const outputTokens = estimateTokens(response.content)

  const costUSD =
    (inputTokens / 1_000_000) * GPT35_TURBO_COST_PER_MILLION_INPUT_TOKENS +
    (outputTokens / 1_000_000) * GPT35_TURBO_COST_PER_MILLION_OUTPUT_TOKENS

  addToTotalCost(costUSD, durationMsIncludingRetries)

  logEvent('tengu_api_success', {
    model,
    messageCount: String(assistantPrompt ? 2 : 1),
    inputTokens: String(inputTokens),
    outputTokens: String(outputTokens),
    durationMs: String(durationMs),
    durationMsIncludingRetries: String(durationMsIncludingRetries),
    attempt: String(attemptNumber),
    ttftMs: String(response.ttftMs),
    provider: '1p',
  })

  return {
    message: {
      content: [
        {
          type: 'text',
          text: response.content,
        },
      ],
      role: 'assistant',
      usage: {
        prompt_tokens: inputTokens,
        completion_tokens: outputTokens,
      },
    },
    costUSD,
    durationMs,
    type: 'assistant',
    uuid: randomUUID(),
  }
}

/**
 * Format system prompt with context
 */
export function formatSystemPromptWithContext(
  systemPrompt: string[],
  context: { [k: string]: string },
): string[] {
  return systemPrompt.map(prompt => {
    let result = prompt
    for (const [key, value] of Object.entries(context)) {
      result = result.replace(`{${key}}`, value)
    }
    return result
  })
}

/**
 * Split system prompt prefix
 */
function splitSysPromptPrefix(systemPrompt: string[]): string[] {
  return systemPrompt
}

/**
 * Add cache breakpoints to messages
 */
function addCacheBreakpoints(
  messages: (UserMessage | AssistantMessage)[],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  return messages.map(msg =>
    msg.type === 'user'
      ? userMessageToOpenAIMessage(msg)
      : assistantMessageToOpenAIMessage(msg)
  )
}

/**
 * Query OpenAI with prompt caching
 */
export async function querySonnetWithPromptCaching(
  messages: (UserMessage | AssistantMessage)[],
  systemPrompt: string[],
  maxThinkingTokens: number,
  tools: Tool[],
  signal: AbortSignal,
  options: {
    dangerouslySkipPermissions: boolean
    model: string
    prependCLISysprompt: boolean
  },
): Promise<AssistantMessage> {
  return queryLarge(
    messages,
    systemPrompt,
    maxThinkingTokens,
    tools,
    signal,
    options
  )
}

/**
 * Query OpenAI without prompt caching
 */
async function queryHaikuWithoutPromptCaching({
  systemPrompt,
  userPrompt,
  assistantPrompt,
  signal,
}: {
  systemPrompt: string[]
  userPrompt: string
  assistantPrompt?: string
  signal?: AbortSignal
}): Promise<AssistantMessage> {
  return querySmall({
    systemPrompt,
    userPrompt,
    assistantPrompt,
    signal,
  })
}

/**
 * Query OpenAI with prompt caching
 */
export async function queryHaikuWithPromptCaching({
  systemPrompt,
  userPrompt,
  assistantPrompt,
  signal,
}: {
  systemPrompt: string[]
  userPrompt: string
  assistantPrompt?: string
  signal?: AbortSignal
}): Promise<AssistantMessage> {
  return querySmall({
    systemPrompt,
    userPrompt,
    assistantPrompt,
    signal,
  })
}

/**
 * Query OpenAI (GPT-3.5 Turbo) with optional prompt caching
 */
export async function queryHaiku({
  systemPrompt = [],
  userPrompt,
  assistantPrompt,
  enablePromptCaching = false,
  signal,
}: {
  systemPrompt: string[]
  userPrompt: string
  assistantPrompt?: string
  enablePromptCaching?: boolean
  signal?: AbortSignal
}): Promise<AssistantMessage> {
  if (enablePromptCaching) {
    return queryHaikuWithPromptCaching({
      systemPrompt,
      userPrompt,
      assistantPrompt,
      signal,
    })
  } else {
    return queryHaikuWithoutPromptCaching({
      systemPrompt,
      userPrompt,
      assistantPrompt,
      signal,
    })
  }
}
