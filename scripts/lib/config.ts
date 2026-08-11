import fs from 'node:fs/promises'
import path from 'node:path'

import type { BlogConfig } from './types'

function requireString(value: unknown, name: string): asserts value is string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`Invalid blog config: ${name} must be a non-empty string`)
  }
}

function requirePositiveInteger(value: unknown, name: string): asserts value is number {
  if (!Number.isInteger(value) || Number(value) <= 0) {
    throw new Error(`Invalid blog config: ${name} must be a positive integer`)
  }
}

export function validateConfig(value: unknown): BlogConfig {
  const config = value as BlogConfig
  if (!config || typeof config !== 'object') throw new Error('Invalid blog config')

  requireString(config.repository?.owner, 'repository.owner')
  requireString(config.repository?.name, 'repository.name')
  requireString(config.repository?.defaultBranch, 'repository.defaultBranch')
  requireString(config.paths?.articles, 'paths.articles')
  requireString(config.paths?.code, 'paths.code')
  requireString(config.paths?.rss, 'paths.rss')
  requireString(config.paths?.manifest, 'paths.manifest')
  requireString(config.paths?.staging, 'paths.staging')
  requireString(config.feed?.title, 'feed.title')
  requireString(config.feed?.description, 'feed.description')
  requireString(config.feed?.language, 'feed.language')
  requirePositiveInteger(config.feed?.maxItems, 'feed.maxItems')
  requirePositiveInteger(config.feed?.summaryLength, 'feed.summaryLength')

  return config
}

export async function loadConfig(projectRoot = process.cwd()): Promise<BlogConfig> {
  const configPath = path.join(projectRoot, 'blog.config.json')
  const content = await fs.readFile(configPath, 'utf8')
  return validateConfig(JSON.parse(content))
}

export function repositoryName(config: BlogConfig): string {
  return `${config.repository.owner}/${config.repository.name}`
}
