import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

import type { BlogConfig, GitHubIssue } from './types'
import { repositoryName } from './config'

const execFileAsync = promisify(execFile)

export async function runGh(args: string[], cwd = process.cwd()): Promise<string> {
  try {
    const { stdout } = await execFileAsync('gh', args, {
      cwd,
      env: process.env,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    })
    return stdout.trim()
  } catch (error) {
    const details = error as { stderr?: string; message?: string }
    throw new Error(details.stderr?.trim() || details.message || 'gh command failed')
  }
}

export async function fetchIssues(
  config: BlogConfig,
  state: 'open' | 'closed' | 'all',
  cwd = process.cwd(),
): Promise<GitHubIssue[]> {
  const output = await runGh(
    [
      'api',
      `repos/${repositoryName(config)}/issues`,
      '--method',
      'GET',
      '-H',
      'Accept: application/vnd.github.full+json',
      '-f',
      `state=${state}`,
      '-f',
      'per_page=100',
      '--paginate',
      '--slurp',
    ],
    cwd,
  )

  const pages = JSON.parse(output || '[]') as unknown[]
  return pages.flatMap(page => (Array.isArray(page) ? page : [page])) as GitHubIssue[]
}
