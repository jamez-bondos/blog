import fs from 'node:fs/promises'
import path from 'node:path'

import type { ArticleManifest, BlogConfig } from './types'

const ARTICLE_ID_PATTERN = /^\d{3}(?:-[a-z0-9]+)+$/
const ARTICLE_MARKER_PATTERN = /<!--\s*issue-blog:article-id=([a-z0-9-]+)\s*-->/g
const MAX_ISSUE_BODY_LENGTH = 65_536

export interface PreparedArticle {
  articleId: string
  title: string
  body: string
  imageFiles: string[]
  codeFiles: string[]
  codeLinkReplaced: boolean
  tableOfContentsRemoved: boolean
  displayMathBlocksConverted: number
  mathMacrosNormalized: number
}

export interface PrepareArticleInput {
  articleId: string
  markdown: string
  assetNames: string[]
  codeNames: string[]
  config: BlogConfig
}

export function articleNumber(articleId: string): string {
  const match = /^(\d{3})(?:-|$)/.exec(articleId)
  if (!match) throw new Error(`Article ID must start with a three-digit number: ${articleId}`)
  return match[1]
}

function assertSafeBasename(name: string, kind: string): void {
  if (path.basename(name) !== name || name.includes('..')) {
    throw new Error(`Unsafe ${kind} filename: ${name}`)
  }
}

function removeTableOfContents(markdown: string): {
  markdown: string
  removed: boolean
} {
  const lines = markdown.split('\n')
  const tocIndex = lines.findIndex(line => {
    const trimmed = line.trim()
    return /^(?:\*\*|__)?目录(?:\*\*|__)?[：:]?$/.test(trimmed)
      || /^#{1,6}\s+目录[：:]?$/.test(trimmed)
  })
  if (tocIndex < 0) return { markdown, removed: false }

  const headingMatch = /^(#{1,6})\s+/.exec(lines[tocIndex].trim())
  const boundaryLevel = headingMatch ? headingMatch[1].length : 2
  let endIndex = tocIndex + 1
  while (endIndex < lines.length) {
    const nextHeading = /^(#{1,6})\s+/.exec(lines[endIndex].trim())
    if (nextHeading && nextHeading[1].length <= boundaryLevel) break
    endIndex += 1
  }

  const candidate = lines.slice(tocIndex + 1, endIndex).join('\n')
  if (!/^\s*[-+*]\s+\[[^\]]+\]\(#[^)]+\)/m.test(candidate)) {
    return { markdown, removed: false }
  }

  const remaining = [...lines.slice(0, tocIndex), ...lines.slice(endIndex)]
  return {
    markdown: remaining.join('\n').replace(/\n{3,}/g, '\n\n'),
    removed: true,
  }
}

function convertDisplayMath(markdown: string): {
  markdown: string
  blocksConverted: number
  macrosNormalized: number
} {
  const lines = markdown.split('\n')
  let inMathBlock = false
  let blocksConverted = 0
  let macrosNormalized = 0

  const converted = lines.map(line => {
    if (line.trim() !== '$$') {
      if (!inMathBlock) return line
      return line.replace(/\\operatorname\{([^{}]+)\}/g, (_match, name: string) => {
        macrosNormalized += 1
        return `\\mathrm{${name}}`
      })
    }
    if (!inMathBlock) {
      inMathBlock = true
      blocksConverted += 1
      return '```math'
    }
    inMathBlock = false
    return '```'
  })

  if (inMathBlock) throw new Error('Unclosed display math block')
  return { markdown: converted.join('\n'), blocksConverted, macrosNormalized }
}

export function prepareArticleContent(input: PrepareArticleInput): PreparedArticle {
  const { articleId, markdown, config } = input
  if (!ARTICLE_ID_PATTERN.test(articleId)) {
    throw new Error(`Invalid article ID: ${articleId}`)
  }
  const codeDirectory = articleNumber(articleId)

  const lines = markdown.split(/\r?\n/)
  const firstContentIndex = lines.findIndex(line => line.trim() !== '')
  if (firstContentIndex < 0 || !lines[firstContentIndex].startsWith('# ')) {
    throw new Error('The first non-empty Markdown line must be an H1 title')
  }

  const title = lines[firstContentIndex].slice(2).trim()
  if (!title) throw new Error('Article title cannot be empty')

  lines.splice(firstContentIndex, 1)
  if (lines[firstContentIndex]?.trim() === '') lines.splice(firstContentIndex, 1)
  let body = lines.join('\n')

  const toc = removeTableOfContents(body)
  body = toc.markdown
  const math = convertDisplayMath(body)
  body = math.markdown

  const assets = new Set(input.assetNames)
  const referencedImages: string[] = []
  body = body.replace(/!\[([^\]]*)\]\(assets\/([^\s)]+)\)/g, (_match, alt: string, name: string) => {
    assertSafeBasename(name, 'image')
    if (!name.toLowerCase().endsWith('.png')) {
      throw new Error(`Only PNG article images are supported: ${name}`)
    }
    if (!assets.has(name)) throw new Error(`Referenced article image is missing: ${name}`)
    referencedImages.push(name)
    return `![${alt}](../blob/${config.repository.defaultBranch}/${config.paths.articles}/${articleId}/assets/${name}?raw=true)`
  })

  const codeFiles = input.codeNames.filter(name => name.toLowerCase().endsWith('.py')).sort()
  for (const name of codeFiles) assertSafeBasename(name, 'code')

  let codeLinkReplaced = false
  body = body.replace(/\[[^\]]+\]\(https:\/\/gist\.github\.com\/[^)]+\)/gi, () => {
    codeLinkReplaced = true
    return `[仓库中的配套代码](../tree/${config.repository.defaultBranch}/${config.paths.code}/${codeDirectory})`
  })

  body = body.replace(ARTICLE_MARKER_PATTERN, '').trimEnd()
  body = `${body}\n\n<!-- issue-blog:article-id=${articleId} -->\n`

  if ([...body].length > MAX_ISSUE_BODY_LENGTH) {
    throw new Error(`Prepared Issue body exceeds ${MAX_ISSUE_BODY_LENGTH} Unicode code points`)
  }

  return {
    articleId,
    title,
    body,
    imageFiles: [...new Set(referencedImages)],
    codeFiles,
    codeLinkReplaced,
    tableOfContentsRemoved: toc.removed,
    displayMathBlocksConverted: math.blocksConverted,
    mathMacrosNormalized: math.macrosNormalized,
  }
}

async function listFiles(directory: string, extension: string): Promise<string[]> {
  const entries = await fs.readdir(directory, { withFileTypes: true })
  return entries
    .filter(entry => entry.isFile() && entry.name.toLowerCase().endsWith(extension))
    .map(entry => entry.name)
    .sort()
}

export async function prepareArticleFromFiles(options: {
  projectRoot: string
  articleId: string
  sourcePath: string
  assetsDirectory: string
  codeDirectory: string
  config: BlogConfig
  dryRun: boolean
}): Promise<PreparedArticle> {
  const markdown = await fs.readFile(options.sourcePath, 'utf8')
  const assetNames = await listFiles(options.assetsDirectory, '.png')
  const codeNames = await listFiles(options.codeDirectory, '.py')
  const prepared = prepareArticleContent({
    articleId: options.articleId,
    markdown,
    assetNames,
    codeNames,
    config: options.config,
  })

  if (options.dryRun) return prepared

  const articleRoot = path.join(
    options.projectRoot,
    options.config.paths.articles,
    options.articleId,
  )
  const assetsTarget = path.join(articleRoot, 'assets')
  const codeTarget = path.join(
    options.projectRoot,
    options.config.paths.code,
    articleNumber(options.articleId),
  )
  await fs.mkdir(assetsTarget, { recursive: true })
  await fs.mkdir(codeTarget, { recursive: true })

  for (const name of prepared.imageFiles) {
    await fs.copyFile(path.join(options.assetsDirectory, name), path.join(assetsTarget, name))
  }
  for (const name of prepared.codeFiles) {
    await fs.copyFile(path.join(options.codeDirectory, name), path.join(codeTarget, name))
  }

  const stagingRoot = path.join(options.projectRoot, options.config.paths.staging)
  await fs.mkdir(stagingRoot, { recursive: true })
  await fs.writeFile(path.join(stagingRoot, `${options.articleId}.md`), prepared.body)
  await fs.writeFile(
    path.join(stagingRoot, `${options.articleId}.json`),
    `${JSON.stringify({
      articleId: prepared.articleId,
      title: prepared.title,
      imageFiles: prepared.imageFiles,
      codeFiles: prepared.codeFiles,
      codeLinkReplaced: prepared.codeLinkReplaced,
      tableOfContentsRemoved: prepared.tableOfContentsRemoved,
      displayMathBlocksConverted: prepared.displayMathBlocksConverted,
      mathMacrosNormalized: prepared.mathMacrosNormalized,
    }, null, 2)}\n`,
  )

  return prepared
}

export function emptyManifest(): ArticleManifest {
  return { version: 1, articles: {} }
}

export function upsertManifest(
  manifest: ArticleManifest,
  articleId: string,
  issue: number,
  url: string,
): ArticleManifest {
  if (!ARTICLE_ID_PATTERN.test(articleId)) throw new Error(`Invalid article ID: ${articleId}`)
  if (!Number.isInteger(issue) || issue <= 0) throw new Error('Issue number must be positive')
  if (!/^https:\/\/github\.com\//.test(url)) throw new Error('Issue URL must be a GitHub URL')
  return {
    version: 1,
    articles: {
      ...manifest.articles,
      [articleId]: { issue, url },
    },
  }
}

export async function readManifest(manifestPath: string): Promise<ArticleManifest> {
  try {
    const parsed = JSON.parse(await fs.readFile(manifestPath, 'utf8')) as ArticleManifest
    if (parsed.version !== 1 || !parsed.articles || typeof parsed.articles !== 'object') {
      throw new Error('Invalid article manifest')
    }
    return parsed
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code
    if (code === 'ENOENT') return emptyManifest()
    throw error
  }
}

export async function writeManifest(manifestPath: string, manifest: ArticleManifest): Promise<void> {
  await fs.mkdir(path.dirname(manifestPath), { recursive: true })
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)
}

export function extractArticleId(body: string | null): string | undefined {
  if (!body) return undefined
  const matches = [...body.matchAll(ARTICLE_MARKER_PATTERN)]
  return matches.at(-1)?.[1]
}
