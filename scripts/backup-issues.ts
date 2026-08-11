import fs from 'node:fs/promises'
import path from 'node:path'

import { loadConfig } from './lib/config'
import { bundleIdentity, renderBundleArticle, selectBackupIssues } from './lib/backup'
import { fetchIssues } from './lib/github'

async function writeIfChanged(filePath: string, content: string): Promise<boolean> {
  try {
    if ((await fs.readFile(filePath, 'utf8')) === content) return false
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
  }
  await fs.mkdir(path.dirname(filePath), { recursive: true })
  await fs.writeFile(filePath, content)
  return true
}

const projectRoot = process.cwd()
const config = await loadConfig(projectRoot)
const issues = selectBackupIssues(await fetchIssues(config, 'all', projectRoot))
let changed = 0

for (const issue of issues) {
  const { articleId } = bundleIdentity(issue)
  const filePath = path.join(projectRoot, config.paths.articles, articleId, 'article.md')
  if (await writeIfChanged(filePath, renderBundleArticle(issue, config))) changed += 1
}

console.log(`Generated ${issues.length} article bundle(s); ${changed} file(s) changed`)
