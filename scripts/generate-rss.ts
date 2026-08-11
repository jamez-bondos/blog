import fs from 'node:fs/promises'
import path from 'node:path'

import { loadConfig } from './lib/config'
import { fetchIssues } from './lib/github'
import { generateRss } from './lib/rss'

const projectRoot = process.cwd()
const config = await loadConfig(projectRoot)
const issues = await fetchIssues(config, 'open', projectRoot)
const rss = generateRss(config, issues)
const outputPath = path.join(projectRoot, config.paths.rss)

await fs.writeFile(outputPath, rss)
console.log(`Generated ${config.paths.rss}`)
