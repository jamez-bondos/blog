import path from 'node:path'

import { loadConfig } from './lib/config'
import {
  prepareArticleFromFiles,
  readManifest,
  upsertManifest,
  writeManifest,
} from './lib/article'

function parseArgs(argv: string[]): { values: Record<string, string>; flags: Set<string> } {
  const values: Record<string, string> = {}
  const flags = new Set<string>()
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (!argument.startsWith('--')) continue
    const name = argument.slice(2)
    const next = argv[index + 1]
    if (!next || next.startsWith('--')) {
      flags.add(name)
      continue
    }
    values[name] = next
    index += 1
  }
  return { values, flags }
}

function requireArg(values: Record<string, string>, name: string): string {
  const value = values[name]
  if (!value) throw new Error(`Missing required argument: --${name}`)
  return value
}

async function main(): Promise<void> {
  const projectRoot = process.cwd()
  const config = await loadConfig(projectRoot)
  const { values, flags } = parseArgs(process.argv.slice(2))
  const articleId = requireArg(values, 'id')

  if (values['register-issue'] || values['issue-url']) {
    const issue = Number.parseInt(requireArg(values, 'register-issue'), 10)
    const issueUrl = requireArg(values, 'issue-url')
    const manifestPath = path.join(projectRoot, config.paths.manifest)
    const manifest = await readManifest(manifestPath)
    await writeManifest(manifestPath, upsertManifest(manifest, articleId, issue, issueUrl))
    console.log(`Registered ${articleId} as Issue #${issue}`)
    return
  }

  const prepared = await prepareArticleFromFiles({
    projectRoot,
    articleId,
    sourcePath: path.resolve(requireArg(values, 'source')),
    assetsDirectory: path.resolve(requireArg(values, 'assets')),
    codeDirectory: path.resolve(requireArg(values, 'code')),
    config,
    dryRun: flags.has('dry-run'),
  })

  if (flags.has('dry-run')) {
    console.log(JSON.stringify(prepared, null, 2))
    return
  }

  console.log(`Prepared: ${prepared.title}`)
  console.log(`Images: ${prepared.imageFiles.length}`)
  console.log(`Code files: ${prepared.codeFiles.length}`)
  console.log(`Staging body: ${path.join(config.paths.staging, `${articleId}.md`)}`)
}

await main()
