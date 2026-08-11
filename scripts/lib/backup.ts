import matter from 'gray-matter'

import type { BlogConfig, GitHubIssue } from './types'
import { extractArticleId } from './article'

export function labelNames(issue: GitHubIssue): string[] {
  return issue.labels.map(label => (typeof label === 'string' ? label : label.name))
}

export function isArticleIssue(issue: GitHubIssue): boolean {
  return !issue.pull_request && extractArticleId(issue.body) !== undefined
}

export function selectBackupIssues(issues: GitHubIssue[]): GitHubIssue[] {
  return issues.filter(isArticleIssue)
}

export function bundleIdentity(issue: GitHubIssue): { articleId: string } {
  const articleId = extractArticleId(issue.body)
  if (!articleId) {
    throw new Error(`Issue #${issue.number} does not contain an issue-blog article marker`)
  }
  return {
    articleId,
  }
}

export function renderBundleArticle(issue: GitHubIssue, config: BlogConfig): string {
  const { articleId } = bundleIdentity(issue)
  const articleRoot = `${config.paths.articles}/${articleId}`
  const imagePrefix = `../blob/${config.repository.defaultBranch}/${articleRoot}/assets/`
  const codeDirectory = /^(\d{3})(?:-|$)/.exec(articleId)?.[1]
  let issueBody = (issue.body ?? '')
    .replaceAll(imagePrefix, './assets/')
    .replace(/(\.\/assets\/[^\s)]+)\?raw=true(?=\))/g, '$1')
  if (codeDirectory) {
    const codePath = `../tree/${config.repository.defaultBranch}/${config.paths.code}/${codeDirectory}`
    issueBody = issueBody.replaceAll(codePath, `../../${config.paths.code}/${codeDirectory}/`)
  }
  const standaloneBody = `# ${issue.title}\n\n${issueBody}`

  return matter.stringify(standaloneBody, {
    article_id: articleId,
    title: issue.title,
    issue: issue.number,
    url: issue.html_url,
    state: issue.state,
    author: issue.user?.login ?? null,
    created_at: issue.created_at,
    updated_at: issue.updated_at,
    labels: labelNames(issue).sort(),
  })
}
