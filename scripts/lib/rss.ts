import { Feed } from 'feed'

import type { BlogConfig, GitHubIssue } from './types'
import { isArticleIssue } from './backup'

function stripMarkdown(markdown: string): string {
  return markdown
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^>\s?/gm, '')
    .replace(/[*_`~]/g, '')
    .replace(/\$+/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

export function issueSummary(issue: GitHubIssue, length: number): string {
  const text = (issue.body_text || '').trim() || stripMarkdown(issue.body ?? '')
  const codePoints = [...text.replace(/\s+/g, ' ').trim()]
  return codePoints.length > length
    ? `${codePoints.slice(0, length).join('')}…`
    : codePoints.join('')
}

export function selectPublishedIssues(
  issues: GitHubIssue[],
  config: BlogConfig,
): GitHubIssue[] {
  return issues
    .filter(issue => isArticleIssue(issue) && issue.state === 'open')
    .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
    .slice(0, config.feed.maxItems)
}

export function generateRss(config: BlogConfig, issues: GitHubIssue[]): string {
  const posts = selectPublishedIssues(issues, config)
  const issuesUrl = `https://github.com/${config.repository.owner}/${config.repository.name}/issues`
  const updated = posts.length
    ? new Date(Math.max(...posts.map(issue => Date.parse(issue.updated_at))))
    : new Date(0)

  const feed = new Feed({
    title: config.feed.title,
    description: config.feed.description,
    id: issuesUrl,
    link: issuesUrl,
    language: config.feed.language,
    updated,
    copyright: '',
  })

  for (const issue of posts) {
    feed.addItem({
      title: issue.title,
      id: issue.html_url,
      link: issue.html_url,
      description: issueSummary(issue, config.feed.summaryLength),
      date: new Date(issue.created_at),
      author: issue.user
        ? [{ name: issue.user.login, link: issue.user.html_url }]
        : undefined,
    })
  }

  return feed.rss2()
}
