import type { BlogConfig, GitHubIssue } from '../scripts/lib/types'

export const config: BlogConfig = {
  repository: { owner: 'jamez-bondos', name: 'blog', defaultBranch: 'main' },
  paths: {
    articles: 'articles',
    code: 'code',
    rss: 'rss.xml',
    manifest: '.issue-blog/articles.json',
    staging: '.issue-blog/staging',
  },
  feed: {
    title: 'Jamez 的技术博客',
    description: 'Jamez Bondos 的技术文章',
    language: 'zh-CN',
    maxItems: 50,
    summaryLength: 200,
  },
}

export function issue(overrides: Partial<GitHubIssue> = {}): GitHubIssue {
  return {
    number: 1,
    html_url: 'https://github.com/jamez-bondos/blog/issues/1',
    title: '大模型推理初学者指南',
    body: '正文\n\n<!-- issue-blog:article-id=001-beginners-guide-to-llm-inference -->\n',
    body_text: '正文',
    state: 'open',
    user: { login: 'jamez-bondos', html_url: 'https://github.com/jamez-bondos' },
    labels: [{ name: '2026' }, { name: 'llm' }],
    created_at: '2026-08-10T01:00:00Z',
    updated_at: '2026-08-10T02:00:00Z',
    ...overrides,
  }
}
