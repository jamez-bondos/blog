export interface BlogConfig {
  repository: {
    owner: string
    name: string
    defaultBranch: string
  }
  paths: {
    articles: string
    code: string
    rss: string
    manifest: string
    staging: string
  }
  feed: {
    title: string
    description: string
    language: string
    maxItems: number
    summaryLength: number
  }
}

export interface GitHubLabel {
  name: string
}

export interface GitHubIssue {
  number: number
  html_url: string
  title: string
  body: string | null
  body_text?: string | null
  state: 'open' | 'closed'
  user?: {
    login: string
    html_url?: string
  } | null
  labels: Array<string | GitHubLabel>
  created_at: string
  updated_at: string
  pull_request?: unknown
}

export interface ArticleManifestRecord {
  issue: number
  url: string
}

export interface ArticleManifest {
  version: 1
  articles: Record<string, ArticleManifestRecord>
}
