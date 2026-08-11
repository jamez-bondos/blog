import { describe, expect, test } from 'bun:test'
import matter from 'gray-matter'

import { renderBundleArticle, selectBackupIssues } from '../scripts/lib/backup'
import { config, issue } from './fixtures'

describe('Issue article bundles', () => {
  test('keeps open and closed marked articles while excluding PRs and unmarked issues', () => {
    const selected = selectBackupIssues(
      [
        issue(),
        issue({ number: 2, state: 'closed' }),
        issue({ number: 3, pull_request: {} }),
        issue({ number: 4, body: '普通 Issue，没有文章标记' }),
      ],
    )

    expect(selected.map(item => item.number)).toEqual([1, 2])
  })

  test('requires the hidden article marker when deriving a bundle identity', () => {
    expect(() => renderBundleArticle(issue({ body: '没有文章标记' }), config)).toThrow(
      'does not contain an issue-blog article marker',
    )
  })

  test('stores metadata and renders a standalone article with local resource links', () => {
    const body = [
      '第一行',
      '',
      '![图](../blob/main/articles/001-beginners-guide-to-llm-inference/assets/demo.png?raw=true)',
      '',
      '[配套代码](../tree/main/code/001)',
      '',
      '<!-- issue-blog:article-id=001-beginners-guide-to-llm-inference -->',
      '',
    ].join('\n')
    const parsed = matter(renderBundleArticle(issue({ body }), config))

    expect(parsed.data.article_id).toBe('001-beginners-guide-to-llm-inference')
    expect(parsed.data.issue).toBe(1)
    expect(parsed.data.state).toBe('open')
    expect(parsed.content).toStartWith('# 大模型推理初学者指南\n\n第一行')
    expect(parsed.content).toContain('![图](./assets/demo.png)')
    expect(parsed.content).toContain('[配套代码](../../code/001/)')
    expect(parsed.content).toContain(
      '<!-- issue-blog:article-id=001-beginners-guide-to-llm-inference -->',
    )
    expect(parsed.content).not.toContain('../blob/main/')
    expect(parsed.content).not.toContain('../tree/main/')
  })
})
