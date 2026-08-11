import { describe, expect, test } from 'bun:test'
import { XMLParser } from 'fast-xml-parser'

import { generateRss, issueSummary, selectPublishedIssues } from '../scripts/lib/rss'
import { config, issue } from './fixtures'

describe('RSS generation', () => {
  test('keeps only open marked articles and excludes pull requests', () => {
    const selected = selectPublishedIssues(
      [
        issue({ labels: [] }),
        issue({ number: 2, state: 'closed' }),
        issue({ number: 3, body: '普通 Issue，没有文章标记' }),
        issue({ number: 4, pull_request: {} }),
      ],
      config,
    )

    expect(selected.map(item => item.number)).toEqual([1])
  })

  test('emits a parseable summary-only RSS item', () => {
    const longText = `${'推理'.repeat(120)} FULL_BODY_SENTINEL`
    const body = `${longText}\n\n<!-- issue-blog:article-id=001-beginners-guide-to-llm-inference -->\n`
    const rss = generateRss(config, [issue({ body, body_text: longText })])
    const parsed = new XMLParser().parse(rss)
    const item = parsed.rss.channel.item

    expect(item.title).toBe('大模型推理初学者指南')
    expect(item.link).toBe('https://github.com/jamez-bondos/blog/issues/1')
    expect(rss).not.toContain('content:encoded')
    expect(rss).not.toContain('FULL_BODY_SENTINEL')
    expect([...issueSummary(issue({ body_text: longText }), 200)]).toHaveLength(201)
  })
})
