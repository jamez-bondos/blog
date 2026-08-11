import { describe, expect, test } from 'bun:test'

import { emptyManifest, prepareArticleContent, upsertManifest } from '../scripts/lib/article'
import { config } from './fixtures'

describe('article preparation', () => {
  test('extracts the title, rewrites nine images and the code link, and adds one marker', () => {
    const imageNames = Array.from({ length: 9 }, (_, index) => `${String(index + 1).padStart(2, '0')}.png`)
    const source = [
      '# 大模型推理初学者指南',
      '',
      '副标题：用 PyTorch 从零实现 GPT-2 推理',
      '',
      ...imageNames.map(name => `![${name}](assets/${name})`),
      '',
      '查看[完整代码](https://gist.github.com/jamez-bondos/example)。',
    ].join('\n')

    const prepared = prepareArticleContent({
      articleId: '001-beginners-guide-to-llm-inference',
      markdown: source,
      assetNames: imageNames,
      codeNames: ['README.md', 'gpt2_naive.py', 'gpt2_kv_cache.py'],
      config,
    })

    expect(prepared.title).toBe('大模型推理初学者指南')
    expect(prepared.body.startsWith('# ')).toBe(false)
    expect(prepared.imageFiles).toHaveLength(9)
    expect(prepared.body.match(/\?raw=true\)/g)).toHaveLength(9)
    expect(prepared.body).toContain('../tree/main/code/001')
    expect(prepared.body).toContain('[仓库中的配套代码]')
    expect(prepared.body).not.toContain('gist.github.com')
    expect(prepared.body.match(/issue-blog:article-id=/g)).toHaveLength(1)
    expect(prepared.codeFiles).toEqual(['gpt2_kv_cache.py', 'gpt2_naive.py'])
    expect(source.startsWith('# 大模型推理初学者指南')).toBe(true)
  })

  test('updates one stable manifest mapping without creating duplicates', () => {
    const first = upsertManifest(
      emptyManifest(),
      '001-beginners-guide-to-llm-inference',
      1,
      'https://github.com/jamez-bondos/blog/issues/1',
    )
    const updated = upsertManifest(
      first,
      '001-beginners-guide-to-llm-inference',
      1,
      'https://github.com/jamez-bondos/blog/issues/1',
    )

    expect(Object.keys(updated.articles)).toEqual(['001-beginners-guide-to-llm-inference'])
    expect(updated.articles['001-beginners-guide-to-llm-inference'].issue).toBe(1)
  })

  test('removes a linked table of contents without removing article lists', () => {
    const source = [
      '# 测试文章',
      '',
      '导语。',
      '',
      '**目录**',
      '',
      '- [1. 第一节](#1-第一节)',
      '  - [1.1 子节](#11-子节)',
      '',
      '## 1. 第一节',
      '',
      '正文。',
      '',
      '- 正文列表项',
    ].join('\n')

    const prepared = prepareArticleContent({
      articleId: '999-test-article',
      markdown: source,
      assetNames: [],
      codeNames: [],
      config,
    })

    expect(prepared.tableOfContentsRemoved).toBe(true)
    expect(prepared.body).not.toContain('**目录**')
    expect(prepared.body).not.toContain('](#1-第一节)')
    expect(prepared.body).toContain('## 1. 第一节')
    expect(prepared.body).toContain('- 正文列表项')
  })

  test('converts display math delimiters to GitHub math fences', () => {
    const source = [
      '# 测试公式',
      '',
      '公式如下：',
      '',
      '$$',
      '\\operatorname{LayerNorm}(x)',
      '+ \\mathrm{bias}',
      '$$',
    ].join('\n')

    const prepared = prepareArticleContent({
      articleId: '998-math-article',
      markdown: source,
      assetNames: [],
      codeNames: [],
      config,
    })

    expect(prepared.displayMathBlocksConverted).toBe(1)
    expect(prepared.mathMacrosNormalized).toBe(1)
    expect(prepared.body).toContain('```math\n\\mathrm{LayerNorm}(x)\n+ \\mathrm{bias}\n```')
    expect(prepared.body).not.toContain('\\operatorname')
    expect(prepared.body).not.toMatch(/^\$\$$/m)
  })

  test('rejects an unclosed display math block', () => {
    const source = ['# 未闭合公式', '', '$$', 'x + y'].join('\n')

    expect(() => prepareArticleContent({
      articleId: '997-broken-math',
      markdown: source,
      assetNames: [],
      codeNames: [],
      config,
    })).toThrow('Unclosed display math block')
  })
})
