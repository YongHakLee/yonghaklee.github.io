# YongHak Lee's Tech Blog

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?style=flat-square&logo=github)](https://yonghaklee.github.io)
[![Jekyll](https://img.shields.io/badge/Jekyll-4.x-CC0000?style=flat-square&logo=jekyll)](https://jekyllrb.com/)
[![Chirpy Theme](https://img.shields.io/badge/Theme-Chirpy%207.4.1-blue?style=flat-square)](https://github.com/cotes2020/jekyll-theme-chirpy)

A personal tech blog documenting my journey of continuous learning and growth.

[Visit Blog](https://yonghaklee.github.io) • [Topics](#topics)

---

## Overview

This is a personal tech blog built with [Jekyll](https://jekyllrb.com/) and the [Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy). Hosted on GitHub Pages, it covers topics including Linux, Python, GitHub, and more.

### Features

- **Markdown-based posts** — Easy content creation and management
- **Dark mode by default** — Comfortable reading experience
- **Responsive design** — Optimized for all devices
- **Search functionality** — Quick content discovery
- **Table of contents** — Easy navigation for long articles
- **PWA support** — Offline caching and installable as an app

## Topics

| Category   | Content                                        |
| ---------- | ---------------------------------------------- |
| **Linux**  | System administration, commands, configuration |
| **Python** | tqdm, uv, syntax, algorithms                   |
| **GitHub** | Version control, collaboration workflows       |
| **CVAT**   | Image labeling, YOLO integration               |


## Writing New Posts

Create a new file in the `_posts` folder with the following format:

```
YYYY-MM-DD-title.md
```

Include front matter at the top of the file:

```yaml
---
title: "Post Title"
date: YYYY-MM-DD HH:MM:SS +0900
categories: [Category1, Category2]
tags: [tag1, tag2]
---

Your post content...
```

> [!NOTE]
> You can write drafts in the `_drafts` folder and move them to `_posts` when ready.

## Project Structure

```
├── _config.yml          # Jekyll configuration
├── _posts/              # Published posts
├── _drafts/             # Draft posts
├── _includes/           # Reusable HTML partials
├── _layouts/            # Page layouts
├── _sass/               # SCSS styles
├── _javascript/         # Client-side JavaScript
├── assets/              # Static files (images, CSS, JS)
└── tools/               # Development scripts
```

## Tech Stack

- **Static Site Generator**: [Jekyll](https://jekyllrb.com/)
- **Theme**: [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) v7.4.1
- **Frontend**: Bootstrap 5, Popper.js
- **Build Tools**: Rollup, PurgeCSS
- **Hosting**: GitHub Pages

## Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Chirpy Theme Guide](https://chirpy.cotes.page/)
- [GitHub Pages Documentation](https://docs.github.com/pages)
