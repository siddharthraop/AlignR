# Developer documentation for AlignR: MyST Markdown, built with Sphinx to /dev/ on the site.
project = "AlignR developer docs"
author = "Siddharth Rao"
copyright = "2026, Siddharth Rao"
release = "0.1.0"
extensions = ["myst_parser"]
myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "AlignR developer docs"
