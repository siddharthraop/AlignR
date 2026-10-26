#!/usr/bin/env bash
# Build the GitHub Pages site into OUT (default _site). Used by .github/workflows/pages.yml and locally.
#
#   /          the design document
#   /app/      the AlignR page: scorecard, Draft, drafts beside the published topics
#   /help/     the DITA help (DITA-OT HTML5, strict), csh.json for the page's help panel, and the PDF user guide
#   /dev/      the developer docs (MyST, Sphinx)
#
# Needs DITA_HOME and the [docs] extra.
set -euo pipefail
OUT="${1:-_site}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -rf "$OUT" && mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cp index.html "$OUT/index.html"
cp -r app "$OUT/app"
echo "Help: DITA-OT HTML5 and PDF, strict"
"$DITA_HOME/bin/dita" -i help/alignr-help.ditamap -f html5 -o "$OUT/help" --processing-mode=strict
"$DITA_HOME/bin/dita" -i help/alignr-user-guide.bookmap -f pdf -o "$OUT/help" --processing-mode=strict
python help/tools/build_csh.py "$OUT/help"
echo "Developer docs: Sphinx, warnings are errors"
python -m sphinx -q -W -b html docs "$OUT/dev"
touch "$OUT/.nojekyll"
echo "Built $OUT"
