#!/usr/bin/env bash
# Build the paper and the supplement with pdflatex + bibtex (MiKTeX or TeX Live on PATH; Git Bash on Windows).
# Output and all intermediate files go to paper/build/ (git-ignored).
#
#   bash scripts/build_paper.sh            # main.pdf and supplement.pdf
#   bash scripts/build_paper.sh main       # one document
#
# Prints, per document: page count, undefined references/citations, overfull lines wider than 5pt.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
P="$ROOT/paper"
O="$P/build"
mkdir -p "$O"
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else echo "$1"; fi; }
SEP=":"; command -v cygpath >/dev/null 2>&1 && SEP=";"
cd "$P" || exit 1
status=0
for DOC in "${@:-main supplement}"; do
  for DOC in $DOC; do
    run() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$(native "$O")" "$DOC.tex" > "$O/$DOC.run.log" 2>&1; }
    if ! run; then
      echo "$DOC: pdflatex FAILED"; grep -n -A6 "^!" "$O/$DOC.log" | head -30; status=1; continue
    fi
    if grep -q '\\bibdata' "$O/$DOC.aux"; then
      (cd "$O" && BIBINPUTS="$(native "$P")$SEP" BSTINPUTS="$(native "$P")$SEP" bibtex "$DOC" > "$DOC.bib.log" 2>&1)
      grep -i -E "^warning|error" "$O/$DOC.bib.log" | grep -v -i miktex | head -10
    fi
    run; run || { echo "$DOC: pdflatex FAILED on the final pass"; status=1; continue; }
    pages=$(tr -d '\r\n' < "$O/$DOC.log" | grep -o "Output written on [^(]*([0-9]* page" | grep -o "[0-9]* page" | grep -o "[0-9]*")
    undef=$(grep -c -E "Reference .* undefined|Citation .* undefined" "$O/$DOC.log")
    over=$(grep -E "Overfull .hbox" "$O/$DOC.log" | awk -F'[(p]' '{ if ($2+0 > 5) n++ } END { print n+0 }')
    echo "$DOC: $pages pages, $undef undefined references/citations, $over overfull lines > 5pt -> paper/build/$DOC.pdf"
  done
done
exit $status
