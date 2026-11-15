"""Static checks on the paper sources (no LaTeX needed).

  1. every capitalised macro used in paper/*.tex and paper/sections/*.tex is
     defined in paper/numbers.tex (or is a known LaTeX command);
  2. every \\input{...} and \\includegraphics{...} target exists;
  3. every \\ref{...} has a matching \\label{...};
  4. HAND-TYPED NUMBERS: any number with a decimal point, or of two or more
     digits, in the prose or math of the sources -- outside macro names,
     \\ref/\\label/\\input/\\includegraphics/\\cite arguments and comments --
     is reported, so that every result number comes from make_tables.py.
     Structural labels are allowed: refinement levels (L0-L4), the IEEE
     14-bus system name, the 95% confidence level.

    python scripts/lint_paper.py      # exit status 1 if anything is reported
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
KNOWN = {
    "IEEEoverridecommandlockouts", "IEEEauthorblockN", "IEEEauthorblockA", "LaTeX", "Delta", "Pr", "Bigl", "Bigr",
}
ALLOWED_NUMBERS = [r"\bL[0-4]\b", r"14-bus", r"95\\%", r"\bp\{=\}1\b", r"_\{10\}"]


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def main():
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (PAPER / "numbers.tex").read_text(encoding="utf-8")))
    sources = [PAPER / "main.tex", PAPER / "supplement.tex"] + sorted((PAPER / "sections").glob("*.tex"))
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8") if (PAPER / "refs.bib").exists() else ""
    bibkeys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    for f in sources:
        for group in re.findall(r"\\cite\{([^}]+)\}", strip_comments(f.read_text(encoding="utf-8"))):
            for key in (k.strip() for k in group.split(",")):
                if key not in bibkeys:
                    print(f"  {f.name}: \\cite key not in refs.bib: {key}")
                    bibkeys_missing = True
    problems = []
    labels, refs = set(), []
    for f in sources:
        raw = f.read_text(encoding="utf-8")
        t = strip_comments(raw)
        labels |= set(re.findall(r"\\label\{([^}]+)\}", t))
        refs += [(f.name, r) for r in re.findall(r"\\ref\{([^}]+)\}", t)]
        for m in set(re.findall(r"\\([A-Z][A-Za-z]+)", t)):
            if m not in defined and m not in KNOWN:
                problems.append(f"{f.name}: undefined macro \\{m}")
        for tgt in re.findall(r"\\input\{([^}]+)\}", t):
            p = PAPER / (tgt if tgt.endswith(".tex") else tgt + ".tex")
            if not p.exists():
                problems.append(f"{f.name}: \\input target missing: {tgt}")
        for tgt in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t):
            if not (PAPER / tgt).exists():
                problems.append(f"{f.name}: figure missing: {tgt}")
        # hand-typed numbers
        body = re.sub(r"\\(ref|label|input|includegraphics|cite|begin|end)(\[[^\]]*\])?\{[^}]*\}", " ", t)
        body = re.sub(r"\\[A-Za-z]+", " ", body)  # macro names (may contain no digits anyway)
        body = re.sub(r"\[[0-9.]+(in|pt|cm)\]", " ", body)
        for pat in ALLOWED_NUMBERS:
            body = re.sub(pat, " ", body)
        for ln, line in enumerate(body.splitlines(), 1):
            for num in re.findall(r"(?<![A-Za-z_])(\d+\.\d+|\d{2,})(?![A-Za-z])", line):
                problems.append(f"{f.name}:{ln}: hand-typed number '{num}': {line.strip()[:90]}")
    for fname, r in refs:
        if r not in labels:
            problems.append(f"{fname}: \\ref{{{r}}} has no \\label")
    used = set()
    for f in sources:
        used |= set(re.findall(r"\\([A-Z][A-Za-z]+)", strip_comments(f.read_text(encoding="utf-8"))))
    for tf in (PAPER / "tables").glob("*.tex"):
        used |= set(re.findall(r"\\([A-Z][A-Za-z]+)", tf.read_text(encoding="utf-8")))
        for m in set(re.findall(r"\\([A-Z][A-Za-z]+)", tf.read_text(encoding="utf-8"))):
            if m not in defined and m not in KNOWN:
                problems.append(f"tables/{tf.name}: undefined macro \\{m}")
    print(f"{len(sources)} source files, {len(defined)} macros defined, {len(used & defined)} used, "
          f"{len(labels)} labels, {len(refs)} refs")
    for p in problems:
        print("  " + p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
