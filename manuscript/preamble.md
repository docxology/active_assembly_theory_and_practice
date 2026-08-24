# LaTeX preamble

The renderer consumes the fenced block below for the combined PDF. Keeping
the declarations in the source manuscript makes code-listing support explicit
and prevents the template from silently dropping them during hydration.

```latex
\usepackage{graphicx}
\usepackage{listings}
\lstset{basicstyle=\ttfamily\small,breaklines=true,columns=fullflexible}
% Keep the generated bibliography compact enough to avoid orphaned references
% while retaining a readable minimum size and the template's tagged structure.
\AtBeginEnvironment{thebibliography}{%
  \small
  \setlength{\itemsep}{0.15em plus 0.05em minus 0.05em}%
  \setlength{\parskip}{0pt}%
}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{remark}[theorem]{Remark}
\newtheorem{example}[theorem]{Example}
```
