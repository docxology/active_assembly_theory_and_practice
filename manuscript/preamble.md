# LaTeX preamble

The renderer consumes the fenced block below for the combined PDF. Keeping
the declarations in the source manuscript makes code-listing support explicit
and prevents the template from silently dropping them during hydration.

```latex
\usepackage{graphicx}
\usepackage{listings}
\lstset{basicstyle=\ttfamily\small,breaklines=true,columns=fullflexible}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{remark}[theorem]{Remark}
\newtheorem{example}[theorem]{Example}
```
