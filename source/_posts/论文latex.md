


\subsection{Figures}

\begin{figure}[ht]
\vskip 0.2in
\begin{center}
\centerline{\includegraphics[width=\columnwidth]{icml_numpapers}}
\caption{Historical locations and number of accepted papers for International
Machine Learning Conferences (ICML 1993 -- ICML 2008) and International
Workshops on Machine Learning (ML 1988 -- ML 1992). At the time this figure was
produced, the number of accepted papers for ICML 2008 was unknown and instead
estimated.}
\label{icml-historical}
\end{center}
\vskip -0.2in
\end{figure}

You may want to include figures in the paper to illustrate
your approach and results. Such artwork should be centered,
legible, and separated from the text. Lines should be dark and at
least 0.5~points thick for purposes of reproduction, and text should
not appear on a gray background.

Label all distinct components of each figure. If the figure takes the
form of a graph, then give a name for each axis and include a legend
that briefly describes each curve. Do not include a title inside the
figure; instead, the caption should serve this function.

Number figures sequentially, placing the figure number and caption
\emph{after} the graphics, with at least 0.1~inches of space before
the caption and 0.1~inches after it, as in
\cref{icml-historical}. The figure caption should be set in
9~point type and centered unless it runs two or more lines, in which
case it should be flush left. You may float figures to the top or
bottom of a column, and you may set wide figures across both columns
(use the environment \texttt{figure*} in \LaTeX). Always place
two-column figures at the top or bottom of the page.

\subsection{Algorithms}

If you are using \LaTeX, please use the ``algorithm'' and ``algorithmic''
environments to format pseudocode. These require
the corresponding stylefiles, algorithm.sty and
algorithmic.sty, which are supplied with this package.
\cref{alg:example} shows an example.

\begin{algorithm}[tb]
   \caption{Bubble Sort}
   \label{alg:example}
\begin{algorithmic}
   \STATE {\bfseries Input:} data $x_i$, size $m$
   \REPEAT
   \STATE Initialize $noChange = true$.
   \FOR{$i=1$ {\bfseries to} $m-1$}
   \IF{$x_i > x_{i+1}$}
   \STATE Swap $x_i$ and $x_{i+1}$
   \STATE $noChange = false$
   \ENDIF
   \ENDFOR
   \UNTIL{$noChange$ is $true$}
\end{algorithmic}
\end{algorithm}

\subsection{Tables}

You may also want to include tables that summarize material. Like
figures, these should be centered, legible, and numbered consecutively.
However, place the title \emph{above} the table with at least
0.1~inches of space before the title and the same after it, as in
\cref{sample-table}. The table title should be set in 9~point
type and centered unless it runs two or more lines, in which case it
should be flush left.

% Note use of \abovespace and \belowspace to get reasonable spacing
% above and below tabular lines.

\begin{table}[t]
\caption{Classification accuracies for naive Bayes and flexible
Bayes on various data sets.}
\label{sample-table}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lcccr}
\toprule
Data set & Naive & Flexible & Better? \\
\midrule
Breast    & 95.9$\pm$ 0.2& 96.7$\pm$ 0.2& $\surd$ \\
Cleveland & 83.3$\pm$ 0.6& 80.0$\pm$ 0.6& $\times$\\
Glass2    & 61.9$\pm$ 1.4& 83.8$\pm$ 0.7& $\surd$ \\
Credit    & 74.8$\pm$ 0.5& 78.3$\pm$ 0.6&         \\
Horse     & 73.3$\pm$ 0.9& 69.7$\pm$ 1.0& $\times$\\
Meta      & 67.1$\pm$ 0.6& 76.5$\pm$ 0.5& $\surd$ \\
Pima      & 75.1$\pm$ 0.6& 73.9$\pm$ 0.5&         \\
Vehicle   & 44.9$\pm$ 0.6& 61.5$\pm$ 0.4& $\surd$ \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}

Tables contain textual material, whereas figures contain graphical material.
Specify the contents of each row and column in the table's topmost
row. Again, you may float tables to a column's top or bottom, and set
wide tables across both columns. Place two-column tables at the
top or bottom of the page.

\subsection{Theorems and such}
The preferred way is to number definitions, propositions, lemmas, etc. consecutively, within sections, as shown below.
\begin{definition}
\label{def:inj}
A function $f:X \to Y$ is injective if for any $x,y\in X$ different, $f(x)\ne f(y)$.
\end{definition}
Using \cref{def:inj} we immediate get the following result:
\begin{proposition}
If $f$ is injective mapping a set $X$ to another set $Y$, 
the cardinality of $Y$ is at least as large as that of $X$
\end{proposition}
\begin{proof} 
Left as an exercise to the reader. 
\end{proof}
\cref{lem:usefullemma} stated next will prove to be useful.
\begin{lemma}
\label{lem:usefullemma}
For any $f:X \to Y$ and $g:Y\to Z$ injective functions, $f \circ g$ is injective.
\end{lemma}
\begin{theorem}
\label{thm:bigtheorem}
If $f:X\to Y$ is bijective, the cardinality of $X$ and $Y$ are the same.
\end{theorem}
An easy corollary of \cref{thm:bigtheorem} is the following:
\begin{corollary}
If $f:X\to Y$ is bijective, 
the cardinality of $X$ is at least as large as that of $Y$.
\end{corollary}
\begin{assumption}
The set $X$ is finite.
\label{ass:xfinite}
\end{assumption}
\begin{remark}
According to some, it is only the finite case (cf. \cref{ass:xfinite}) that is interesting.
\end{remark}
%restatable

\subsection{Citations and References}

Please use APA reference format regardless of your formatter
or word processor. If you rely on the \LaTeX\/ bibliographic
facility, use \texttt{natbib.sty} and \texttt{icml2025.bst}
included in the style-file package to obtain this format.

Citations within the text should include the authors' last names and
year. If the authors' names are included in the sentence, place only
the year in parentheses, for example when referencing Arthur Samuel's
pioneering work \yrcite{Samuel59}. Otherwise place the entire
reference in parentheses with the authors and year separated by a
comma \cite{Samuel59}. List multiple references separated by
semicolons \cite{kearns89,Samuel59,mitchell80}. Use the `et~al.'
construct only for citations with three or more authors or after
listing all authors to a publication in an earlier reference \cite{MachineLearningI}.

Authors should cite their own work in the third person
in the initial version of their paper submitted for blind review.
Please refer to \cref{author info} for detailed instructions on how to
cite your own papers.

Use an unnumbered first-level section heading for the references, and use a
hanging indent style, with the first line of the reference flush against the
left margin and subsequent lines indented by 10 points. The references at the
end of this document give examples for journal articles \cite{Samuel59},
conference publications \cite{langley00}, book chapters \cite{Newell81}, books
\cite{DudaHart2nd}, edited volumes \cite{MachineLearningI}, technical reports
\cite{mitchell80}, and dissertations \cite{kearns89}.

Alphabetize references by the surnames of the first authors, with
single author entries preceding multiple author entries. Order
references for the same authors by year of publication, with the
earliest first. Make sure that each reference includes all relevant
information (e.g., page numbers).

Please put some effort into making references complete, presentable, and
consistent, e.g. use the actual current name of authors.
If using bibtex, please protect capital letters of names and
abbreviations in titles, for example, use \{B\}ayesian or \{L\}ipschitz
in your .bib file.


% In the unusual situation where you want a paper to appear in the
% references without citing it in the main text, use \nocite
\nocite{langley00}

# vspace不同位置
以下是关于 LaTeX 中 `\vspace` 命令的详细介绍和教程：

### 1. 基本语法
`\vspace` 命令用于在文档中添加垂直空白。其基本语法如下：
```latex
\vspace{<长度>}
```
- `<长度>` 可以是任何 LaTeX 支持的长度单位，如 `pt`（点）、`mm`（毫米）、`cm`（厘米）、`in`（英寸）等。
### LaTeX 中 `\vspace` 命令详解

#### 1. 基本语法
`\vspace` 命令用于在文档中添加垂直空白，其基本语法为：
```latex
\vspace{<长度>}
```
- `<长度>` 可以是任何 LaTeX 支持的长度单位，如 `pt`（点）、`mm`（毫米）、`cm`（厘米）、`in`（英寸）等。
- 正值表示添加空白，负值表示减少空白。

#### 2. 使用位置及其效果

##### 2.1 在段落中使用
在段落中使用 `\vspace` 会从当前位置开始添加垂直空白。
```latex
This is the first line of text.

\vspace{1cm}
This is the second line of text.
```
这段代码会在两行文本之间添加 1 厘米的空白。

##### 2.2 在标题后使用
在标题后使用 `\vspace` 可以调整标题与正文之间的空白。
```latex
\section{Introduction}

\vspace{0.5cm}
This is the introduction text.
```
这段代码会在 "Introduction" 标题和正文之间添加 0.5 厘米的空白。

##### 2.3 在列表中使用
在列表中使用 `\vspace` 可以调整列表项之间的空白。
```latex
\begin{itemize}
    \item First item

    \vspace{0.3cm}
    \item Second item

    \vspace{0.3cm}
    \item Third item
\end{itemize}
```
这段代码会在每个列表项之间添加 0.3 厘米的空白。

##### 2.4 在表格中使用
在表格中使用 `\vspace` 可以调整表格行之间的空白。
```latex
\begin{tabular}{l l}
    First row & Some text \\

    \vspace{0.2cm}
    Second row & More text \\

    \vspace{0.2cm}
    Third row & Final text
\end{tabular}
```
这段代码会在表格的每一行之间添加 0.2 厘米的空白。

##### 2.5 在图像或图表中使用
在图像或图表中使用 `\vspace` 可以调整图像与标题或其他内容之间的空白。
```latex
\begin{figure}
    \centering
    \includegraphics[width=\linewidth]{example_image.png}

    \vspace{-0.5cm}
    \caption{This is the caption for the image.}
    \label{fig:example_image}
\end{figure}
```
这段代码会在图像和标题之间减少 0.5 厘米的空白。

#### 3. 常见用法和技巧

##### 3.1 调整段落间距
```latex
This is the first paragraph.

\vspace{1cm}
This is the second paragraph.
```

##### 3.2 调整标题间距
```latex
\section{Introduction}

\vspace{0.5cm}
This is the introduction text.
```

##### 3.3 调整列表间距
```latex
\begin{itemize}
    \item First item

    \vspace{0.3cm}
    \item Second item

    \vspace{0.3cm}
    \item Third item
\end{itemize}
```

##### 3.4 调整表格间距
```latex
\begin{tabular}{l l}
    First row & Some text \\

    \vspace{0.2cm}
    Second row & More text \\

    \vspace{0.2cm}
    Third row & Final text
\end{tabular}
```

##### 3.5 调整图像间距
```latex
\begin{figure}
    \centering
    \includegraphics[width=\linewidth]{example_image.png}

    \vspace{-0.5cm}
    \caption{This is the caption for the image.}
    \label{fig:example_image}
\end{figure}
```

#### 4. 注意事项

##### 4.1 负值的使用
使用负值可以减少空白，但需谨慎，因为过度减少可能导致内容重叠。

##### 4.2 不同环境中的表现
`\vspace` 在不同环境中的表现可能略有不同。例如，在表格中可能需要结合 `\arraystretch` 来调整行高。

##### 4.3 与其他命令的结合使用
可以结合其他命令（如 `\smallskip`、`\medskip`、`\bigskip`）来调整空白。这些命令分别对应较小、中等和较大的空白。

#### 5. 示例代码
以下是一个综合示例，展示 `\vspace` 在不同场景中的使用：
```latex
\documentclass{article}
\usepackage{graphicx}

\begin{document}

\section{Introduction}

\vspace{0.5cm}
This is the introduction text. We will demonstrate the use of \texttt{\vspace} in different contexts.

\vspace{1cm}
\subsection{Paragraph Spacing}

This is the first paragraph.

\vspace{0.5cm}
This is the second paragraph with additional spacing.

\vspace{1cm}
\subsection{List Spacing}

\begin{itemize}
    \item First item

    \vspace{0.3cm}
    \item Second item with extra spacing

    \vspace{0.3cm}
    \item Third item
\end{itemize}

\vspace{1cm}
\subsection{Table Spacing}

\begin{tabular}{l l}
    First row & Some text \\

    \vspace{0.2cm}
    Second row & More text \\

    \vspace{0.2cm}
    Third row & Final text
\end{tabular}

\vspace{1cm}
\subsection{Figure Spacing}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.5\linewidth]{example_image.png}

    \vspace{-0.5cm}
    \caption{This is the caption for the image.}
    \label{fig:example_image}
\end{figure}

\end{document}
```


# 表格
top bottom midrule
颜色
caption
scale

# 插图片

\begin{figure}
\small
\centering
\vspace{-.1cm}
    \includegraphics[width=\linewidth]{figs/fig5-llms-judge.pdf}
\vspace{-.5cm}
\caption{\small
We manually prompt four different DeepResearch variants to generate research plans, which were then evaluated by five independent LLMs across eight dimensions, with scores ranging from 1 to 10. Detailed prompt, outputs and scores are provided in Appendix~\ref{app:llm_as_judge_detail}.
}
\vspace{-.5cm}
\label{fig:llm-judge}
\end{figure}

双栏 wrapfigure无效
要用minipage 但minipage不能环绕

# wrapfigure

adjust box

# 半列

[!htbp]

\begin{table}
\centering \vspace{-.5cm} 
\caption{\small Ablation study on the impact of key framework components. Performance comparison of different scAgents configurations across gene knockout, drug, and cytokine perturbation datasets.}\vspace{-.2cm}
\scalebox{0.8}{
\resizebox{0.5\columnwidth}{!}{
\begin{tabular}{lcccccc}
\toprule\midrule
\textsc{Model} & \texttt{MSE} $\downarrow$ & \texttt{PCC} $\uparrow$ & \texttt{$R^2$} $\uparrow$ &\texttt{MSE(DE)} $\downarrow$ & \texttt{PCC(DE)} $\uparrow$ & \texttt{$R^2$(DE)} $\uparrow$ \\
\toprule\midrule
\rowcolor[RGB]{230, 255, 230}
\multicolumn{7}{c}{\it Gene Knock Out Perturbation (Adamson Dataset \cite{adamson2016multiplexed})} \\
\midrule\bottomrule
scAgents(baseline) & 0.4776 & 0.0087 & 0.0410 & 0.6061 & 0.0940 & 0.1280 \\
\hdashline
+ Normal RAG & 0.2442 & 0.1008 & 0.1119 & 0.3997 & 0.3354 & 0.3667 \\
+ Agentic Retrieval & 0.1267 & 0.5643 & 0.5431 & \textbf{0.1152} & 0.5922 & 0.6067 \\
+ Graph-Based Discussion & 0.2751 & 0.5310 & 0.5874 & 0.2792 & 0.6540 & 0.5311 \\
+ Normal RAG \& Graph-Based Discussion & 0.0909 & 0.8951 & 0.8658 & 0.3416 & 0.8547 & 0.6770 \\
+ Agentic Retrieval \& Graph-Based Discussion & \textbf{0.0051} & \textbf{0.9883} & \textbf{0.9761} & 0.2013 & \textbf{0.9474} & \textbf{0.8912} \\ 
\toprule\midrule
\rowcolor[RGB]{230, 255, 230}
\multicolumn{7}{c}{\it Drug Perturbation (Srivatsan Dataset \cite{srivatsan2020massively})} \\
\midrule\bottomrule
scAgents(baseline) & 0.5760 & 0.0298 & 0.0475 & 0.6409 & 0.0992 &  0.1039 \\
\hdashline
+ Normal RAG & 0.2572 & 0.1584 & 0.1038 & 0.3022 & 0.3472 & 0.2901 \\
+ Agentic Retrieval & 0.1309 & 0.3437 & 0.4350 & 0.1210 & 0.3836 &  0.4169 \\
+ Graph-Based Discussion & 0.1670 & 0.4193 & 0.3764 & 0.1325 & 0.4266 & 0.3865 \\
+ Normal RAG \& Graph-Based Discussion & 0.0995  & 0.6512 & 0.5933 & 0.985 & 0.6784  & 0.7548 \\
+ Agentic Retrieval \& Graph-Based Discussion & \textbf{0.0053} & \textbf{0.9881} & \textbf{0.9665} & \textbf{0.0080} & \textbf{0.9953} & \textbf{0.9802} \\
\toprule\midrule
\rowcolor[RGB]{230, 255, 230}
 \multicolumn{7}{c}{\it Cytokine Perturbation (Schiebinger Dataset \cite{schiebinger2019optimal})} \\
\midrule\bottomrule
scAgents-Model (baseline) & 0.5892 & 0.0065 & 0.0021 & 0.5876 & 0.0797 & 0.0999 \\
\hdashline
+ Normal RAG & 0.4321 & 0.1765 & 0.0243 & 0.4756 & 0.1987 & 0.0934 \\
+ Agentic Retrieval & 0.3456 & 0.2034 & 0.2421 & 0.3076 & 0.2068 & 0.1176 \\
+ Graph-Based Discussion & 0.3512 & 0.2051 & 0.2765 & 0.2454 & 0.2239 & 0.1123 \\
+ Normal RAG \& Graph-Based Discussion & 0.0987 & 0.4875 & 0.4654 & 0.1065 & 0.2534 & 0.1053 \\
+ Agentic Retrieval \& Graph-Based Discussion & \textbf{0.0428} & \textbf{0.5697} & \textbf{0.5042} & \textbf{0.0144} & \textbf{0.3396} & \textbf{0.1240} \\
\midrule\bottomrule
\end{tabular}
}}
\vspace{-.7cm}
\label{tab:ablation_study}
\end{table}


# 框引用lstlisting


# .bib引用格式