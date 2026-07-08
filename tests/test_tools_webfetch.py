"""Tests for the WebFetch tool."""

import json
from unittest.mock import patch

from tiz.tools.webfetch import WebFetch, _html_to_markdown

_TEST_SOCKET = "/tmp/test.sock"


class TestHTMLToMarkdown:
    def test_empty_html(self) -> None:
        assert _html_to_markdown("") == ""

    def test_plain_text(self) -> None:
        assert _html_to_markdown("hello world") == "hello world"

    def test_headings(self) -> None:
        md = _html_to_markdown("<h1>Title</h1><h2>Sub</h2><h3>H3</h3>")
        assert "# Title" in md
        assert "## Sub" in md
        assert "### H3" in md

    def test_paragraphs(self) -> None:
        md = _html_to_markdown("<p>First</p><p>Second</p>")
        assert md == "First\n\nSecond"

    def test_bold_and_strong(self) -> None:
        md = _html_to_markdown("<b>bold</b> and <strong>strong</strong>")
        assert md == "**bold** and **strong**"

    def test_italic_and_em(self) -> None:
        md = _html_to_markdown("<i>italic</i> and <em>em</em>")
        assert md == "*italic* and *em*"

    def test_nested_bold_italic(self) -> None:
        md = _html_to_markdown("<b><i>both</i></b>")
        assert md == "***both***"

    def test_code(self) -> None:
        md = _html_to_markdown("<code>print(1)</code>")
        assert md == "`print(1)`"

    def test_code_with_backticks(self) -> None:
        """Bug 9: Code containing backticks uses appropriate delimiter."""
        md = _html_to_markdown("<code>a `b` c</code>")
        assert md == "``a `b` c``", f"Got: {md!r}"

    def test_code_with_multiple_backticks(self) -> None:
        """Bug 9: Code containing double backticks uses triple-delimiter."""
        md = _html_to_markdown("<code>a ``b`` c</code>")
        assert md == "```a ``b`` c```", f"Got: {md!r}"

    def test_pre(self) -> None:
        md = _html_to_markdown("<pre>line1\nline2</pre>")
        assert "```" in md
        assert "line1" in md
        assert "line2" in md

    def test_pre_with_link(self) -> None:
        """Bug 2: Links inside <pre> should produce valid Markdown."""
        md = _html_to_markdown('<pre><a href="x">link</a></pre>')
        assert "link" in md
        assert "[link](x)" in md, f"Got: {md!r}"

    def test_br(self) -> None:
        md = _html_to_markdown("a<br>b")
        assert "a" in md
        assert "b" in md

    def test_hr(self) -> None:
        md = _html_to_markdown("<hr>")
        assert "---" in md

    def test_link(self) -> None:
        md = _html_to_markdown('<a href="https://x.com">click</a>')
        assert md == "[click](https://x.com)"

    def test_link_no_href(self) -> None:
        md = _html_to_markdown("<a>text</a>")
        assert md == "text"

    def test_nested_links(self) -> None:
        """Bug 3: Nested <a> tags should handle outer link correctly."""
        md = _html_to_markdown('<a href="a">outer <a href="b">inner</a> outer2</a>')
        assert md == "[outer [inner](b) outer2](a)", f"Got: {md!r}"

    def test_nested_links_inner_empty(self) -> None:
        """Nested <a> with empty inner link should render correctly."""
        md = _html_to_markdown('<a href="a">outer <a href="b"></a> outer2</a>')
        assert md == "[outer [](b) outer2](a)", f"Got: {md!r}"

    def test_nested_links_inner_no_href_no_text(self) -> None:
        """Nested <a> where inner has no href and no text (rendered is empty)."""
        md = _html_to_markdown('<a href="a"><a></a>after</a>')
        assert md == "[after](a)", f"Got: {md!r}"

    def test_image(self) -> None:
        md = _html_to_markdown('<img src="x.png" alt="pic">')
        assert md == "![pic](x.png)"

    def test_simple_table(self) -> None:
        html = (
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        )
        md = _html_to_markdown(html)
        assert "| A | B |" in md
        assert "| - | - |" in md
        assert "| 1 | 2 |" in md

    def test_nested_tables(self) -> None:
        """Bug 4: Nested tables should not corrupt outer table state."""
        html = (
            "<table><tr><td><table><tr><td>nested</td></tr></table></td></tr></table>"
        )
        md = _html_to_markdown(html)
        assert "nested" in md, f"Got: {md!r}"

    def test_table_nested_outside_cell_empty(self) -> None:
        """Nested table with no rows restored outside a cell."""
        html = "<table><table></table></table>"
        md = _html_to_markdown(html)
        assert md == "", f"Got: {md!r}"

    def test_table_nested_outside_cell(self) -> None:
        """Nested table restored when in_cell is False should still work."""
        html = "<table><table><tr><td>inner</td></tr></table></table>"
        md = _html_to_markdown(html)
        assert "inner" in md, f"Got: {md!r}"

    def test_table_with_mixed_rows(self) -> None:
        html = "<table><tr><td>a</td></tr><tr><td>b</td><td>c</td></tr></table>"
        md = _html_to_markdown(html)
        assert "| a" in md
        assert "| b | c |" in md

    def test_empty_table(self) -> None:
        assert _html_to_markdown("<table></table>") == ""

    def test_unordered_list(self) -> None:
        md = _html_to_markdown("<ul><li>A</li><li>B</li></ul>")
        lines = md.split("\n")
        assert any("- A" in line for line in lines)
        assert any("- B" in line for line in lines)

    def test_ordered_list(self) -> None:
        md = _html_to_markdown("<ol><li>First</li><li>Second</li></ol>")
        assert "1. First" in md
        assert "2. Second" in md

    def test_orphan_li(self) -> None:
        """Bug 6: Orphan <li> outside any list should not produce spurious markers."""
        md = _html_to_markdown("<li>orphan</li><p>after</p>")
        assert "orphan" in md
        assert "after" in md

    def test_blockquote(self) -> None:
        md = _html_to_markdown("<blockquote>quote</blockquote>")
        assert ">" in md
        assert "quote" in md

    def test_stray_close_blockquote(self) -> None:
        """Bug 10: Stray </blockquote> should not add spurious blank lines."""
        md = _html_to_markdown("</blockquote><p>text</p>")
        assert "text" in md

    def test_skip_script_style(self) -> None:
        md = _html_to_markdown(
            "<script>evil()</script><style>body{}</style><p>good</p>"
        )
        assert "evil" not in md
        assert "body{}" not in md
        assert "good" in md

    def test_skip_nested_skip_tags(self) -> None:
        md = _html_to_markdown("<script>a<style>hidden</style>c</script><p>visible</p>")
        assert "a" not in md
        assert "hidden" not in md
        assert "c" not in md
        assert "visible" in md

    def test_skip_endtag_of_non_skip(self) -> None:
        md = _html_to_markdown("<script><div>ignored</div>still</script><p>ok</p>")
        assert "ignored" not in md
        assert "still" not in md
        assert "ok" in md

    def test_code_inside_pre(self) -> None:
        md = _html_to_markdown("<pre><code>block</code></pre>")
        assert "```" in md
        assert "block" in md

    def test_close_a_when_not_in_link(self) -> None:
        md = _html_to_markdown("</a><p>text</p>")
        assert "text" in md

    def test_link_with_href_no_text(self) -> None:
        md = _html_to_markdown('<a href="https://x.com"></a>')
        assert "[](https://x.com)" in md

    def test_link_no_href_no_text(self) -> None:
        md = _html_to_markdown("<a></a><p>ok</p>")
        assert "ok" in md

    def test_tr_with_no_cells(self) -> None:
        md = _html_to_markdown("<table><tr></tr><tr><td>x</td></tr></table>")
        assert "| x |" in md

    def test_td_without_tr_flushes_as_row(self) -> None:
        """Malformed <td> without <tr> should still render its content."""
        md = _html_to_markdown("<table><td>orphan</td></table>")
        assert "orphan" in md, f"Got: {md!r}"
        assert "| orphan |" in md, f"Got: {md!r}"

    def test_td_without_tr_multiple_cells(self) -> None:
        """Multiple <td> without <tr> should produce a single row."""
        md = _html_to_markdown("<table><td>a</td><td>b</td></table>")
        assert "| a | b |" in md, f"Got: {md!r}"

    def test_td_without_tr_then_tr(self) -> None:
        """Orphan <td> followed by <tr> should flush orphan as its own row."""
        md = _html_to_markdown("<table><td>orphan</td><tr><td>row</td></tr></table>")
        assert "orphan" in md, f"Got: {md!r}"
        assert "row" in md, f"Got: {md!r}"

    def test_close_list_with_empty_stack(self) -> None:
        md = _html_to_markdown("</ul><p>x</p>")
        assert "x" in md

    def test_table_all_empty_cells(self) -> None:
        md = _html_to_markdown("<table><tr><td></td><td></td></tr></table>")
        assert "|  |  |" in md

    def test_endtag_non_skip_in_skip(self) -> None:
        md = _html_to_markdown("<head><div>hidden</div></head><p>visible</p>")
        assert "hidden" not in md
        assert "visible" in md

    def test_nested_skip_tag(self) -> None:
        md = _html_to_markdown("<head><title>hidden</title></head><p>visible</p>")
        assert "hidden" not in md
        assert "visible" in md

    def test_skip_head_title_noscript_iframe(self) -> None:
        """head, title, noscript, iframe content is skipped.
        meta and link are void elements and cannot contain content, so they are not in _SKIP.
        """
        md = _html_to_markdown(
            "<head>x</head><title>t</title>"
            "<noscript>ns</noscript><iframe>fr</iframe><p>ok</p>"
        )
        assert "x" not in md
        assert "t" not in md
        assert "ns" not in md
        assert "fr" not in md
        assert "ok" in md

    def test_entity_decoding(self) -> None:
        md = _html_to_markdown("<p>&amp; &lt; &gt; &quot;</p>")
        assert "&" in md
        assert "<" in md
        assert ">" in md
        assert '"' in md

    def test_html_escaped_href(self) -> None:
        """Bug 1: HTML entities in href should be decoded once (no double decode)."""
        md = _html_to_markdown('<a href="https://x.com?a=1&amp;b=2">link</a>')
        assert "[link](https://x.com?a=1&b=2)" in md

    def test_html_double_escaped_href(self) -> None:
        """Bug 1: Double &amp;amp; should NOT be over-decoded."""
        md = _html_to_markdown('<a href="https://x.com?a=1&amp;amp;b=2">link</a>')
        assert "[link](https://x.com?a=1&amp;b=2)" in md, f"Got: {md!r}"

    def test_collapse_whitespace(self) -> None:
        md = _html_to_markdown("<p>a   b\n\nc</p>")
        assert "a b c" in md

    def test_cell_whitespace_normalized(self) -> None:
        """Bug 5: Whitespace in table cells should be normalized."""
        md = _html_to_markdown("<table><tr><td>a   b\n  c</td></tr></table>")
        assert "a b c" in md, f"Got: {md!r}"

    def test_div_section_etc_end_tags(self) -> None:
        md = _html_to_markdown(
            "<div>a</div><section>b</section><article>c</article>"
            "<header>d</header><footer>e</footer><main>f</main>"
            "<nav>g</nav><aside>h</aside>"
        )
        for ch in "abcdefgh":
            assert ch in md

    # --- Bug fix tests ---

    def test_pre_opening_fence(self) -> None:
        """<pre> blocks must have opening triple-backtick."""
        md = _html_to_markdown("<pre>code block</pre>")
        assert md.startswith("```"), f"Expected opening ```, got: {md!r}"
        assert "code block" in md
        assert "```" in md

    def test_bold_inside_link(self) -> None:
        """Bold inside <a> should produce [**text**](url)."""
        md = _html_to_markdown('<a href="http://x.com"><b>bold</b></a>')
        assert md == "[**bold**](http://x.com)", f"Got: {md!r}"

    def test_italic_inside_link(self) -> None:
        """Italic inside <a> should produce [*text*](url)."""
        md = _html_to_markdown('<a href="http://x.com"><i>italic</i></a>')
        assert md == "[*italic*](http://x.com)", f"Got: {md!r}"

    def test_bold_italic_inside_link(self) -> None:
        """Bold+italic inside <a> should produce [***text***](url)."""
        md = _html_to_markdown('<a href="http://x.com"><b><i>both</i></b></a>')
        assert md == "[***both***](http://x.com)", f"Got: {md!r}"

    def test_code_inside_link(self) -> None:
        """Inline code inside <a> should produce [`code`](url)."""
        md = _html_to_markdown('<a href="http://x.com"><code>print(1)</code></a>')
        assert md == "[`print(1)`](http://x.com)", f"Got: {md!r}"

    def test_table_pipe_escaping(self) -> None:
        """Pipe characters in table cells must be escaped."""
        html = "<table><tr><td>a|b</td></tr></table>"
        md = _html_to_markdown(html)
        assert "\\|" in md, f"Pipe not escaped in: {md!r}"

    def test_ordered_list_start_attribute(self) -> None:
        """<ol start=\"N\"> should start numbering from N."""
        html = '<ol start="42"><li>A</li><li>B</li></ol>'
        md = _html_to_markdown(html)
        assert "42. A" in md, f"Expected start=42, got: {md!r}"
        assert "43. B" in md, f"Expected second item 43, got: {md!r}"

    def test_nested_blockquotes(self) -> None:
        """Nested blockquotes should get > and >> prefixes."""
        html = "<blockquote>outer<blockquote>inner</blockquote></blockquote>"
        md = _html_to_markdown(html)
        assert md == "> outer\n\n>> inner", f"Got: {md!r}"

    # --- inline formatting inside table cells ---

    def test_bold_inside_table_cell(self) -> None:
        """Bold markers should route to cell buffer, not main output."""
        html = "<table><tr><td><b>Hello</b></td></tr></table>"
        md = _html_to_markdown(html)
        assert "**Hello**" in md, f"Got: {md!r}"
        assert md.count("**") == 2, f"Got: {md!r}"

    def test_italic_inside_table_cell(self) -> None:
        """Italic markers should route to cell buffer."""
        html = "<table><tr><td><i>Italic</i></td></tr></table>"
        md = _html_to_markdown(html)
        assert "*Italic*" in md, f"Got: {md!r}"

    def test_code_inside_table_cell(self) -> None:
        """Code markers should route to cell buffer."""
        html = "<table><tr><td><code>code</code></td></tr></table>"
        md = _html_to_markdown(html)
        assert "`code`" in md, f"Got: {md!r}"

    # --- links inside table cells ---

    def test_link_inside_table_cell(self) -> None:
        """Link text and href should produce [text](url) inside cell."""
        html = '<table><tr><td><a href="x">link</a></td></tr></table>'
        md = _html_to_markdown(html)
        assert "[link](x)" in md, f"Got: {md!r}"

    def test_formatted_link_inside_table_cell(self) -> None:
        """Bold link inside table cell should render cell with bold link."""
        html = '<table><tr><td><a href="x"><b>bold link</b></a></td></tr></table>'
        md = _html_to_markdown(html)
        assert "[**bold link**](x)" in md, f"Got: {md!r}"

    # --- blockquote multi-paragraph ---

    def test_blockquote_multi_paragraph(self) -> None:
        """Each paragraph in a blockquote should get the > prefix."""
        html = "<blockquote><p>Line 1</p><p>Line 2</p></blockquote>"
        md = _html_to_markdown(html)
        assert "> Line 1" in md, f"Got: {md!r}"
        assert "> Line 2" in md, f"Got: {md!r}"

    def test_blockquote_single_paragraph(self) -> None:
        """Single paragraph in blockquote should still get > prefix."""
        html = "<blockquote><p>Single</p></blockquote>"
        md = _html_to_markdown(html)
        assert "> Single" in md, f"Got: {md!r}"

    # --- <br> and <img> inside <a> tags ---

    def test_br_inside_link(self) -> None:
        """<br> inside <a> should accumulate into link text."""
        md = _html_to_markdown('<a href="http://x.com">Click<br>here</a>')
        assert md == "[Click\nhere](http://x.com)", f"Got: {md!r}"

    def test_img_inside_link(self) -> None:
        """<img> inside <a> should accumulate into link text."""
        md = _html_to_markdown(
            '<a href="http://x.com"><img src="pic.png" alt="pic"></a>'
        )
        assert md == "[![pic](pic.png)](http://x.com)", f"Got: {md!r}"


class TestHTMLToMarkdownSecurity:
    """Security-focused tests for the HTML-to-Markdown converter."""

    def test_javascript_url_in_link_stripped(self) -> None:
        md = _html_to_markdown('<a href="javascript:alert(1)">click</a>')
        assert md == "click", f"Got: {md!r}"
        assert "javascript" not in md
        assert "alert" not in md

    def test_data_url_in_link_stripped(self) -> None:
        md = _html_to_markdown(
            '<a href="data:text/html,<script>alert(1)</script>">click</a>'
        )
        assert md == "click", f"Got: {md!r}"
        assert "data:" not in md
        assert "script" not in md
        assert "alert" not in md

    def test_javascript_url_in_image_stripped(self) -> None:
        md = _html_to_markdown('<img src="javascript:alert(1)" alt="x">')
        assert md == "x", f"Got: {md!r}"
        assert "javascript" not in md
        assert "alert" not in md

    def test_data_url_in_image_stripped(self) -> None:
        md = _html_to_markdown('<img src="data:image/svg+xml,<svg onload=alert(1)>">')
        assert md == "", f"Got: {md!r}"
        assert "data:" not in md
        assert "onload" not in md

    def test_vbscript_url_stripped(self) -> None:
        md = _html_to_markdown('<a href="vbscript:msgbox(1)">click</a>')
        assert md == "click", f"Got: {md!r}"
        assert "vbscript" not in md

    def test_link_text_markdown_injection_escaped(self) -> None:
        md = _html_to_markdown('<a href="http://x.com">[evil](http://evil.com)</a>')
        assert "[evil](http://evil.com)" not in md, f"Got: {md!r}"
        assert "\\[evil\\](http://evil.com)" in md, f"Got: {md!r}"
        assert "http://x.com" in md

    def test_img_alt_markdown_injection_escaped(self) -> None:
        md = _html_to_markdown('<img src="x.png" alt="]evil](http://bad)">')
        assert "]evil](http://bad)" not in md, f"Got: {md!r}"
        assert "\\]evil\\](http://bad)" in md, f"Got: {md!r}"

    def test_href_closing_paren_escaped(self) -> None:
        md = _html_to_markdown('<a href="http://x.com/path)">link</a>')
        assert md == "[link](http://x.com/path\\))", f"Got: {md!r}"

    def test_href_newline_neutralized(self) -> None:
        md = _html_to_markdown('<a href="http://x.com\n\n[evil](http://bad)">link</a>')
        assert "\n" not in md, f"Newline leaked into output: {md!r}"
        assert md.startswith("[link]"), f"Got: {md!r}"
        assert md.endswith(")"), f"Link should end with closing paren: {md!r}"
        assert "\\[evil\\]" in md, f"Brackets in URL should be escaped: {md!r}"
        assert "[evil]" not in md.replace("\\[evil\\]", ""), f"Unescaped link: {md!r}"

    def test_pre_with_backticks_uses_longer_fence(self) -> None:
        md = _html_to_markdown("<pre>a ``` b</pre>")
        assert md.startswith("````"), f"Got: {md!r}"
        assert md.endswith("````"), f"Got: {md!r}"

    def test_pre_with_double_backticks_uses_triple_fence(self) -> None:
        md = _html_to_markdown("<pre>a `` b</pre>")
        assert md.startswith("```"), f"Got: {md!r}"
        assert md.endswith("```"), f"Got: {md!r}"

    def test_pre_empty_uses_triple_fence(self) -> None:
        md = _html_to_markdown("<pre></pre>")
        assert md == "```\n\n```", f"Got: {md!r}"

    def test_nested_pre_blocks(self) -> None:
        md = _html_to_markdown("<pre>outer<pre>inner</pre>after</pre>")
        assert "outerinnerafter" in md, f"Got: {md!r}"
        assert md.count("```") == 2, f"Got: {md!r}"

    def test_relative_url_preserved(self) -> None:
        md = _html_to_markdown('<a href="/path">click</a>')
        assert md == "[click](/path)", f"Got: {md!r}"

    def test_anchor_url_preserved(self) -> None:
        md = _html_to_markdown('<a href="#section">click</a>')
        assert md == "[click](#section)", f"Got: {md!r}"

    def test_mailto_url_preserved(self) -> None:
        md = _html_to_markdown('<a href="mailto:a@b.com">email</a>')
        assert md == "[email](mailto:a@b.com)", f"Got: {md!r}"

    def test_tel_url_preserved(self) -> None:
        md = _html_to_markdown('<a href="tel:+1234567890">call</a>')
        assert md == "[call](tel:+1234567890)", f"Got: {md!r}"

    def test_ftp_url_preserved(self) -> None:
        md = _html_to_markdown('<a href="ftp://ftp.example.com">download</a>')
        assert md == "[download](ftp://ftp.example.com)", f"Got: {md!r}"

    def test_img_no_src_renders_alt_text(self) -> None:
        md = _html_to_markdown('<img alt="description">')
        assert md == "description", f"Got: {md!r}"

    def test_img_javascript_src_renders_alt_text(self) -> None:
        md = _html_to_markdown('<img src="javascript:alert(1)" alt="safe text">')
        assert md == "safe text", f"Got: {md!r}"

    def test_br_inside_pre(self) -> None:
        md = _html_to_markdown("<pre>a<br>b</pre>")
        assert "a\nb" in md, f"Got: {md!r}"

    def test_hr_inside_pre(self) -> None:
        md = _html_to_markdown("<pre>a<hr>b</pre>")
        assert "---" in md, f"Got: {md!r}"
        assert "a" in md
        assert "b" in md

    def test_bold_inside_pre(self) -> None:
        md = _html_to_markdown("<pre><b>bold</b></pre>")
        assert "**bold**" in md, f"Got: {md!r}"

    def test_italic_inside_pre(self) -> None:
        md = _html_to_markdown("<pre><i>italic</i></pre>")
        assert "*italic*" in md, f"Got: {md!r}"

    def test_bold_endtag_inside_pre(self) -> None:
        md = _html_to_markdown("<pre><b>bold</b> after</pre>")
        assert "**bold** after" in md, f"Got: {md!r}"

    def test_italic_endtag_inside_pre(self) -> None:
        md = _html_to_markdown("<pre><i>italic</i> after</pre>")
        assert "*italic* after" in md, f"Got: {md!r}"

    def test_img_inside_pre(self) -> None:
        md = _html_to_markdown('<pre><img src="x.png" alt="pic"></pre>')
        assert "![pic](x.png)" in md, f"Got: {md!r}"

    def test_backslash_in_link_text_escaped(self) -> None:
        md = _html_to_markdown('<a href="http://x.com">a\\b</a>')
        assert "a\\\\b" in md, f"Got: {md!r}"

    def test_backslash_in_href_escaped(self) -> None:
        md = _html_to_markdown('<a href="http://x.com/a\\b">link</a>')
        assert "\\\\" in md, f"Got: {md!r}"

    def test_tab_in_url_neutralized(self) -> None:
        """Tabs in safe URLs are replaced with spaces to prevent scheme obfuscation."""
        md = _html_to_markdown('<a href="http://x.com\tpath">link</a>')
        assert "\t" not in md, f"Tab leaked into output: {md!r}"
        assert "http://x.com" in md, f"Got: {md!r}"

    def test_cr_in_url_neutralized(self) -> None:
        """Carriage returns in safe URLs are replaced with spaces."""
        md = _html_to_markdown('<a href="http://x.com\rpath">link</a>')
        assert "\r" not in md, f"CR leaked into output: {md!r}"
        assert "http://x.com" in md, f"Got: {md!r}"

    def test_control_char_in_scheme_stripped(self) -> None:
        """Control characters in the scheme are stripped before scheme validation."""
        md = _html_to_markdown('<a href="java\x01script:alert(1)">click</a>')
        assert md == "click", f"Got: {md!r}"
        assert "javascript" not in md
        assert "alert" not in md

    def test_tab_obfuscated_javascript_scheme_stripped(self) -> None:
        """Tab inside 'javascript' scheme is stripped, scheme is still detected."""
        md = _html_to_markdown('<a href="java\tscript:alert(1)">click</a>')
        assert md == "click", f"Got: {md!r}"
        assert "javascript" not in md
        assert "alert" not in md

    def test_null_byte_obfuscated_scheme_stripped(self) -> None:
        """Null bytes in the scheme are stripped before validation."""
        md = _html_to_markdown('<a href="java\x00script:alert(1)">click</a>')
        assert md == "click", f"Got: {md!r}"
        assert "javascript" not in md

    def test_del_in_url_stripped(self) -> None:
        """DEL (0x7f) in the scheme is stripped before validation."""
        md = _html_to_markdown('<a href="java\x7fscript:alert(1)">click</a>')
        assert md == "click", f"Got: {md!r}"
        assert "javascript" not in md

    def test_multiple_whitespace_in_url_collapsed(self) -> None:
        """Multiple whitespace chars in a URL are collapsed to a single space."""
        md = _html_to_markdown('<a href="http://x.com\n\r\t path">link</a>')
        assert "\n" not in md, f"Got: {md!r}"
        assert "\r" not in md, f"Got: {md!r}"
        assert "\t" not in md, f"Got: {md!r}"
        assert "http://x.com" in md, f"Got: {md!r}"

    def test_img_src_tab_neutralized(self) -> None:
        """Tabs in image src URLs are neutralized."""
        md = _html_to_markdown('<img src="http://x.com\tpath" alt="pic">')
        assert "\t" not in md, f"Got: {md!r}"
        assert "http://x.com" in md, f"Got: {md!r}"


class TestWebFetchPrompt:
    def test_prompt_returns_valid_json(self) -> None:
        tool = WebFetch(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert data["name"] == "WebFetch"
        assert "description" in data
        assert "parameters" in data

    def test_prompt_has_required_fields(self) -> None:
        tool = WebFetch(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        assert "url" in data["parameters"]["required"]
        assert data["parameters"]["properties"]["url"]["type"] == "string"

    def test_prompt_has_optional_fields(self) -> None:
        tool = WebFetch(_TEST_SOCKET)
        prompt = tool.prompt()
        data = json.loads(prompt)
        props = data["parameters"]["properties"]
        assert "method" in props
        assert "headers" in props
        assert "body" in props
        assert "timeout" in props
        assert "max_redirects" in props
        assert "raw" in props
        assert "to_markdown" in props


class TestWebFetchFname:
    def test_fname(self) -> None:
        assert WebFetch.fname() == "WebFetch"


class TestWebFetchInit:
    def test_default_user_agent(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        assert tool.user_agent is not None
        assert "tiz" in tool.user_agent

    def test_custom_user_agent(self, socket_path: str) -> None:
        tool = WebFetch(socket_path, user_agent="CustomAgent/1.0")
        assert tool.user_agent == "CustomAgent/1.0"

    def test_none_user_agent_uses_default(self, socket_path: str) -> None:
        tool = WebFetch(socket_path, user_agent=None)
        assert tool.user_agent is not None
        assert "tiz" in tool.user_agent


class TestWebFetchRunValidation:
    def test_missing_url(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({})
        assert result == "ERROR: WebFetch takes a mandatory 'url' argument!"

    def test_empty_url(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": ""})
        assert result == "ERROR: WebFetch takes a mandatory 'url' argument!"

    def test_url_not_string(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": 123})
        assert result == "ERROR: url must start with http:// or https://"

    def test_url_no_scheme(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "example.com"})
        assert result == "ERROR: url must start with http:// or https://"

    def test_url_ftp_scheme(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "ftp://example.com"})
        assert result == "ERROR: url must start with http:// or https://"

    def test_url_http(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"url": "http://example.com"})
        assert result == "ok"
        mock_call.assert_called_once()

    def test_url_https(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"url": "https://example.com"})
        assert result == "ok"
        mock_call.assert_called_once()

    def test_invalid_timeout_type(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "timeout": "30"})
        assert result == "ERROR: timeout must be an integer between 1 and 120"

    def test_timeout_too_low(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "timeout": 0})
        assert result == "ERROR: timeout must be an integer between 1 and 120"

    def test_timeout_too_high(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "timeout": 121})
        assert result == "ERROR: timeout must be an integer between 1 and 120"

    def test_negative_timeout(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "timeout": -5})
        assert result == "ERROR: timeout must be an integer between 1 and 120"

    def test_valid_timeout(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"url": "https://example.com", "timeout": 30})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["timeout"] == 30

    def test_invalid_max_redirects_type(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "max_redirects": "5"})
        assert result == "ERROR: max_redirects must be an integer between 0 and 20"

    def test_max_redirects_too_low(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "max_redirects": -1})
        assert result == "ERROR: max_redirects must be an integer between 0 and 20"

    def test_max_redirects_too_high(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "max_redirects": 21})
        assert result == "ERROR: max_redirects must be an integer between 0 and 20"

    def test_negative_max_redirects(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "max_redirects": -10})
        assert result == "ERROR: max_redirects must be an integer between 0 and 20"

    def test_valid_max_redirects(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"url": "https://example.com", "max_redirects": 10})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["max_redirects"] == 10

    def test_user_agent_injected(self, socket_path: str) -> None:
        tool = WebFetch(socket_path, user_agent="TestAgent/1.0")
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"url": "https://example.com"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["user_agent"] == "TestAgent/1.0"

    def test_user_agent_not_overridden(self, socket_path: str) -> None:
        tool = WebFetch(socket_path, user_agent="DefaultAgent/1.0")
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run(
                {"url": "https://example.com", "user_agent": "CustomAgent/1.0"}
            )
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["user_agent"] == "CustomAgent/1.0"

    def test_name_removed(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        args = {"url": "https://example.com", "name": "WebFetch"}
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            tool.run(args)
        # Original args should not be mutated since we copy internally
        assert "name" in args
        call_args = mock_call.call_args[0][0]
        assert "name" not in call_args

    def test_description_is_removed_from_call_args(self, socket_path: str) -> None:
        """WebFetch now strips description consistently with other tools."""
        tool = WebFetch(socket_path)
        args = {"url": "https://example.com", "description": "some desc"}
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run(args)
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert "description" not in call_args

    def test_raw_flag_passed(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"url": "https://example.com", "raw": True})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["raw"] is True

    def test_body_passed(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run(
                {
                    "url": "https://example.com",
                    "method": "POST",
                    "body": '{"key": "value"}',
                }
            )
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["body"] == '{"key": "value"}'
        assert call_args["method"] == "POST"

    def test_invalid_method(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "method": "INVALID"})
        assert "ERROR: method must be one of" in result
        assert "GET" in result
        assert "POST" in result

    def test_method_case_insensitive(self, socket_path: str) -> None:
        """Bug 7: Method should be case-insensitive."""
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"url": "https://example.com", "method": "get"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["method"] == "GET"

    def test_method_mixed_case(self, socket_path: str) -> None:
        """Bug 7: Mixed case method should work."""
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run({"url": "https://example.com", "method": "Post"})
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["method"] == "POST"

    def test_method_not_string(self, socket_path: str) -> None:
        """Bug 7: Non-string method should be rejected."""
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "method": 123})
        assert "ERROR: method must be one of" in result

    def test_headers_passed(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="ok") as mock_call:
            result = tool.run(
                {
                    "url": "https://example.com",
                    "headers": {"Authorization": "Bearer token"},
                }
            )
        assert result == "ok"
        call_args = mock_call.call_args[0][0]
        assert call_args["headers"] == {"Authorization": "Bearer token"}

    def test_calls_call_on_valid_input(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        with patch.object(
            WebFetch, "_call", return_value='{"result": "ok"}'
        ) as mock_call:
            result = tool.run({"url": "https://example.com"})
        assert result == '{"result": "ok"}'
        mock_call.assert_called_once()

    def test_to_markdown_invalid_type(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "to_markdown": "yes"})
        assert result == "ERROR: to_markdown must be a boolean"

    def test_to_markdown_false_no_conversion(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        response = json.dumps(
            {"url": "https://x.com", "status": 200, "body": "<h1>Hi</h1>"}
        )
        with patch.object(WebFetch, "_call", return_value=response) as mock_call:
            result = tool.run({"url": "https://example.com", "to_markdown": False})
        assert result == response
        call_args = mock_call.call_args[0][0]
        assert "to_markdown" not in call_args

    def test_to_markdown_true_converts_html(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        response = json.dumps(
            {"url": "https://x.com", "status": 200, "body": "<h1>Hi</h1>"}
        )
        with patch.object(WebFetch, "_call", return_value=response):
            result = tool.run({"url": "https://example.com", "to_markdown": True})
        data = json.loads(result)
        assert data["body"] == "# Hi"

    def test_to_markdown_true_table_and_link(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        html_body = (
            "<table><tr><th>Key</th><th>Val</th></tr>"
            "<tr><td>a</td><td>1</td></tr></table>"
            '<a href="https://x.com">click</a>'
        )
        response = json.dumps(
            {"url": "https://x.com", "status": 200, "body": html_body}
        )
        with patch.object(WebFetch, "_call", return_value=response):
            result = tool.run({"url": "https://example.com", "to_markdown": True})
        data = json.loads(result)
        assert "| Key | Val |" in data["body"]
        assert "| a   | 1   |" in data["body"]
        assert "[click](https://x.com)" in data["body"]

    def test_to_markdown_true_non_json_result(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="plain text"):
            result = tool.run({"url": "https://example.com", "to_markdown": True})
        assert result == "plain text"

    def test_to_markdown_true_json_no_body(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        response = json.dumps({"status": 200, "headers": {}})
        with patch.object(WebFetch, "_call", return_value=response):
            result = tool.run({"url": "https://example.com", "to_markdown": True})
        assert result == response

    def test_to_markdown_true_error_result(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        with patch.object(WebFetch, "_call", return_value="ERROR: something"):
            result = tool.run({"url": "https://example.com", "to_markdown": True})
        assert result == "ERROR: something"

    def test_to_markdown_true_with_truncated_flag(self, socket_path: str) -> None:
        tool = WebFetch(socket_path)
        response = json.dumps(
            {
                "url": "https://x.com",
                "status": 200,
                "body": "<p>text</p>",
                "truncated": True,
            }
        )
        with patch.object(WebFetch, "_call", return_value=response):
            result = tool.run({"url": "https://example.com", "to_markdown": True})
        data = json.loads(result)
        assert data["body"] == "text"
        assert data["truncated"] is True

    def test_to_markdown_does_not_mutate_args(self, socket_path: str) -> None:
        """Bug 6: Calling run() must not mutate the caller's args dict."""
        tool = WebFetch(socket_path)
        args: dict = {"url": "https://example.com", "to_markdown": True}
        args_copy = dict(args)
        with patch.object(WebFetch, "_call", return_value='{"body": "<p>hi</p>"}'):
            tool.run(args)
        assert args == args_copy, f"Args were mutated: {args} != {args_copy}"

    def test_bool_timeout_rejected(self, socket_path: str) -> None:
        """Boolean values must not pass as valid timeout."""
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "timeout": True})
        assert result == "ERROR: timeout must be an integer between 1 and 120"

    def test_bool_max_redirects_rejected(self, socket_path: str) -> None:
        """Boolean values must not pass as valid max_redirects."""
        tool = WebFetch(socket_path)
        result = tool.run({"url": "https://example.com", "max_redirects": True})
        assert result == "ERROR: max_redirects must be an integer between 0 and 20"


class TestWebFetchFormatConfirmation:
    def test_format_confirmation_get_default(self) -> None:
        tool = WebFetch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"url": "https://example.com"}, markdown=False
        )
        assert result == "Fetch GET: https://example.com"

    def test_format_confirmation_post(self) -> None:
        tool = WebFetch("/tmp/test.sock")
        result = tool.format_confirmation(
            {"url": "https://api.example.com", "method": "POST"}, markdown=False
        )
        assert result == "Fetch POST: https://api.example.com"

    def test_format_confirmation_markdown(self) -> None:
        tool = WebFetch("/tmp/test.sock")
        result = tool.format_confirmation({"url": "https://example.com"}, markdown=True)
        assert result == "Fetch `GET`: `https://example.com`"

    def test_format_confirmation_empty_url(self) -> None:
        tool = WebFetch("/tmp/test.sock")
        result = tool.format_confirmation({}, markdown=False)
        assert result == "Fetch GET: "
