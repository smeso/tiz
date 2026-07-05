"""Web fetch tool for making HTTP requests."""

import json
import re
from html.parser import HTMLParser
from typing import Any

from tiz.sandbox_worker import DEFAULT_USER_AGENT
from tiz.tools.base import SocketTool


class _HTMLToMarkdown(HTMLParser):
    """Convert HTML to Markdown preserving tables, links, and formatting."""

    _SKIP = frozenset({"head", "script", "style", "title", "noscript", "iframe"})
    _VOID = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )
    _BLOCK_END_TAGS = frozenset(
        {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "div",
            "section",
            "article",
            "header",
            "footer",
            "main",
            "nav",
            "aside",
        }
    )
    _SAFE_URL_SCHEMES = frozenset({"http", "https", "ftp", "mailto", "tel", "sms"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self._skip_level = 0
        # Table state
        self._table_rows: list[list[str]] = []
        self._current_cells: list[str] = []
        self._in_cell = False
        self._in_table = False
        self._table_stack: list[tuple[list[list[str]], list[str], bool, int]] = []
        # Link state
        self._in_link = False
        self._link_href = ""
        self._link_text: list[str] = []
        self._link_stack: list[tuple[str, list[str]]] = []
        # Code state
        self._in_code = False
        self._code_content: list[str] = []
        # Pre state
        self._in_pre = False
        self._pre_content: list[str] = []
        self._pre_stack: list[list[str]] = []
        # List state
        self._list_kind: list[str] = []
        self._list_num: list[int] = []
        # Blockquote state
        self._blockquote_level = 0
        self._blockquote_first_para = True

    def _emit_cell(self, s: str) -> None:
        """Emit to the current cell buffer when inside a table cell."""
        if self._current_cells:  # pragma: no cover
            self._current_cells[-1] += s

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Return *url* only if it uses a safe scheme, otherwise ''.

        Relative URLs (no scheme) and URLs with one of the safe schemes
        are kept.  Dangerous schemes like ``javascript:`` or ``data:``
        are neutralised.

        Control characters (which browsers strip before interpreting a URL)
        are removed so that obfuscated schemes like ``java\\tscript:`` cannot
        survive the scheme check.
        """
        cleaned = re.sub(r"[\x00-\x20\x7f]", "", url)
        stripped = cleaned.strip()
        if ":" not in stripped:
            return url
        scheme = stripped.split(":", 1)[0].lower()
        if scheme in _HTMLToMarkdown._SAFE_URL_SCHEMES:
            return url
        return ""

    @staticmethod
    def _escape_md_text(text: str) -> str:
        """Escape characters that could form unintended Markdown constructs."""
        return (
            text.replace("\\", "\\\\")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace("!", "\\!")
        )

    @staticmethod
    def _escape_md_url(url: str) -> str:
        """Escape characters that could break out of a Markdown link URL.

        All whitespace and control characters are replaced with spaces so
        that the URL stays on a single line and cannot be split by a
        Markdown renderer.
        """
        return (
            re.sub(r"[\x00-\x20]+", " ", url)
            .replace("\\", "\\\\")
            .replace(")", "\\)")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .strip()
        )

    def _emit(self, s: str) -> None:
        """Emit to the main output, or to the cell buffer if inside a cell."""
        if self._in_cell:
            self._emit_cell(s)
            return
        self.output.append(s)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = {k: v or "" for k, v in attrs}
        if self._skip_level:
            if tag not in self._VOID and tag in self._SKIP:
                self._skip_level += 1
            return
        if tag in self._SKIP:
            self._skip_level = 1
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._emit("\n" + "#" * int(tag[1]) + " ")
        elif tag == "p":
            if self._blockquote_level:
                if not self._blockquote_first_para:
                    prefix = ">" * self._blockquote_level
                    self._emit(f"\n{prefix} ")
                self._blockquote_first_para = False
            else:
                self._emit("\n\n")
        elif tag == "br":
            if self._in_pre:
                self._pre_content.append("\n")
            elif self._in_link:
                self._link_text.append("\n")
            else:
                self._emit("\n")
        elif tag == "hr":
            if self._in_pre:
                self._pre_content.append("\n\n---\n")
            else:
                self._emit("\n\n---\n")
        elif tag == "pre":
            if self._in_pre:
                self._pre_stack.append(self._pre_content)
            self._in_pre = True
            self._pre_content = []
        elif tag == "code":
            if not self._in_pre:
                self._in_code = True
                self._code_content = []
        elif tag in ("b", "strong"):
            if self._in_pre:
                self._pre_content.append("**")
            elif self._in_link:
                self._link_text.append("**")
            else:
                self._emit("**")
        elif tag in ("i", "em"):
            if self._in_pre:
                self._pre_content.append("*")
            elif self._in_link:
                self._link_text.append("*")
            else:
                self._emit("*")
        elif tag == "a":
            if self._in_link:
                self._link_stack.append((self._link_href, self._link_text))
            self._in_link = True
            self._link_href = self._sanitize_url(attrs_d.get("href", ""))
            self._link_text = []
        elif tag == "img":
            alt = self._escape_md_text(attrs_d.get("alt", ""))
            src = self._sanitize_url(attrs_d.get("src", ""))
            img_md = alt if not src else f"![{alt}]({self._escape_md_url(src)})"
            if self._in_pre:
                self._pre_content.append(img_md)
            elif self._in_link:
                self._link_text.append(img_md)
            else:
                self._emit(img_md)
        elif tag == "table":
            if self._in_table:
                self._table_stack.append(
                    (
                        self._table_rows,
                        self._current_cells,
                        self._in_cell,
                        len(self.output),
                    )
                )
            self._in_table = True
            self._table_rows = []
            self._current_cells = []
            self._in_cell = False
        elif tag == "tr":
            if self._current_cells:
                self._table_rows.append([c.strip() for c in self._current_cells])
            self._current_cells = []
        elif tag in ("th", "td"):
            self._in_cell = True
            self._current_cells.append("")
        elif tag in ("ul", "ol"):
            self._list_kind.append(tag)
            if tag == "ol":
                start = attrs_d.get("start", "")
                self._list_num.append(int(start) - 1 if start.isdigit() else 0)
            else:
                self._list_num.append(0)
        elif tag == "li":
            if not self._list_kind:
                return
            indent = "  " * (len(self._list_kind) - 1)
            self._emit("\n" + indent)
            if self._list_kind[-1] == "ol":
                self._list_num[-1] += 1
                self._emit(f"{self._list_num[-1]}. ")
            else:
                self._emit("- ")
        elif tag == "blockquote":
            self._blockquote_level += 1
            self._blockquote_first_para = True
            prefix = ">" * self._blockquote_level
            self._emit(f"\n\n{prefix} ")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_level:
            if tag in self._SKIP:
                self._skip_level -= 1
            return

        if tag in self._BLOCK_END_TAGS:
            self._emit("\n\n")
        elif tag == "pre":
            self._in_pre = False
            content = "".join(self._pre_content)
            max_backticks = 0
            count = 0
            for ch in content:
                if ch == "`":
                    count += 1
                    max_backticks = max(max_backticks, count)
                else:
                    count = 0
            delimiter = "`" * max(3, max_backticks + 1)
            rendered = f"\n{delimiter}\n{content}\n{delimiter}"
            if self._pre_stack:
                self._pre_content = self._pre_stack.pop()
                self._pre_content.append(content)
                self._in_pre = True
            else:
                self._emit(rendered)
        elif tag == "code" and not self._in_pre and self._in_code:
            self._in_code = False
            content = "".join(self._code_content)
            max_backticks = 0
            count = 0
            for ch in content:
                if ch == "`":
                    count += 1
                    max_backticks = max(max_backticks, count)
                else:
                    count = 0
            delimiter = "`" * (max_backticks + 1)
            marker = f"{delimiter}{content}{delimiter}"
            if self._in_link:
                self._link_text.append(marker)
            else:
                self._emit(marker)
        elif tag in ("b", "strong"):
            if self._in_pre:
                self._pre_content.append("**")
            elif self._in_link:
                self._link_text.append("**")
            else:
                self._emit("**")
        elif tag in ("i", "em"):
            if self._in_pre:
                self._pre_content.append("*")
            elif self._in_link:
                self._link_text.append("*")
            else:
                self._emit("*")
        elif tag == "a":
            if self._in_link:
                text = "".join(self._link_text).strip()
                href = self._escape_md_url(self._link_href)
                if href:
                    rendered = f"[{text}]({href})"
                elif text:
                    rendered = text
                else:
                    rendered = ""
                if self._link_stack:
                    outer_href, outer_text = self._link_stack.pop()
                    self._link_href = outer_href
                    self._link_text = outer_text
                    if rendered:
                        self._link_text.append(rendered)
                else:
                    self._in_link = False
                    self._link_href = ""
                    self._link_text = []
                    if rendered:
                        if self._in_pre:
                            self._pre_content.append(rendered)
                        else:
                            self._emit(rendered)
        elif tag == "table":
            if self._current_cells:
                self._table_rows.append([c.strip() for c in self._current_cells])
                self._current_cells = []
            self._render_table()
            if self._table_stack:
                rows, cells, in_cell, output_len = self._table_stack.pop()
                inner_output = self.output[output_len:]
                self.output = self.output[:output_len]
                self._table_rows = rows
                self._current_cells = cells
                if inner_output:
                    inner_text = "".join(inner_output)
                    if in_cell:
                        self._current_cells[-1] += inner_text
                    else:
                        self.output.append(inner_text)
                self._in_cell = in_cell
            else:
                self._in_table = False
        elif tag == "tr":
            if self._current_cells:
                self._table_rows.append([c.strip() for c in self._current_cells])
            self._current_cells = []
        elif tag in ("th", "td"):
            self._in_cell = False
        elif tag in ("ul", "ol"):
            if self._list_kind:
                self._list_kind.pop()
                self._list_num.pop()
            self._emit("\n")
        elif tag == "blockquote" and self._blockquote_level > 0:
            self._blockquote_level -= 1
            self._emit("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_level:
            return
        if self._in_code:
            self._code_content.append(data)
            return
        if self._in_link:
            self._link_text.append(self._escape_md_text(data))
            return
        if self._in_pre:
            self._pre_content.append(data)
            return
        if self._in_cell:
            self._current_cells[-1] += re.sub(r"\s+", " ", data).replace("|", "\\|")
            return
        text = re.sub(r"\s+", " ", data)
        self._emit(text)

    def _render_table(self) -> None:
        if not self._table_rows:
            return
        max_cols = max((len(r) for r in self._table_rows), default=0)
        if max_cols == 0:  # pragma: no cover
            return
        for row in self._table_rows:
            while len(row) < max_cols:
                row.append("")
        widths = [0] * max_cols
        for row in self._table_rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        for i, row in enumerate(self._table_rows):
            cells = [row[j].ljust(widths[j]) for j in range(max_cols)]
            self._emit("\n| " + " | ".join(cells) + " |")
            if i == 0:
                seps = ["-" * widths[j] for j in range(max_cols)]
                self._emit("\n| " + " | ".join(seps) + " |")
        self._emit("\n\n")

    def get_markdown(self) -> str:
        md = "".join(self.output)
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()


def _html_to_markdown(html_text: str) -> str:
    parser = _HTMLToMarkdown()
    parser.feed(html_text)
    parser.close()
    return parser.get_markdown()


class WebFetch(SocketTool):
    """Tool to fetch URLs via HTTP/HTTPS."""

    def __init__(self, socket_path: str, user_agent: str | None = None) -> None:
        super().__init__(socket_path)
        self.user_agent = user_agent or DEFAULT_USER_AGENT

    def format_confirmation(self, args: dict[str, Any], markdown: bool = False) -> str:
        url = args.get("url", "")
        method = args.get("method", "GET")
        if markdown:
            return f"Fetch `{self._safe_md(method)}`: `{self._safe_md(url)}`"
        return f"Fetch {method}: {url}"

    def prompt(self) -> str:
        return json.dumps(
            {
                "name": "WebFetch",
                "description": "Fetch a URL via HTTP/HTTPS. Returns the response status, headers, and body content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch (required). Must start with http:// or https://",
                        },
                        "method": {
                            "type": "string",
                            "description": "HTTP method to use (default: GET). Supported: GET, POST, PUT, DELETE, HEAD, PATCH",
                            "enum": ["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"],
                        },
                        "headers": {
                            "type": "object",
                            "description": "Additional HTTP headers as key-value pairs",
                            "additionalProperties": {"type": "string"},
                        },
                        "body": {
                            "type": "string",
                            "description": "Request body for POST/PUT/PATCH requests (e.g., JSON string, form data)",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Request timeout in seconds (1-120, default 30)",
                        },
                        "max_redirects": {
                            "type": "integer",
                            "description": "Maximum number of redirects to follow (0-20, default 5)",
                        },
                        "raw": {
                            "type": "boolean",
                            "description": "If true, return raw content without truncation. Default: false (truncates at 100KB)",
                        },
                        "to_markdown": {
                            "type": "boolean",
                            "description": "If true, convert the HTML response body to Markdown preserving tables and links. Default: false",
                        },
                    },
                    "required": ["url"],
                },
            }
        )

    @staticmethod
    def fname() -> str:
        return "WebFetch"

    def run(self, args: dict[str, Any]) -> str:
        args = dict(args)  # copy to avoid mutating caller's dict
        if "url" not in args or not args["url"]:
            return f"ERROR: {self.fname()} takes a mandatory 'url' argument!"
        url = args["url"]
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return "ERROR: url must start with http:// or https://"
        allowed_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"}
        method = args.get("method", "GET")
        if not isinstance(method, str):
            return f"ERROR: method must be one of {', '.join(sorted(allowed_methods))}"
        method = method.upper()
        args["method"] = method
        if method not in allowed_methods:
            return f"ERROR: method must be one of {', '.join(sorted(allowed_methods))}"
        timeout = args.get("timeout", 30)
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or timeout < 1
            or timeout > 120
        ):
            return "ERROR: timeout must be an integer between 1 and 120"
        max_redirects = args.get("max_redirects", 5)
        if (
            not isinstance(max_redirects, int)
            or isinstance(max_redirects, bool)
            or max_redirects < 0
            or max_redirects > 20
        ):
            return "ERROR: max_redirects must be an integer between 0 and 20"
        to_markdown = args.pop("to_markdown", False)
        if not isinstance(to_markdown, bool):
            return "ERROR: to_markdown must be a boolean"
        args.setdefault("user_agent", self.user_agent)
        if "name" in args:
            del args["name"]
        result = self._call(args)
        if to_markdown and isinstance(result, str) and not result.startswith("ERROR:"):
            try:
                data = json.loads(result)
                if isinstance(data, dict) and isinstance(data.get("body"), str):
                    data["body"] = _html_to_markdown(data["body"])
                    result = json.dumps(data, indent=2)
            except (json.JSONDecodeError, TypeError):
                pass
        return result
