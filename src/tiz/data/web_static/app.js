/* ========== BASE PATH (where index.html lives) ========== */
const BASE = (function () {
    const scripts = document.getElementsByTagName('script');
    for (let i = 0; i < scripts.length; i++) {
        const src = scripts[i].src;
        const idx = src.lastIndexOf('/app.js');
        if (idx !== -1) {
            const url = new URL(
                src.substring(0, idx + 1),
                window.location.href,
            );
            return url.pathname;
        }
    }
    const path = window.location.pathname;
    return path.substring(0, path.lastIndexOf('/') + 1);
})();

/* ========== STATE ========== */
let ws = null;
let isWaiting = false;
let streamActive = false;
let streamingContent = '';
let streamingReasoning = '';
let currentContentEl = null;
let currentReasoningEl = null;

let streamLastActivity = 0;
let reconnectTimer = null;
let reconnectAttempts = 0;
let manualDisconnect = false;
let firstConnect = true;

let endpoints = [];
let currentEndpoint = null;

/* ========== AUTOSAVE STATE (in-memory for reconnect) ========== */
let lastSavedData = null;
let lastSavedMsgCount = 0;
let pendingRestoreMsgs = null;
let restoringAfterReconnect = false;

/* ========== DOM REFS ========== */
const container = document.getElementById('chat-container');
const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const themeToggle = document.getElementById('theme-toggle');
const typingIndicator = document.getElementById('typing-indicator');
const scrollAnchor = document.getElementById('scroll-anchor');
const reconnectBanner = document.getElementById('reconnect-banner');
const reconnectNow = document.getElementById('reconnect-now');
const connectionStatus = document.getElementById('connection-status');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

/* ========== ATTACHMENT DOM REFS ========== */
const attachBtn = document.getElementById('attach-btn');
const fileInput = document.getElementById('file-input');
const cameraInput = document.getElementById('camera-input');
const cameraBtn = document.getElementById('camera-btn');
const audioBtn = document.getElementById('audio-btn');
const filePreviews = document.getElementById('file-previews');

const usagePanel = document.getElementById('usage-panel');
const usagePanelToggle = document.getElementById('usage-panel-toggle');
const usagePanelContent = document.getElementById('usage-panel-content');
const progressEl = document.getElementById('progress-bar');

const attachedFiles = [];
const objectUrls = [];

/* ========== CHAT SELECTOR ========== */
const chatSelectorBtn = document.getElementById('chat-selector-btn');
const chatSelectorLabel = document.getElementById('chat-selector-label');
const chatSelectorMenu = document.getElementById('chat-selector-menu');
const chatSelectorOptions = document.getElementById('chat-selector-options');

/* ========== THEME ========== */
function getPreferredTheme() {
    const stored = localStorage.getItem('theme');
    if (stored) {
        return stored;
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    // Remove media-qualified theme-color meta tags and create a single
    // overriding bare tag so manual toggles take effect regardless of
    // OS-level prefers-color-scheme.
    document
        .querySelectorAll('meta[name="theme-color"]')
        .forEach(function (el) {
            el.remove();
        });
    const meta = document.createElement('meta');
    meta.name = 'theme-color';
    meta.content = theme === 'dark' ? '#0a0a0f' : '#f5f5f7';
    document.head.appendChild(meta);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    setTheme(current === 'dark' ? 'light' : 'dark');
}

setTheme(getPreferredTheme());
themeToggle.addEventListener('click', toggleTheme);

/* ========== CLEAR CHAT ========== */
const clearBtn = document.getElementById('clear-btn');

/* ========== RETRY (send /replay) ========== */
const retryBtn = document.getElementById('retry-btn');

function resetChatState() {
    const oldMsgs = messagesEl.querySelectorAll('.message');
    for (let i = 0; i < oldMsgs.length; i++) {
        revokeMsgObjectUrls(oldMsgs[i]);
    }
    messagesEl.innerHTML = '';
    attachedFiles.length = 0;
    revokeObjectUrls();
    renderFilePreviews();
    if (streamActive) {
        finishStreaming();
    }
    isWaiting = false;
    streamActive = false;
    streamingContent = '';
    streamingReasoning = '';
    currentContentEl = null;
    currentReasoningEl = null;
    hideReconnectBanner();
    hideProgress();
    hideTyping();
    usagePanel.classList.add('hidden');
    lastSavedData = null;
    lastSavedMsgCount = 0;
    pendingRestoreMsgs = null;
    restoringAfterReconnect = false;
}

function clearChat() {
    manualDisconnect = true;
    if (ws) {
        ws.onclose = null;
        ws.close();
        ws = null;
    }
    resetChatState();
    welcomeEl.classList.remove('hidden');
    manualDisconnect = false;
    updateSendButton();
    connectWebSocket();
}

clearBtn.addEventListener('click', clearChat);

/* ========== RETRY BUTTON ========== */
retryBtn.addEventListener('click', sendReplay);

/* ========== AUTO-RESIZE TEXTAREA ========== */
function autoResize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
}
input.addEventListener('input', autoResize);

/* ========== SEND STATE ========== */
function updateSendButton() {
    if (isWaiting) {
        sendBtn.disabled = false;
        sendBtn.classList.add('streaming');
        showTyping();
    } else {
        const hasContent = input.value.trim() || attachedFiles.length > 0;
        sendBtn.disabled =
            !hasContent || !ws || ws.readyState !== WebSocket.OPEN;
        sendBtn.classList.remove('streaming');
        hideTyping();
    }
}

/* ========== SCROLL ========== */
let userScrolledUp = false;

container.addEventListener('scroll', function () {
    const threshold = 80;
    const atBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight <
        threshold;
    userScrolledUp = !atBottom;
});

function scrollToBottom(instant) {
    if (userScrolledUp) {
        return;
    }
    scrollAnchor.scrollIntoView(
        instant ? { block: 'end' } : { behavior: 'smooth', block: 'end' },
    );
}

/* ========== PLACEHOLDER MARKERS ========== */
const CB_MARKER = '\x00cb\x00';
const SP_MARKER = '\x00sp\x00';

/* ========== RENDER (minimal markdown, no deps) ========== */
function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;');
}

function escapeAttr(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;');
}

function isSafeUrl(url) {
    const safeSchemes = ['http:', 'https:', 'mailto:'];
    try {
        const parsed = new URL(url, window.location.href);
        return safeSchemes.includes(parsed.protocol);
    } catch (e) {
        return false;
    }
}

/* ========== SYNTAX HIGHLIGHTING ========== */

const HIGHLIGHT = {
    python: {
        keywords: [
            'False',
            'None',
            'True',
            'and',
            'as',
            'assert',
            'async',
            'await',
            'break',
            'class',
            'continue',
            'def',
            'del',
            'elif',
            'else',
            'except',
            'finally',
            'for',
            'from',
            'global',
            'if',
            'import',
            'in',
            'is',
            'lambda',
            'nonlocal',
            'not',
            'or',
            'pass',
            'raise',
            'return',
            'try',
            'while',
            'with',
            'yield',
        ].join('|'),
        builtins: [
            'print',
            'len',
            'range',
            'int',
            'str',
            'float',
            'list',
            'dict',
            'set',
            'tuple',
            'bool',
            'type',
            'super',
            'open',
            'input',
            'zip',
            'map',
            'filter',
            'sorted',
            'reversed',
            'enumerate',
            'isinstance',
            'hasattr',
            'getattr',
            'setattr',
            'abs',
            'max',
            'min',
            'sum',
            'any',
            'all',
            'hex',
            'oct',
            'bin',
            'ord',
            'chr',
            'repr',
            'staticmethod',
            'classmethod',
            'property',
            '__init__',
            '__str__',
            '__repr__',
            '__call__',
            '__len__',
            '__getitem__',
            '__setitem__',
        ].join('|'),
    },
    c: {
        keywords: [
            'auto',
            'break',
            'case',
            'char',
            'const',
            'continue',
            'default',
            'do',
            'double',
            'else',
            'enum',
            'extern',
            'float',
            'for',
            'goto',
            'if',
            'int',
            'long',
            'register',
            'return',
            'short',
            'signed',
            'sizeof',
            'static',
            'struct',
            'switch',
            'typedef',
            'union',
            'unsigned',
            'void',
            'volatile',
            'while',
            'include',
            'define',
            'ifdef',
            'ifndef',
            'endif',
            'undef',
            'pragma',
        ].join('|'),
        builtins: [
            'printf',
            'scanf',
            'malloc',
            'calloc',
            'realloc',
            'free',
            'fopen',
            'fclose',
            'fread',
            'fwrite',
            'fprintf',
            'fscanf',
            'fgets',
            'fputs',
            'fgetc',
            'fputc',
            'fseek',
            'ftell',
            'rewind',
            'feof',
            'ferror',
            'perror',
            'strlen',
            'strcpy',
            'strncpy',
            'strcat',
            'strncat',
            'strcmp',
            'strncmp',
            'strchr',
            'strrchr',
            'strstr',
            'strtok',
            'memset',
            'memcpy',
            'memmove',
            'memcmp',
            'atoi',
            'atol',
            'atof',
            'strtol',
            'strtod',
            'sprintf',
            'sscanf',
            'puts',
            'getchar',
            'putchar',
            'exit',
            'abs',
            'rand',
            'srand',
            'time',
            'qsort',
            'bsearch',
            'assert',
            'FILE',
            'NULL',
            'size_t',
            'int32_t',
            'uint32_t',
            'int64_t',
            'uint64_t',
        ].join('|'),
    },
    bash: {
        keywords: [
            'if',
            'then',
            'elif',
            'else',
            'fi',
            'for',
            'while',
            'until',
            'do',
            'done',
            'in',
            'case',
            'esac',
            'select',
            'function',
            'return',
            'exit',
            'break',
            'continue',
            'declare',
            'local',
            'export',
            'readonly',
            'unset',
            'eval',
            'exec',
            'shift',
            'source',
            'trap',
            'type',
            'typeset',
            'ulimit',
            'umask',
            'wait',
            'let',
            'test',
            'true',
            'false',
        ].join('|'),
        builtins: [
            'echo',
            'printf',
            'cd',
            'ls',
            'cat',
            'rm',
            'mv',
            'cp',
            'mkdir',
            'rmdir',
            'touch',
            'chmod',
            'chown',
            'grep',
            'egrep',
            'fgrep',
            'find',
            'awk',
            'sed',
            'sort',
            'uniq',
            'wc',
            'head',
            'tail',
            'cut',
            'tr',
            'tee',
            'diff',
            'patch',
            'tar',
            'gzip',
            'gunzip',
            'bzip2',
            'xz',
            'zip',
            'unzip',
            'make',
            'gcc',
            'clang',
            'python',
            'python3',
            'pip',
            'npm',
            'node',
            'git',
            'curl',
            'wget',
            'ssh',
            'scp',
            'rsync',
            'ps',
            'kill',
            'pkill',
            'jobs',
            'bg',
            'fg',
            'nohup',
            'time',
            'date',
            'basename',
            'dirname',
            'read',
            'sleep',
            'xargs',
            'env',
            'which',
            'whoami',
            'id',
            'hostname',
            'uname',
            'df',
            'du',
            'free',
            'top',
            'htop',
            'lsof',
            'mount',
            'umount',
            'systemctl',
            'journalctl',
            'service',
            'sudo',
            'su',
            'passwd',
            'useradd',
            'usermod',
            'groupadd',
            'apt',
            'apt-get',
            'yum',
            'dnf',
            'pacman',
            'brew',
            'docker',
            'kubectl',
            'aws',
            'gcloud',
            'az',
        ].join('|'),
    },
};

const HASH_COMMENT_LANGS = { python: true, bash: true };
const SLASH_COMMENT_LANGS = { c: true };
const DECORATOR_LANGS = { python: true };
const SHEBANG_LANGS = { bash: true };

const BASE_PATTERNS = [
    // Block comments (C)
    { regex: /(\/\*[\s\S]*?\*\/)/g, cls: 'hl-comment' },
    // Strings (double)
    { regex: /("(?:[^"\\]|\\.)*")/g, cls: 'hl-string' },
    // Strings (single)
    { regex: /('(?:[^'\\]|\\.)*')/g, cls: 'hl-string' },
    // Numbers
    { regex: /\b(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/g, cls: 'hl-number' },
];

function buildPatterns(langName) {
    const patterns = BASE_PATTERNS.slice();
    if (HASH_COMMENT_LANGS[langName]) {
        patterns.unshift({ regex: /(#.*$)/gm, cls: 'hl-comment' });
    }
    if (SLASH_COMMENT_LANGS[langName]) {
        patterns.unshift({ regex: /(\/\/.*$)/gm, cls: 'hl-comment' });
    }
    if (DECORATOR_LANGS[langName]) {
        patterns.push({ regex: /(@[\w.]+)/g, cls: 'hl-decorator' });
    }
    if (SHEBANG_LANGS[langName]) {
        patterns.push({ regex: /^(#!.*$)/gm, cls: 'hl-shebang' });
    }
    return patterns;
}

const PATTERNS_CACHE = {};

function getPatterns(langName) {
    if (!PATTERNS_CACHE[langName]) {
        PATTERNS_CACHE[langName] = buildPatterns(langName);
    }
    return PATTERNS_CACHE[langName];
}

const KW_REGEX_CACHE = {};
const BI_REGEX_CACHE = {};

function getKwRegex(lang) {
    const key = lang.keywords;
    if (!KW_REGEX_CACHE[key]) {
        KW_REGEX_CACHE[key] = new RegExp('\\b(' + lang.keywords + ')\\b', 'g');
    }
    return KW_REGEX_CACHE[key];
}

function getBiRegex(lang) {
    const key = lang.builtins;
    if (!BI_REGEX_CACHE[key]) {
        BI_REGEX_CACHE[key] = new RegExp('\\b(' + lang.builtins + ')\\b', 'g');
    }
    return BI_REGEX_CACHE[key];
}

const SP_RESTORE_RE = new RegExp(SP_MARKER + '(\\d+)' + SP_MARKER, 'g');

function highlightTokens(line, langName) {
    const lang = HIGHLIGHT[langName];
    if (!lang) {
        return escapeHtml(line);
    }

    const kwRegex = getKwRegex(lang);
    const biRegex = getBiRegex(lang);

    // Apply patterns first (comments, strings, numbers, decorators, shebang)
    // Replace spans with placeholders so keyword/builtin regexes don't match inside them
    const patterns = getPatterns(langName);
    let result = escapeHtml(line);

    const spans = [];
    for (let i = 0; i < patterns.length; i++) {
        result = result.replace(patterns[i].regex, function (match) {
            const idx = spans.length;
            spans.push(
                '<span class="' + patterns[i].cls + '">' + match + '</span>',
            );
            return SP_MARKER + idx + SP_MARKER;
        });
    }

    // Keywords (only matches outside pattern spans)
    result = result.replace(kwRegex, function (match) {
        return '<span class="hl-keyword">' + match + '</span>';
    });

    // Builtins (only matches outside pattern spans)
    result = result.replace(biRegex, function (match) {
        return '<span class="hl-builtin">' + match + '</span>';
    });

    // Restore pattern spans
    result = result.replace(SP_RESTORE_RE, function (_, idx) {
        return spans[parseInt(idx, 10)] || '';
    });

    return result;
}

function highlightCode(code, langName) {
    const lines = code.split('\n');
    const highlighted = [];
    let inMultilineComment = false;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();

        // Skip empty lines
        if (!trimmed) {
            highlighted.push('');
            continue;
        }

        // Handle multi-line comments for C
        if (langName === 'c') {
            if (inMultilineComment) {
                const endIdx = line.indexOf('*/');
                if (endIdx !== -1) {
                    highlighted.push(
                        '<span class="hl-comment">' +
                            escapeHtml(line.substring(0, endIdx + 2)) +
                            '</span>' +
                            highlightTokens(
                                line.substring(endIdx + 2),
                                langName,
                            ),
                    );
                    inMultilineComment = false;
                } else {
                    highlighted.push(
                        '<span class="hl-comment">' +
                            escapeHtml(line) +
                            '</span>',
                    );
                }
                continue;
            }
            const startIdx = line.indexOf('/*');
            if (startIdx !== -1) {
                const endIdx = line.indexOf('*/', startIdx + 2);
                if (endIdx !== -1) {
                    // Single line comment
                    highlighted.push(
                        highlightTokens(line.substring(0, startIdx), langName) +
                            '<span class="hl-comment">' +
                            escapeHtml(line.substring(startIdx, endIdx + 2)) +
                            '</span>' +
                            highlightTokens(
                                line.substring(endIdx + 2),
                                langName,
                            ),
                    );
                } else {
                    // Multi-line starts
                    highlighted.push(
                        highlightTokens(line.substring(0, startIdx), langName) +
                            '<span class="hl-comment">' +
                            escapeHtml(line.substring(startIdx)) +
                            '</span>',
                    );
                    inMultilineComment = true;
                }
                continue;
            }
        }

        const escaped = highlightTokens(line, langName);
        highlighted.push(escaped);
    }

    return highlighted.join('\n');
}

function renderContent(text) {
    if (!text) {
        return '';
    }

    // Process code blocks on the original text (before escapeHtml) to avoid
    // double-escaping. Replace them with placeholders, then restore later.
    const codeBlocks = [];
    const textWithoutCode = text.replace(
        /```(\w*)\n([\s\S]*?)```/g,
        function (_, lang, code) {
            const langName = lang.toLowerCase();
            const langClass = lang ? ' class="language-' + langName + '"' : '';
            const supported =
                langName === 'python' ||
                langName === 'c' ||
                langName === 'bash';
            const codeContent = code.trimEnd();
            const highlighted = supported
                ? highlightCode(codeContent, langName)
                : escapeHtml(codeContent);
            const idx = codeBlocks.length;
            codeBlocks.push(
                '<pre><code' + langClass + '>' + highlighted + '</code></pre>',
            );
            return CB_MARKER + idx + CB_MARKER;
        },
    );

    let html = escapeHtml(textWithoutCode);

    // Inline elements (order matters: code, images, links, bold, italic, del)
    // Run these before restoring code blocks so backticks inside code blocks
    // don't match the inline code regex
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function (_, alt, url) {
        if (!isSafeUrl(url)) {
            return '<img src="" alt="' + alt + '">';
        }
        const safeUrl = new URL(url, window.location.href).href;
        return '<img src="' + escapeAttr(safeUrl) + '" alt="' + alt + '">';
    });
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (_, text, url) {
        if (!isSafeUrl(url)) {
            return text;
        }
        const safeUrl = new URL(url, window.location.href).href;
        return (
            '<a href="' +
            escapeAttr(safeUrl) +
            '" target="_blank" rel="noopener noreferrer">' +
            text +
            '</a>'
        );
    });
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/~~(.+?)~~/g, '<del>$1</del>');

    // Restore code blocks (after inline elements, so backticks inside
    // <pre><code> don't match the inline code regex)
    const cbRe = new RegExp(CB_MARKER + '(\\d+)' + CB_MARKER, 'g');
    html = html.replace(cbRe, function (_, idx) {
        return codeBlocks[parseInt(idx, 10)] || '';
    });

    // Block-level elements
    const blocks = html.split(/\n\n+/);
    let result = '';
    for (let i = 0; i < blocks.length; i++) {
        const block = blocks[i].trim();
        if (!block) {
            continue;
        }

        const blockLines = block.split('\n');

        // Header (# to ######)
        const headerMatch = blockLines[0].match(/^(#{1,6})\s+(.+)$/);
        if (headerMatch) {
            const level = headerMatch[1].length;
            const headerContent = headerMatch[2];
            const rest = [];
            for (let k = 1; k < blockLines.length; k++) {
                if (blockLines[k].trim()) {
                    rest.push(blockLines[k]);
                }
            }
            result += '<h' + level + '>' + headerContent + '</h' + level + '>';
            if (rest.length) {
                result += '<p>' + rest.join('<br>') + '</p>';
            }
            continue;
        }

        // Horizontal rule
        if (
            blockLines.length === 1 &&
            /^[-*_]{3,}$/.test(blockLines[0].trim())
        ) {
            result += '<hr>';
            continue;
        }

        // Blockquote
        let allQuotes = true;
        for (let q = 0; q < blockLines.length; q++) {
            if (!blockLines[q].trim().startsWith('>')) {
                allQuotes = false;
                break;
            }
        }
        if (allQuotes) {
            const quoteLines = [];
            for (let q = 0; q < blockLines.length; q++) {
                quoteLines.push(blockLines[q].replace(/^>\s?/, ''));
            }
            result +=
                '<blockquote>' + quoteLines.join('<br>') + '</blockquote>';
            continue;
        }

        // Table
        let isTable = false;
        if (blockLines.length >= 2) {
            const sepLine = blockLines[1].trim();
            if (/^\|?[\s:-]+\|[\s:-]+(\|[\s:-]+)*\|?$/.test(sepLine)) {
                isTable = true;
            }
        }
        if (isTable) {
            const headerCells = blockLines[0]
                .split('|')
                .map(function (c) {
                    return c.trim();
                })
                .filter(function (c) {
                    return c !== '';
                });
            const rows = [];
            for (let t = 2; t < blockLines.length; t++) {
                const cells = blockLines[t]
                    .split('|')
                    .map(function (c) {
                        return c.trim();
                    })
                    .filter(function (c) {
                        return c !== '';
                    });
                if (cells.length > 0) {
                    rows.push(cells);
                }
            }
            let tableHtml = '<table><thead><tr>';
            for (let h = 0; h < headerCells.length; h++) {
                tableHtml += '<th>' + headerCells[h] + '</th>';
            }
            tableHtml += '</tr></thead><tbody>';
            for (let r = 0; r < rows.length; r++) {
                tableHtml += '<tr>';
                for (let c = 0; c < rows[r].length; c++) {
                    tableHtml += '<td>' + rows[r][c] + '</td>';
                }
                tableHtml += '</tr>';
            }
            tableHtml += '</tbody></table>';
            result += tableHtml;
            continue;
        }

        // Unordered list
        let allUl = true;
        for (let u = 0; u < blockLines.length; u++) {
            if (!/^\s*[-*+]\s/.test(blockLines[u])) {
                allUl = false;
                break;
            }
        }
        if (allUl) {
            let items = '';
            for (let u = 0; u < blockLines.length; u++) {
                items +=
                    '<li>' +
                    blockLines[u].replace(/^\s*[-*+]\s+(.*)$/, '$1') +
                    '</li>';
            }
            result += '<ul>' + items + '</ul>';
            continue;
        }

        // Ordered list
        let allOl = true;
        for (let o = 0; o < blockLines.length; o++) {
            if (!/^\s*\d+\.\s/.test(blockLines[o])) {
                allOl = false;
                break;
            }
        }
        if (allOl) {
            let items = '';
            for (let o = 0; o < blockLines.length; o++) {
                items +=
                    '<li>' +
                    blockLines[o].replace(/^\s*\d+\.\s+(.*)$/, '$1') +
                    '</li>';
            }
            result += '<ol>' + items + '</ol>';
            continue;
        }

        // Regular paragraph
        result += '<p>' + blockLines.join('<br>') + '</p>';
    }

    return result || html;
}

/* ========== ADD MESSAGE ========== */
function addMessage(role, content, isError, isStale) {
    if (isError === undefined) {
        isError = false;
    }
    if (isStale === undefined) {
        isStale = false;
    }
    welcomeEl.classList.add('hidden');

    const avatar = role === 'user' ? 'U' : 'T';
    const bubbleContent =
        role === 'user' ? escapeHtml(content) : renderContent(content);

    const div = document.createElement('div');
    div.className =
        'message ' +
        role +
        (isError ? ' error' : '') +
        (isStale ? ' stale' : '');
    div.innerHTML =
        '<div class="avatar">' +
        avatar +
        '</div><div class="bubble">' +
        bubbleContent +
        '</div>';
    if (role === 'bot' && !isError) {
        addRetryButton(div);
    }
    messagesEl.appendChild(div);
    scrollToBottom();
    return div;
}

/* ========== RETRY (send /replay) ========== */
function sendReplay() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        return;
    }
    ws.send(JSON.stringify({ command: '/replay', message: '' }));
    welcomeEl.classList.add('hidden');
}

/* ========== ADD RETRY BUTTON ========== */
function addRetryButton(msgEl) {
    const bubble = msgEl.querySelector('.bubble');
    if (!bubble) {
        return;
    }
    const btn = document.createElement('button');
    btn.className = 'retry-btn';
    btn.textContent = 'Retry';
    btn.setAttribute('aria-label', 'Retry with /replay');
    btn.addEventListener('click', function (e) {
        e.stopPropagation();
        sendReplay();
    });
    bubble.appendChild(btn);
}

/* ========== TRANSCRIPTION ========== */
function addTranscription(text) {
    // Remove the audio attachment message that triggered this transcription
    const lastUserMsg = messagesEl.querySelector('.message.user:last-of-type');
    if (lastUserMsg) {
        revokeMsgObjectUrls(lastUserMsg);
        lastUserMsg.remove();
    }

    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML =
        '<div class="avatar">U</div><div class="bubble">' +
        '<span class="transcription-icon">🎤</span> ' +
        escapeHtml(text) +
        '</div>';
    messagesEl.appendChild(div);
    scrollToBottom();
}

/* ========== STREAMING ========== */
function startStreaming() {
    if (streamActive) {
        return;
    }
    const div = document.createElement('div');
    div.className = 'message bot';
    div.innerHTML =
        '<div class="avatar">T</div><div class="bubble">' +
        '<div class="bot-content"></div>' +
        '</div>';
    messagesEl.appendChild(div);
    const contentEl = div.querySelector('.bot-content');
    currentContentEl = contentEl;
    currentReasoningEl = null;
    streamingContent = '';
    streamingReasoning = '';
    streamActive = true;
    streamLastActivity = Date.now();
    scrollToBottom(true);
}

function ensureReasoningSection() {
    if (currentReasoningEl) {
        return;
    }
    const bubble = currentContentEl ? currentContentEl.parentElement : null;
    if (!bubble) {
        return;
    }
    const wrap = document.createElement('div');
    wrap.className = 'reasoning-wrap collapsed';
    wrap.innerHTML =
        '<button class="reasoning-toggle" aria-expanded="false">' +
        '<span class="reasoning-icon">&#9654;</span> ' +
        '<span class="reasoning-label">Show reasoning</span>' +
        '</button>' +
        '<div class="reasoning-content"></div>';
    bubble.insertBefore(wrap, currentContentEl);
    currentReasoningEl = wrap.querySelector('.reasoning-content');
    const toggle = wrap.querySelector('.reasoning-toggle');
    toggle.addEventListener('click', function () {
        const expanded = toggle.getAttribute('aria-expanded') === 'true';
        toggle.setAttribute('aria-expanded', String(!expanded));
        const icon = toggle.querySelector('.reasoning-icon');
        const label = toggle.querySelector('.reasoning-label');
        icon.innerHTML = expanded ? '&#9654;' : '&#9660;';
        label.textContent = expanded ? 'Show reasoning' : 'Hide reasoning';
        wrap.classList.toggle('collapsed');
    });
}

function updateStreaming(chunk, isReasoning) {
    if (!chunk) {
        return;
    }
    if (isReasoning) {
        streamingReasoning += chunk;
        if (!currentReasoningEl) {
            ensureReasoningSection();
        }
        if (currentReasoningEl) {
            currentReasoningEl.textContent = streamingReasoning;
        }
    } else {
        streamingContent += chunk;
        if (currentContentEl) {
            currentContentEl.innerHTML = renderContent(streamingContent);
        }
    }
    streamLastActivity = Date.now();
}

function finishStreaming() {
    if (currentContentEl) {
        currentContentEl.innerHTML = renderContent(streamingContent);
    }
    if (currentReasoningEl) {
        currentReasoningEl.textContent = streamingReasoning;
    }
    currentContentEl = null;
    currentReasoningEl = null;
    streamingContent = '';
    streamingReasoning = '';
    streamActive = false;
}

/* ========== TYPING INDICATOR ========== */
function showTyping() {
    typingIndicator.classList.add('active');
    scrollToBottom(true);
}
function hideTyping() {
    typingIndicator.classList.remove('active');
}

/* ========== CONNECTION STATUS ========== */
function setConnecting(text) {
    connectionStatus.classList.add('visible');
    statusDot.className = 'status-dot connecting';
    statusText.textContent = text || 'Connecting...';
}
function setConnected(text) {
    connectionStatus.classList.add('visible');
    statusDot.className = 'status-dot';
    statusText.textContent = text || 'Connected';
}
function setDisconnected(text) {
    connectionStatus.classList.add('visible');
    statusDot.className = 'status-dot disconnected';
    statusText.textContent = text || 'Disconnected';
}

/* ========== RECONNECT ========== */
function showReconnectBanner() {
    reconnectBanner.classList.add('active');
}
function hideReconnectBanner() {
    reconnectBanner.classList.remove('active');
}

/* ========== RESPONSE COMPLETION ========== */
function handleResponseDone() {
    if (streamActive) {
        finishStreaming();
    }
    hideTyping();
    hideProgress();
    if (isWaiting) {
        isWaiting = false;
        updateSendButton();
    }
    scrollToBottom(true);
    input.focus({ preventScroll: true });
}

/* ========== WEBSOCKET ========== */
function getWsUrl() {
    if (!currentEndpoint) {
        return null;
    }
    if (
        !currentEndpoint.websocket ||
        typeof currentEndpoint.websocket !== 'string'
    ) {
        return null;
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return (
        protocol +
        '//' +
        window.location.host +
        BASE +
        currentEndpoint.websocket.replace(/^\//, '')
    );
}

function connectWebSocket() {
    const url = getWsUrl();
    if (!url) {
        return;
    }

    if (
        ws &&
        (ws.readyState === WebSocket.OPEN ||
            ws.readyState === WebSocket.CONNECTING)
    ) {
        return;
    }

    manualDisconnect = false;
    setConnecting();

    ws = new WebSocket(url);

    ws.onopen = function () {
        hideReconnectBanner();
        reconnectAttempts = 0;
        setConnected(firstConnect ? undefined : 'Reconnected');
        firstConnect = false;
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        updateSendButton();
        loadSuggestions();
        /// On reconnect, immediately restore conversation from last save
        if (lastSavedData && !manualDisconnect) {
            restoringAfterReconnect = true;
            // Capture messages that appeared after last save (stale/unrecovered)
            const allMsgs = messagesEl.querySelectorAll('.message');
            const staleFrag = document.createDocumentFragment();
            for (let i = lastSavedMsgCount; i < allMsgs.length; i++) {
                staleFrag.appendChild(allMsgs[i].cloneNode(true));
            }
            pendingRestoreMsgs = staleFrag;
            ws.send(
                JSON.stringify({
                    command: '/load',
                    message: 'autosave.json',
                    contents: lastSavedData,
                }),
            );
        }
    };

    ws.onmessage = function (event) {
        let msg;
        try {
            msg = JSON.parse(event.data);
        } catch (e) {
            return;
        }

        switch (msg.type) {
            case 'update':
                handleUpdate(msg.data);
                break;
            case 'confirm':
                handleConfirm(msg.fmt_serialized);
                break;
            case 'save_result':
                if (pendingSaveCallbacks.length > 0) {
                    const cb = pendingSaveCallbacks.shift();
                    cb(msg.contents || msg.data || '');
                }
                break;
        }
    };

    ws.onerror = function (err) {
        console.error('WebSocket error:', err);
    };

    ws.onclose = function () {
        ws = null;
        pendingSaveCallbacks.length = 0;
        handleResponseDone();
        setDisconnected();
        if (!manualDisconnect) {
            showReconnectBanner();
            if (!reconnectTimer) {
                const delay = Math.min(
                    2000 * Math.pow(2, reconnectAttempts),
                    60000,
                );
                reconnectAttempts++;
                reconnectTimer = setTimeout(connectWebSocket, delay);
            }
        }
        updateSendButton();
    };
}

/* ========== HANDLE UPDATE MESSAGES ========== */
function handleUpdate(data) {
    if (!data) {
        return;
    }
    /* handle prompt progress */
    if (data.prompt_progress) {
        const pp = data.prompt_progress;
        const pct =
            pp.total > 0 ? Math.round((pp.processed / pp.total) * 100) : 0;
        showProgress(pct, pp.processed, pp.total);
        return;
    }

    const internal = data['tiz-internal'];
    if (internal) {
        if (internal.interactive_chat_feedback) {
            const feedback = internal.interactive_chat_feedback;
            if (feedback.startsWith('Available commands:')) {
                return;
            }
            if (feedback.startsWith('Attached')) {
                return;
            }
            if (feedback.startsWith('Loaded')) {
                if (restoringAfterReconnect && pendingRestoreMsgs) {
                    restoringAfterReconnect = false;
                    // Clear current messages (they're stale from before reconnect)
                    const oldMsgs = messagesEl.querySelectorAll('.message');
                    for (let i = 0; i < oldMsgs.length; i++) {
                        revokeMsgObjectUrls(oldMsgs[i]);
                    }
                    messagesEl.innerHTML = '';
                    welcomeEl.classList.add('hidden');
                    // Re-add unrecovered (stale) messages using cloned nodes
                    const fragment = pendingRestoreMsgs;
                    const children = Array.from(fragment.childNodes);
                    for (let i = 0; i < children.length; i++) {
                        children[i].classList.add('stale');
                        messagesEl.appendChild(children[i]);
                    }
                    pendingRestoreMsgs = null;
                    if (streamActive || isWaiting) {
                        handleResponseDone();
                    }
                    addMessage(
                        'bot',
                        'Reconnected. Older messages above may be incomplete.',
                    );
                    scrollToBottom();
                    return;
                }
            }
            if (streamActive || isWaiting) {
                handleResponseDone();
            }
            if (feedback === '') {
                return;
            }
            addMessage('bot', feedback);
            return;
        }
        if (internal.interactive_chat_usage) {
            if (streamActive || isWaiting) {
                handleResponseDone();
            }
            updateUsagePanel(internal.interactive_chat_usage);
            return;
        }
        if (internal.transcribe) {
            handleResponseDone();
            const text = internal.transcribe;
            addTranscription(text);
            return;
        }
        const action = internal.action;
        if (action === 'iterator' || action === 'generating_items') {
            return;
        }
        return;
    }

    const delta = data.delta;
    if (delta) {
        hideProgress();
        if (!streamActive) {
            startStreaming();
        }
        if (delta.content) {
            updateStreaming(delta.content, false);
        }
        if (delta.reasoning) {
            updateStreaming(delta.reasoning, true);
        }
        return;
    }

    if (streamActive || isWaiting) {
        handleResponseDone();
    }
}

/* ========== HANDLE CONFIRM ========== */
function handleConfirm(data) {
    const result = window.confirm(String(data));
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'confirm_response', confirm: result }));
    }
}

/* ========== SEND MESSAGE ========== */
function sendMessage(text) {
    const files = attachedFiles.slice();
    const msg = (text || '').trim();
    if ((!msg && files.length === 0) || isWaiting) {
        return;
    }

    isWaiting = true;
    updateSendButton();
    input.value = '';
    autoResize();

    renderAttachedMessage(msg, files);
    attachedFiles.length = 0;
    renderFilePreviews();

    if (files.length > 0) {
        const fileMessages = new Array(files.length);
        let pending = files.length;
        let hadError = false;
        files.forEach(function (file, idx) {
            readFileAsBase64(file)
                .then(function (b64) {
                    fileMessages[idx] = {
                        name: file.name,
                        type: file.type,
                        size: file.size,
                        content: b64,
                    };
                    pending--;
                    if (pending === 0 && !hadError) {
                        sendPayload(msg, fileMessages);
                    }
                })
                .catch(function (err) {
                    hadError = true;
                    pending--;
                    addMessage(
                        'bot',
                        'Error reading file "' +
                            file.name +
                            '": ' +
                            err.message,
                        true,
                    );
                    if (pending === 0) {
                        sendPayload(msg, fileMessages);
                    }
                });
        });
    } else {
        sendPayload(msg, []);
    }
}

function renderAttachedMessage(text, files) {
    welcomeEl.classList.add('hidden');
    const div = document.createElement('div');
    div.className = 'message user';
    let bubbleHtml = '<div class="avatar">U</div><div class="bubble">';
    const msgUrls = [];
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.type.startsWith('image/')) {
            const url = URL.createObjectURL(file);
            objectUrls.push(url);
            msgUrls.push(url);
            bubbleHtml +=
                '<img src="' +
                url +
                '" alt="' +
                escapeHtml(file.name) +
                '" class="preview-img">';
        } else if (file.type.startsWith('audio/')) {
            const url = URL.createObjectURL(file);
            objectUrls.push(url);
            msgUrls.push(url);
            bubbleHtml +=
                '<div class="audio-preview">' +
                '<audio controls src="' +
                url +
                '"></audio>' +
                '<span class="preview-filename">' +
                escapeHtml(file.name) +
                '</span></div>';
        } else {
            bubbleHtml +=
                '<div class="file-attachment">' +
                '<span>' +
                getFileIcon(file) +
                '</span>' +
                '<span>' +
                escapeHtml(file.name) +
                '</span><span class="file-size">' +
                formatFileSize(file.size) +
                '</span></div>';
        }
    }
    if (text) {
        bubbleHtml += escapeHtml(text);
    }
    bubbleHtml += '</div>';
    div.innerHTML = bubbleHtml;
    if (msgUrls.length > 0) {
        div.dataset.objectUrls = msgUrls.join(' ');
    }
    messagesEl.appendChild(div);
    scrollToBottom();
}

function sendPayload(text, fileMessages) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        const hasFilesOnly = !text && fileMessages.length > 0;
        const payload = {
            command: hasFilesOnly ? '/replay' : '',
            message: text,
        };
        if (fileMessages.length > 0) {
            payload.files = fileMessages;
        }
        ws.send(JSON.stringify(payload));
    }
}

/* ========== INTERRUPT ========== */
function interrupt() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'interrupt' }));
    }
    handleResponseDone();
}

/* ========== EVENT HANDLERS ========== */
sendBtn.addEventListener('click', function () {
    if (isWaiting) {
        interrupt();
    } else {
        sendMessage(input.value);
    }
});

input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (isWaiting) {
            interrupt();
        } else {
            sendMessage(input.value);
        }
    }
});

input.addEventListener('input', updateSendButton);

/* ========== SUGGESTIONS ========== */

function loadSuggestions() {
    const container = document.getElementById('suggestions');
    if (!container || !currentEndpoint) {
        return;
    }
    const suggestions = currentEndpoint.suggestions || [];
    container.innerHTML = '';
    suggestions.forEach(function (s) {
        const chip = document.createElement('button');
        chip.className = 'suggestion-chip';
        chip.dataset.prompt = s;
        chip.textContent = s;
        container.appendChild(chip);
    });
}

document.getElementById('suggestions').addEventListener('click', function (e) {
    const chip = e.target.closest('.suggestion-chip');
    if (chip) {
        sendMessage(chip.dataset.prompt);
    }
});

/* ========== RECONNECT NOW ========== */
if (reconnectNow) {
    reconnectNow.addEventListener('click', function () {
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        hideReconnectBanner();
        connectWebSocket();
    });
}

/* ========== FOCUS MANAGEMENT ========== */
container.addEventListener('click', function (e) {
    if (e.target.closest('.message, .suggestions, .welcome')) {
        return;
    }
    input.focus({ preventScroll: true });
});

/* ========== VISIBILITY: reconnect on tab return ========== */
document.addEventListener('visibilitychange', function () {
    if (
        !document.hidden &&
        !manualDisconnect &&
        (!ws || ws.readyState !== WebSocket.OPEN)
    ) {
        connectWebSocket();
    }
});

/* ========== KEYBOARD VISIBILITY on mobile ========== */
if ('visualViewport' in window) {
    const initialViewportHeight = window.visualViewport.height;
    window.visualViewport.addEventListener('resize', function () {
        const diff = initialViewportHeight - window.visualViewport.height;
        if (diff > 100) {
            setTimeout(function () {
                scrollToBottom(true);
            }, 100);
        }
    });
}

/* ========== USAGE PANEL ========== */
function toFiniteNumber(val) {
    const n = Number(val);
    return isFinite(n) ? n : 0;
}

function updateUsagePanel(usage) {
    if (!usage) {
        return;
    }

    const promptTokens = toFiniteNumber(usage.prompt_tokens);
    const completionTokens = toFiniteNumber(usage.completion_tokens);
    const cachedRead = toFiniteNumber(usage.cached_tokens);
    const cachedWrite = toFiniteNumber(usage.cache_write_tokens);
    const promptTime = toFiniteNumber(usage.prompt_time);
    const completionTime = toFiniteNumber(usage.completion_time);
    const cost = toFiniteNumber(usage.cost);
    const inputRate =
        promptTime > 0 ? (promptTokens / promptTime).toFixed(2) : '0.00';
    const outputRate =
        completionTime > 0
            ? (completionTokens / completionTime).toFixed(2)
            : '0.00';
    const toolCalls = usage.tool_calls || [];
    let toolCallsHtml = '';
    const counts = {};
    for (let i = 0; i < toolCalls.length; i++) {
        const name = toolCalls[i][0];
        counts[name] = (counts[name] || 0) + 1;
    }
    const sorted = Object.keys(counts).sort(function (a, b) {
        return counts[b] - counts[a];
    });
    for (let i = 0; i < sorted.length; i++) {
        toolCallsHtml +=
            '<tr><td>' +
            escapeHtml(sorted[i]) +
            '</td><td>' +
            counts[sorted[i]] +
            '</td></tr>';
    }

    usagePanelContent.innerHTML =
        "<table class='usage-table'>" +
        '<tr><td>Input tokens</td><td>' +
        promptTokens +
        ' (' +
        inputRate +
        ' tk/s)</td></tr>' +
        '<tr><td>Output tokens</td><td>' +
        completionTokens +
        ' (' +
        outputRate +
        ' tk/s)</td></tr>' +
        '<tr><td>Cached tokens</td><td>' +
        cachedRead +
        '</td></tr>' +
        '<tr><td>Cache write tokens</td><td>' +
        cachedWrite +
        '</td></tr>' +
        '<tr><td>Credits spent</td><td>$' +
        cost.toFixed(10) +
        '</td></tr>' +
        (toolCallsHtml
            ? "<tr><td colspan='2'><strong>Tools usage</strong></td></tr>" +
              toolCallsHtml
            : '') +
        '</table>';

    usagePanel.classList.remove('hidden');
    scrollToBottom();
}

usagePanelToggle.addEventListener('click', function () {
    const expanded = usagePanelToggle.getAttribute('aria-expanded') === 'true';
    usagePanelToggle.setAttribute('aria-expanded', String(!expanded));
    const icon = usagePanelToggle.querySelector('.usage-panel-icon');
    const label = usagePanelToggle.querySelector('.usage-panel-label');
    icon.innerHTML = expanded ? '&#9654;' : '&#9660;';
    label.textContent = expanded ? 'Show usage' : 'Hide usage';
    usagePanelContent.classList.toggle('hidden');
});

/* ========== PROGRESS ========== */
const progressFill = progressEl.querySelector('.progress-fill');
const progressText = progressEl.querySelector('.progress-text');

function showProgress(pct, processed, total) {
    progressFill.style.width = pct + '%';
    progressText.textContent =
        'Processing: ' + pct + '% (' + processed + '/' + total + ')';
    progressEl.classList.add('visible');
    hideTyping();
    scrollToBottom(true);
}

function hideProgress() {
    progressEl.classList.remove('visible');
    if (isWaiting) {
        showTyping();
    }
}

/* ========== STREAM COMPLETION MONITOR ========== */
const streamMonitorInterval = setInterval(function () {
    if (streamActive && isWaiting) {
        const idleTime = Date.now() - streamLastActivity;
        if (idleTime > 5000) {
            handleResponseDone();
        }
    }
}, 1000);

/* ========== SERVICE WORKER REGISTRATION ========== */
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register(BASE + 'sw.js').catch(function (err) {
        console.warn('SW registration failed:', err);
    });
}

/* ========== FILE ATTACHMENTS ========== */
function formatFileSize(bytes) {
    if (bytes < 1024) {
        return bytes + ' B';
    }
    if (bytes < 1048576) {
        return (bytes / 1024).toFixed(1) + ' KB';
    }
    if (bytes < 1073741824) {
        return (bytes / 1048576).toFixed(1) + ' MB';
    }
    return (bytes / 1073741824).toFixed(1) + ' GB';
}

function getFileIcon(file) {
    if (file.type.startsWith('image/')) {
        return '\uD83D\uDDBC';
    }
    if (file.type.startsWith('video/')) {
        return '\uD83C\uDFAC';
    }
    if (file.type.startsWith('audio/')) {
        return '\uD83C\uDFB5';
    }
    if (file.type.includes('pdf')) {
        return '\uD83D\uDCC4';
    }
    if (
        file.type.includes('zip') ||
        file.type.includes('gzip') ||
        file.type.includes('tar')
    ) {
        return '\uD83D\uDCE6';
    }
    return '\uD83D\uDCCE';
}

function readFileAsBase64(file) {
    return new Promise(function (resolve, reject) {
        const reader = new FileReader();
        reader.onload = function () {
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function removeUrlFromObjectUrls(url) {
    const idx = objectUrls.indexOf(url);
    if (idx !== -1) {
        objectUrls.splice(idx, 1);
    }
}

function removeFile(index) {
    if (attachedFiles[index] && attachedFiles[index]._objectUrl) {
        URL.revokeObjectURL(attachedFiles[index]._objectUrl);
        removeUrlFromObjectUrls(attachedFiles[index]._objectUrl);
    }
    attachedFiles.splice(index, 1);
    renderFilePreviews();
    updateSendButton();
}

function revokeObjectUrls() {
    for (let i = 0; i < objectUrls.length; i++) {
        URL.revokeObjectURL(objectUrls[i]);
    }
    objectUrls.length = 0;
    for (let i = 0; i < attachedFiles.length; i++) {
        if (attachedFiles[i]._objectUrl) {
            URL.revokeObjectURL(attachedFiles[i]._objectUrl);
            attachedFiles[i]._objectUrl = null;
        }
    }
}

function revokeMsgObjectUrls(msgEl) {
    const urls = msgEl.dataset.objectUrls;
    if (!urls) {
        return;
    }
    const urlList = urls.split(' ');
    for (let i = 0; i < urlList.length; i++) {
        if (urlList[i]) {
            URL.revokeObjectURL(urlList[i]);
            removeUrlFromObjectUrls(urlList[i]);
        }
    }
}

function renderFilePreviews() {
    filePreviews.innerHTML = '';
    if (attachedFiles.length === 0) {
        filePreviews.classList.remove('has-files');
        return;
    }
    filePreviews.classList.add('has-files');
    attachedFiles.forEach(function (file, index) {
        const preview = document.createElement('div');
        preview.className = 'file-preview';

        let innerHtml = '';
        if (file.type.startsWith('image/')) {
            // Revoke previous object URL for this file if any
            if (file._objectUrl) {
                URL.revokeObjectURL(file._objectUrl);
                removeUrlFromObjectUrls(file._objectUrl);
            }
            const imgUrl = URL.createObjectURL(file);
            file._objectUrl = imgUrl;
            objectUrls.push(imgUrl);
            innerHtml +=
                '<img src="' +
                imgUrl +
                '" alt="' +
                escapeHtml(file.name) +
                '">';
        } else {
            innerHtml +=
                '<span class="file-icon">' + getFileIcon(file) + '</span>';
        }
        innerHtml +=
            '<span class="file-name">' +
            escapeHtml(file.name) +
            '</span><span class="file-size">' +
            formatFileSize(file.size) +
            '</span>' +
            '<button class="file-remove" data-index="' +
            index +
            '" aria-label="Remove file">&times;</button>';
        preview.innerHTML = innerHtml;
        preview
            .querySelector('.file-remove')
            .addEventListener('click', function () {
                removeFile(index);
            });
        filePreviews.appendChild(preview);
    });
}

function addFiles(fileList) {
    for (let i = 0; i < fileList.length; i++) {
        attachedFiles.push(fileList[i]);
    }
    renderFilePreviews();
    updateSendButton();
}

/* ========== ATTACH BUTTON EVENTS ========== */
attachBtn.addEventListener('click', function () {
    fileInput.value = '';
    fileInput.click();
});

fileInput.addEventListener('change', function () {
    if (fileInput.files.length > 0) {
        addFiles(fileInput.files);
    }
});

cameraInput.addEventListener('change', function () {
    if (cameraInput.files.length > 0) {
        addFiles(cameraInput.files);
    }
});

/* ========== CAMERA & AUDIO BUTTONS ========== */
if (cameraBtn) {
    cameraBtn.addEventListener('click', function () {
        cameraInput.value = '';
        cameraInput.click();
    });
}

if (audioBtn) {
    audioBtn.addEventListener('click', function () {
        handleAudioAction();
    });
}

/* ========== AUDIO RECORDING ========== */
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let isRecording = false;

function handleAudioAction() {
    if (isRecording && mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    } else {
        startAudioRecording();
    }
}

function startAudioRecording() {
    if (isRecording) {
        return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        addMessage('bot', 'Audio recording is not supported in this browser.');
        return;
    }
    navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then(function (stream) {
            audioStream = stream;
            audioChunks = [];
            isRecording = true;
            audioBtn.classList.add('recording');
            try {
                mediaRecorder = new MediaRecorder(stream);
            } catch (e) {
                isRecording = false;
                audioBtn.classList.remove('recording');
                stream.getTracks().forEach(function (t) {
                    t.stop();
                });
                audioStream = null;
                addMessage(
                    'bot',
                    'Could not start audio recording: ' + e.message,
                );
                return;
            }
            mediaRecorder.addEventListener('dataavailable', function (e) {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            });
            mediaRecorder.addEventListener('stop', function () {
                isRecording = false;
                audioBtn.classList.remove('recording');
                if (audioStream) {
                    audioStream.getTracks().forEach(function (t) {
                        t.stop();
                    });
                    audioStream = null;
                }
                if (audioChunks.length === 0) {
                    return;
                }
                const mimeType = MediaRecorder.isTypeSupported('audio/webm')
                    ? 'audio/webm'
                    : MediaRecorder.isTypeSupported('audio/ogg')
                      ? 'audio/ogg'
                      : 'audio/mp4';
                const blob = new Blob(audioChunks, { type: mimeType });
                const ext =
                    mimeType === 'audio/webm'
                        ? '.ogg' // .webm can be mistaken for video/webm
                        : mimeType === 'audio/ogg'
                          ? '.ogg'
                          : '.mp4';
                const file = new File([blob], 'recording-' + Date.now() + ext, {
                    type: mimeType,
                });
                addFiles([file]);
                audioChunks = [];
            });
            mediaRecorder.start();
        })
        .catch(function (err) {
            if (
                err.name === 'NotAllowedError' ||
                err.name === 'PermissionDeniedError'
            ) {
                addMessage(
                    'bot',
                    'Microphone access denied. Please allow microphone permissions.',
                );
            } else {
                addMessage(
                    'bot',
                    'Could not start audio recording: ' + err.message,
                );
            }
        });
}

/* ========== ENDPOINT DISCOVERY & CHAT SWITCHING ========== */

function switchChat(endpoint) {
    if (currentEndpoint && currentEndpoint.name === endpoint.name) {
        return;
    }

    /* close existing connection */
    manualDisconnect = true;
    if (ws) {
        ws.onclose = null;
        ws.close();
        ws = null;
    }
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }

    /* clear chat history */
    resetChatState();

    /* update current endpoint */
    currentEndpoint = endpoint;

    /* update selector label */
    chatSelectorLabel.textContent = endpoint.name || 'Chat';

    /* refresh active state in dropdown */
    populateChatSelector();

    /* close menu */
    chatSelectorMenu.classList.remove('open');
    chatSelectorBtn.setAttribute('aria-expanded', 'false');

    /* show welcome */
    welcomeEl.classList.remove('hidden');

    /* connect to new endpoint */
    manualDisconnect = false;
    firstConnect = true;
    updateSendButton();
    connectWebSocket();
}

function populateChatSelector() {
    chatSelectorOptions.innerHTML = '';
    endpoints.forEach(function (ep) {
        const item = document.createElement('button');
        item.className = 'chat-selector-option';
        if (currentEndpoint && currentEndpoint.name === ep.name) {
            item.classList.add('active');
        }
        item.innerHTML =
            '<span class="chat-selector-option-name">' +
            escapeHtml(ep.name || '?') +
            '</span>' +
            '<span class="chat-selector-option-desc">' +
            escapeHtml(ep.description || '') +
            '</span>';
        item.addEventListener('click', function () {
            switchChat(ep);
        });
        chatSelectorOptions.appendChild(item);
    });
}

chatSelectorBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    const isOpen = chatSelectorMenu.classList.toggle('open');
    chatSelectorBtn.setAttribute('aria-expanded', String(isOpen));
});

document.addEventListener('click', function () {
    chatSelectorMenu.classList.remove('open');
    chatSelectorBtn.setAttribute('aria-expanded', 'false');
});

chatSelectorMenu.addEventListener('click', function (e) {
    e.stopPropagation();
});

/* ========== AUTOSAVE (in-memory for reconnect) ========== */

const pendingSaveCallbacks = [];

function autosaveConversation() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        return;
    }
    lastSavedMsgCount = messagesEl.querySelectorAll('.message').length;
    pendingSaveCallbacks.push(function (b64) {
        lastSavedData = b64;
    });
    ws.send(JSON.stringify({ command: '/save', message: 'autosave.json' }));
}

function initAutosave() {
    return setInterval(function () {
        autosaveConversation();
    }, 300000);
}

/* ========== INIT ========== */

function init() {
    fetch(BASE + 'api/endpoints')
        .then(function (response) {
            if (!response.ok) {
                throw new Error('Failed to fetch endpoints');
            }
            return response.json();
        })
        .then(function (data) {
            endpoints = data.endpoints || [];
            if (endpoints.length === 0) {
                setDisconnected('No chats available');
                chatSelectorLabel.textContent = 'No chats';
                return;
            }
            currentEndpoint = endpoints[0];
            chatSelectorLabel.textContent = currentEndpoint.name || 'Chat';
            populateChatSelector();
            connectWebSocket();
            updateSendButton();
            const autosaveInterval = initAutosave();
            window.addEventListener('beforeunload', function () {
                clearInterval(autosaveInterval);
                clearInterval(streamMonitorInterval);
            });
        })
        .catch(function (err) {
            setDisconnected('Failed to load: ' + err.message);
            chatSelectorLabel.textContent = 'Error';
        });
}

init();
