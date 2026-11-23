"""MIME parsing and ownership-aware segmentation of email messages.

Replies and forwarded/quoted history are *not* attributed to the current author.
Four zones are kept for compatibility: subject, greeting, body, signature.
"""
from __future__ import annotations

import email
import re
from email import policy
from html.parser import HTMLParser

RULES_VERSION = 'rules-v2-inline-boundaries'

KINDS = ('subject', 'greeting', 'body', 'signature')
GREETING = re.compile(
    r'^(?:hi\b|hey\b|hello\b|dear\b|good (?:morning|evening|afternoon)\b|'
    r'здравств(?:уй|уйте)\b|добр(?:ый\s+(?:день|вечер)|ое\s+утро|ой\s+ночи)\b|привет\b|хай\b|уважаем\w*|коллеги\b|всем привет\b)', re.I)
SIGNOFF = re.compile(
    r'^(?:--\s*$|с уважением\b|с наилучшими\b|всего доброго\b|'
    r'best regards\b|kind regards\b|warm regards\b|regards\b|sincerely\b|'
    r'cheers\b|best wishes\b|thanks(?:\s+(?:and|&amp;)\s+regards)?\s*[,!]*$|'
    r'отправлено с (?:моего|iphone|ipad|android)\b|sent from my\b)', re.I)
_INLINE_GREETING = re.compile(
    r'^\s*(?:'
    r'(?:hi|hey|hello|dear)\b[^.!?\n]{0,100}?[,!:]|'
    r'good\s+(?:morning|afternoon|evening)\b[^.!?\n]{0,100}?[,!:]|'
    r'здравствуйте\b[^.!?\n]{0,100}?[,!:]|'
    r'добр(?:ый\s+(?:день|вечер)|ое\s+утро|ой\s+ночи)\b[^.!?\n]{0,100}?[,!:]|'
    r'(?:привет|хай|уважаем\w*|коллеги)\b[^.!?\n]{0,100}?[,!:]'
    r')(?=\s|$)', re.I)
_INLINE_SIGNOFF_TERM = re.compile(
    r'(?<!\w)(?:best\s+regards|kind\s+regards|warm\s+regards|regards|'
    r'sincerely|best\s+wishes|cheers|с\s+уважением|с\s+наилучшими(?:\s+пожеланиями)?|'
    r'всего\s+доброго|thanks|thank\s+you)(?!\w)', re.I)
_INLINE_SIGNOFF_TAIL = re.compile(
    r'^(?:best\s+regards|kind\s+regards|warm\s+regards|regards|sincerely|'
    r'best\s+wishes|cheers|с\s+уважением|с\s+наилучшими(?:\s+пожеланиями)?|'
    r'всего\s+доброго)(?:\s*[,!:]?\s*[\wА-Яа-яЁё][^\n]{0,120})?$|'
    r'^(?:thanks|thank\s+you)\s*[,!]\s*[\wА-Яа-яЁё][^\n]{0,120}$', re.I)
CONTACT = re.compile(r'(?:[\w.+-]+@[\w.-]+\.\w+|\+\d[\d ()-]{6,}|(?:тел|phone|tel)\s*[:.])', re.I)
FORWARD_HEADER = re.compile(
    r'^(?:-{2,}\s*)?(?:begin )?(?:original message|forwarded message|'
    r'исходное сообщение|пересланное сообщение)\s*(?:-{2,}|:)?\s*$', re.I)
REPLY_HEADER = re.compile(
    r'^(?:'
    r'on .{1,200}\bwrote:\s*$|'
    r'в .{1,200}\bписал(?:а|и|\(а\))?:\s*$|'
    r'from:\s*.+$|от:\s*.+$)', re.I)
DISCLAIMER = re.compile(r'^(?:confidentiality notice|this (?:e-?mail|message) (?:and any attachments )?is confidential|'
                        r'данное (?:электронное )?письмо (?:и вложения )?содержит)', re.I)


class _HTMLText(HTMLParser):
    def __init__(self, include_quotes=False):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.stack = []
        self.include_quotes = include_quotes
        self.at_line_start = True

    def _append(self, text):
        quoted = self.stack and self.stack[-1][2]
        for piece in text.splitlines(keepends=True):
            if quoted and self.at_line_start and piece.strip():
                self.parts.append('> ')
            self.parts.append(piece)
            self.at_line_start = (piece.endswith(('\n', '\r')) or
                                  (self.at_line_start and not piece.strip()))

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        classes = values.get('class', '') or ''
        block_quote = (tag == 'blockquote' or
                       bool(re.search(r'\b(?:gmail_quote|yahoo_quoted|moz-cite-prefix)\b', classes, re.I)) or
                       values.get('type') == 'cite')
        style = re.sub(r'\s+', '', values.get('style') or '').lower()
        hidden = (tag in {'script', 'style', 'head', 'noscript', 'template', 'iframe'} or
                  'hidden' in values or 'display:none' in style or 'visibility:hidden' in style)
        if not self._skipped and (block_quote or tag in {'p', 'div', 'br', 'li', 'tr', 'h1', 'h2', 'h3'}):
            self._append('\n')
        if tag in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
                   'meta', 'param', 'source', 'track', 'wbr'}:
            return
        quoted = block_quote or bool(self.stack and self.stack[-1][2])
        self.stack.append((tag, self._skipped or hidden or (quoted and not self.include_quotes), quoted))
        if tag == 'li' and not self._skipped:
            self._append('- ')

    @property
    def _skipped(self):
        return bool(self.stack and self.stack[-1][1])

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                if not self._skipped and (tag in {'p', 'div', 'li', 'tr'} or
                                          (self.stack[i][2] and not (i and self.stack[i - 1][2]))):
                    self._append('\n')
                del self.stack[i:]
                break

    def handle_data(self, data):
        if not self._skipped:
            self._append(data)


def parse_eml(raw: bytes | str, *, include_html_quotes=False) -> tuple[str, str]:
    if isinstance(raw, str):
        raw = raw.encode('utf-8', 'replace')
    msg = email.message_from_bytes(raw, policy=policy.default)
    subject = str(msg.get('Subject') or '').strip()
    plain, html = [], []
    pending = [msg]
    while pending:
        part = pending.pop()
        if part.get_content_disposition() == 'attachment' or part.get_content_maintype() == 'message':
            continue
        if part.is_multipart():
            pending.extend(reversed(list(part.iter_parts())))
            continue
        if part.get_content_type() not in {'text/plain', 'text/html'}:
            continue
        try:
            value = part.get_content()
        except (LookupError, UnicodeError, ValueError):
            value = (part.get_payload(decode=True) or b'').decode('utf-8', 'replace')
        if isinstance(value, bytes):
            value = value.decode('utf-8', 'replace')
        (plain if part.get_content_type() == 'text/plain' else html).append(value)
    if any(value.strip() for value in plain):
        body = '\n'.join(plain)
    else:
        parser = _HTMLText(include_quotes=include_html_quotes)
        parser.feed('\n'.join(html))
        parser.close()
        body = ''.join(parser.parts)
    body = re.sub(r'\n{3,}', '\n\n', body.replace('\r\n', '\n').replace('\r', '\n')).strip()
    return subject, body


def is_quote_line(line: str) -> bool:
    return line.lstrip().startswith(('>', '| '))


def is_reply_header(lines: list[str], i: int) -> bool:
    line = lines[i].strip()
    if FORWARD_HEADER.match(line):
        return True
    if not REPLY_HEADER.match(line):
        return False
    if not line.lower().startswith(('from:', 'от:')):
        return True
    return any(re.match(r'^\s*(?:sent|date|to|subject|отправлено|дата|кому|тема):',
                        following, re.I) for following in lines[i + 1:i + 5])


def _remove_quoted_history(lines: list[str]) -> list[str]:
    output = []
    for i, line in enumerate(lines):
        s = line.strip()
        if is_reply_header(lines, i):
            next_line = next((following for following in lines[i + 1:] if following.strip()), '')
            if s.lower().startswith(('on ', 'в ')) and is_quote_line(next_line):
                continue
            break
        if DISCLAIMER.match(s) and output:
            break
        if is_quote_line(line):
            continue
        output.append(line)
    return output


def owned_lines(body: str) -> list[str]:
    """Current author's lines, excluding known quoted history and blank edges."""
    lines = [line.rstrip() for line in body.replace('\r\n', '\n').replace('\r', '\n').split('\n')]
    lines = _remove_quoted_history(lines)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _inline_signature_start(text: str, *, min_start: int = 0) -> int | None:
    """Find a final sign-off embedded in a single physical line.

    A phrase such as "best regards" in the middle of a sentence is not a
    signature. The marker must be the final fragment and must either start
    the message or follow sentence-ending punctuation.
    """
    for match in _INLINE_SIGNOFF_TERM.finditer(text, max(0, min_start)):
        prefix = text[:match.start()].rstrip()
        tail = text[match.start():].strip()
        if prefix and prefix[-1] not in '.!?;':
            continue
        if _INLINE_SIGNOFF_TAIL.fullmatch(tail):
            return match.start()
    return None


def _inline_greeting_end(text: str) -> int | None:
    match = _INLINE_GREETING.match(text)
    if not match:
        return None
    end = match.end()
    # In greetings such as "Здравствуйте, коллеги!" or "Hello, everyone!"
    # the first comma is not the end of the salutation. Extend only for a
    # small, explicit recipient vocabulary so body prose is not swallowed.
    if match.group(0).rstrip().endswith(','):
        recipient_pattern = (
            r'\s*(?:коллеги(?:\s+и\s+друзья)?|друзья|команда|всем|все|'
            r'партнёры|партнеры|everyone|team|folks|guys|'
            r'ladies\s+and\s+gentlemen|all)\s*[.!?]'
        )
        recipient = re.match(recipient_pattern, text[end:], re.I)
        if recipient:
            end += recipient.end()
    return end


def _single_line_sentence_ends(text: str, start: int, end: int) -> list[int]:
    """Find conservative sentence/heading boundaries in a one-line message."""
    endings = '.!?。！？:'
    closing = '»”’\'\")]}'
    abbreviations = ('mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'e.g.', 'i.e.',
                     'etc.', 'vs.', 'г.', 'ул.', 'д.', 'стр.', 'т.д.', 'т.п.', 'им.')
    result = []
    i = start
    while i < end:
        if text[i] not in endings:
            i += 1
            continue
        if text[i] == '.' and i > start and i + 1 < end and text[i - 1].isdigit() and text[i + 1].isdigit():
            i += 1
            continue
        punct_end = i + 1
        while punct_end < end and text[punct_end] in '.!?。！？':
            punct_end += 1
        while punct_end < end and text[punct_end] in closing:
            punct_end += 1
        if punct_end >= end or not text[punct_end].isspace():
            i = punct_end
            continue
        left = text[max(start, i - 18):i + 1].casefold()
        token_start = max(text.rfind(' ', start, i), text.rfind('\t', start, i)) + 1
        token = text[token_start:i + 1].casefold()
        if text[i] == '.' and (token in abbreviations or
                               re.fullmatch(r'[a-zа-яё]\.', token, flags=re.I) or
                               any(left.endswith(abbr) for abbr in abbreviations)):
            i = punct_end
            continue
        # A colon in a time, URL, or key:value pair is not a sentence break.
        if text[i] == ':' and i > start and text[i - 1].isdigit():
            i += 1
            continue
        result.append(punct_end)
        i = punct_end
        while i < end and text[i].isspace():
            i += 1
    return result


def segmentation_units(body: str) -> tuple[list[str], list[str]]:
    """Return exact text units plus their original separators for LLM indexing.

    The normal case is one unit per physical line. For a one-line email only,
    clear greeting/sign-off boundaries are exposed as additional units. The
    separators are retained separately, so applying model boundaries never
    inserts new characters or rewrites the source text.
    """
    lines = owned_lines(body)
    if len(lines) != 1:
        return lines, ['\n'] * max(0, len(lines) - 1)

    text = lines[0]
    greeting_end = _inline_greeting_end(text)
    signature_start = _inline_signature_start(text, min_start=greeting_end or 0)
    body_start = greeting_end or 0
    body_end = signature_start if signature_start is not None else len(text)
    cuts = {cut for cut in (greeting_end, signature_start)
            if cut is not None and 0 < cut < len(text)}
    # Sentence/heading units give the LLM useful boundaries even when a user
    # pasted the entire email into a single textbox line.
    cuts.update(cut for cut in _single_line_sentence_ends(text, body_start, body_end)
                if 0 < cut < len(text))
    cuts = sorted(cuts)
    if not cuts:
        return lines, []

    raw_ranges = list(zip([0, *cuts], [*cuts, len(text)]))
    spans = []
    for start, end in raw_ranges:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    units = [text[start:end] for start, end in spans]
    separators = [text[left[1]:right[0]] for left, right in zip(spans, spans[1:])]
    return (units, separators) if units else (lines, [])


def split_email(subject: str, body: str) -> dict[str, str]:
    lines = owned_lines(body)
    greeting = signature = ''
    # A pasted message often arrives as one long line. Detect only unambiguous
    # leading salutations and final sign-offs; preserve every source character
    # between the resulting zones (apart from trimmed zone-edge whitespace).
    if len(lines) == 1:
        line = lines[0]
        greeting_end = _inline_greeting_end(line)
        signature_start = _inline_signature_start(line, min_start=greeting_end or 0)
        greeting = line[:greeting_end].strip() if greeting_end is not None else ''
        body_end = signature_start if signature_start is not None else len(line)
        body_start = greeting_end if greeting_end is not None else 0
        message = line[body_start:body_end].strip()
        signature = line[signature_start:].strip() if signature_start is not None else ''
        if greeting or signature:
            return {'subject': subject.strip(), 'greeting': greeting,
                    'body': message, 'signature': signature}
    if lines:
        # Some emails begin with a name / salutation on a single line.
        first = lines[0].strip()
        if len(first) <= 120 and GREETING.match(first):
            greeting = lines.pop(0).strip()
            while lines and not lines[0].strip():
                lines.pop(0)
    # Signature markers near end; prefer the *first* valid marker so name/title remain inside.
    for i in range(max(0, len(lines) - 12), len(lines)):
        if SIGNOFF.match(lines[i].strip()):
            signature = '\n'.join(lines[i:]).strip()
            lines = lines[:i]
            break
    if not signature and len(lines) >= 2:
        blocks = re.split(r'\n\s*\n', '\n'.join(lines))
        last = blocks[-1].strip()
        if len(blocks) > 1 and len(last.splitlines()) <= 5 and CONTACT.search(last):
            signature = last
            lines = '\n\n'.join(blocks[:-1]).split('\n')
    return {'subject': subject.strip(), 'greeting': greeting,
            'body': '\n'.join(lines).strip(), 'signature': signature}


def document_text(zones: dict[str, str]) -> str:
    """Stable embedding input for the current author's text only."""
    labels = {'subject': 'Subject', 'greeting': 'Greeting', 'body': 'Message', 'signature': 'Signature'}
    return '\n'.join(f'{labels[k]}: {zones[k]}' for k in KINDS if zones[k].strip())
