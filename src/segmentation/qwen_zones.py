"""Additional Qwen inputs derived from cleaned author text."""
import re

from src.segmentation.segments import KINDS

BLOCK_LIMIT = 6
STRUCTURAL_KINDS = ('introduction', 'main_text', 'conclusion', 'lists', 'postscript')
PARAGRAPH_KINDS = tuple(f'paragraph_{i}' for i in range(1, BLOCK_LIMIT + 1))
SEMANTIC_KINDS = tuple(f'semantic_block_{i}' for i in range(1, BLOCK_LIMIT + 1))
EXTRA_KINDS = STRUCTURAL_KINDS + PARAGRAPH_KINDS + SEMANTIC_KINDS
QWEN_KINDS = KINDS + EXTRA_KINDS

BULLET = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+')
POSTSCRIPT = re.compile(r'^\s*(?:P\.?\s*S\.?|П\.?\s*С\.?)(?:\s*[:.\-]\s*|\s+|$)', re.I)


def expand_qwen_zones(zones):
    """Keep full body for Qwen and derive overlapping, bounded diagnostic inputs."""
    body_lines = zones['body'].splitlines()
    signature_lines = zones['signature'].splitlines()
    postscript = ''
    for lines in (body_lines, signature_lines):
        index = next((i for i, line in enumerate(lines) if POSTSCRIPT.match(line)), None)
        if index is not None:
            postscript = '\n'.join(lines[index:]).strip()
            if lines is body_lines:
                body_lines = lines[:index]
            break

    paragraphs = [part.strip() for part in re.split(r'\n\s*\n', '\n'.join(body_lines))
                  if part.strip()]
    prose = [part for part in paragraphs if any(not BULLET.match(line) for line in part.splitlines())]
    introduction = prose[0] if len(prose) >= 2 else ''
    conclusion = prose[-1] if len(prose) >= 3 else ''
    main = prose[1:-1] if conclusion else prose[1:] if introduction else prose
    result = {kind: zones[kind] for kind in KINDS}
    result.update(introduction=introduction, main_text='\n\n'.join(main), conclusion=conclusion,
                  lists='\n'.join(line for line in body_lines if BULLET.match(line)), postscript=postscript)
    if len(paragraphs) > BLOCK_LIMIT:
        paragraphs = paragraphs[:BLOCK_LIMIT - 1] + ['\n\n'.join(paragraphs[BLOCK_LIMIT - 1:])]
    for kinds, blocks in ((PARAGRAPH_KINDS, paragraphs),
                          (SEMANTIC_KINDS, zones.get('_semantic_blocks', ()))):
        result.update({kind: blocks[i] if i < len(blocks) else '' for i, kind in enumerate(kinds)})
    return result
