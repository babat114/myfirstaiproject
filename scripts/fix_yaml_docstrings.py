"""
Fix compact YAML docstrings in API route files.
Converts ;-separated compact format to proper indented YAML for flasgger.

Usage: python scripts/fix_yaml_docstrings.py
"""
import re
import os
import sys

FILES = [
    'app/routes/api/datasets.py',
    'app/routes/api/models.py',
    'app/routes/api/training.py',
    'app/routes/api/comments.py',
    'app/routes/api/stream.py',
]


def fix_compact_yaml(yaml_block: str) -> str:
    """
    Convert compact YAML format to proper indented YAML.

    Handles:
      1. - in: query; name: X; ...; schema: {type: T, ...}
      2. - in: path; name: X; required: true; schema: {type: T}
      3. requestBody: {content: {type: {schema: {type: object, ...}}}}
      4. responses: {200: {description: ...}}
      5. tags: [Tag1, Tag2]
      6. summary: ...; description: ...
      7. security: [{...}, {...}]
    """
    lines = yaml_block.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        base_indent = line[:len(line) - len(stripped)]

        # Pattern 1: "- in: query; name: ...; ..."
        param_m = re.match(
            r'^- in:\s*(query|path|header|cookie|formData)\s*;\s*(.+)$',
            stripped
        )
        if param_m:
            in_val = param_m.group(1)
            rest = param_m.group(2)
            result.append(f'{base_indent}- in: {in_val}')
            result.extend(_expand_pairs(rest, base_indent + '  '))
            i += 1
            continue

        # Pattern 3: requestBody compact
        rb_m = re.match(r'^requestBody:\s*\{(.+)\}$', stripped)
        if rb_m:
            result.append(f'{base_indent}requestBody:')
            result.extend(_expand_braced_content(rb_m.group(1), base_indent + '  '))
            i += 1
            continue

        # Pattern 4: responses: {200: {description: ...}, 400: {...}}
        resp_m = re.match(r'^responses:\s*\{(.+)\}$', stripped)
        if resp_m:
            result.append(f'{base_indent}responses:')
            result.extend(_expand_responses_compact(resp_m.group(1), base_indent + '  '))
            i += 1
            continue

        # Pattern 5: tags: [Tag1, Tag2]
        tags_m = re.match(r'^tags:\s*\[(.+)\]$', stripped)
        if tags_m:
            result.append(f'{base_indent}tags:')
            for tag in _split_balanced(tags_m.group(1), ','):
                tag = tag.strip()
                if tag:
                    result.append(f'{base_indent}  - {tag}')
            i += 1
            continue

        # Pattern 6: tags: [X]; summary: ...
        ts_m = re.match(r'^tags:\s*\[(.+)\];\s*summary:\s*(.+)$', stripped)
        if ts_m:
            result.append(f'{base_indent}tags:')
            for tag in _split_balanced(ts_m.group(1), ','):
                tag = tag.strip()
                if tag:
                    result.append(f'{base_indent}  - {tag}')
            result.append(f'{base_indent}summary: {ts_m.group(2).strip()}')
            i += 1
            continue

        # Pattern 7: summary: ...; description: ...
        sd_m = re.match(r'^summary:\s*(.+);\s*description:\s*(.+)$', stripped)
        if sd_m:
            result.append(f'{base_indent}summary: {sd_m.group(1).strip()}')
            result.append(f'{base_indent}description: {sd_m.group(2).strip()}')
            i += 1
            continue

        # Pattern 7: security: [{...}, {...}]
        sec_m = re.match(r'^security:\s*\[(.+)\]$', stripped)
        if sec_m:
            result.append(f'{base_indent}security:')
            for item in _split_balanced(sec_m.group(1), ',', braces_only=True):
                item = item.strip()
                if item:
                    result.append(f'{base_indent}  - {item}')
            i += 1
            continue

        # Pattern 8: parameters: [{in: path; name: X; ...}, {in: query; name: Y; ...}]
        params_arr_m = re.match(r'^parameters:\s*\[(.+)\]$', stripped)
        if params_arr_m:
            result.append(f'{base_indent}parameters:')
            items = _split_balanced(params_arr_m.group(1), ';', braces_only=True)
            for item in items:
                item = item.strip()
                if item.startswith('{') and item.endswith('}'):
                    item = item[1:-1]
                if item:
                    # Parse the item's key-value pairs
                    pairs = _parse_semicolon_pairs(item)
                    # First pair is always "in: path" or "in: query"
                    if pairs:
                        first_key, first_val = pairs[0]
                        result.append(f'{base_indent}  - {first_key}: {first_val}')
                        for key, value in pairs[1:]:
                            if key in ('schema',) and value.startswith('{') and value.endswith('}'):
                                result.append(f'{base_indent}    {key}:')
                                result.extend(_expand_braced_content(value[1:-1], base_indent + '      '))
                            else:
                                result.append(f'{base_indent}    {key}: {value}')
            i += 1
            continue

        # Pattern 9: Single-line key: {complex value with ; inside}
        # e.g. "content: {application/json: {schema: {type: object}}}"
        # These are usually inside requestBody, already handled above

        result.append(line)
        i += 1

    return '\n'.join(result)


def _expand_pairs(s: str, indent: str) -> list:
    """Expand 'key1: val1; key2: {inner}; key3: val3' into indented lines."""
    out = []
    pairs = _parse_semicolon_pairs(s)
    for key, value in pairs:
        if key in ('schema', 'content', 'properties'):
            if value.startswith('{') and value.endswith('}'):
                out.append(f'{indent}{key}:')
                out.extend(_expand_braced_content(value[1:-1], indent + '  '))
            else:
                out.append(f'{indent}{key}: {value}')
        elif key == 'requestBody':
            out.append(f'{indent}{key}:')
            if value.startswith('{') and value.endswith('}'):
                out.extend(_expand_braced_content(value[1:-1], indent + '  '))
            else:
                out.append(f'{indent}  {value}')
        elif key == 'responses':
            out.append(f'{indent}{key}:')
            inner = value
            if inner.startswith('{') and inner.endswith('}'):
                inner = inner[1:-1]
            out.extend(_expand_responses_compact(inner, indent + '  '))
        elif key == 'enum':
            if value.startswith('[') and value.endswith(']'):
                out.append(f'{indent}{key}:')
                for item in _split_balanced(value[1:-1], ','):
                    item = item.strip()
                    if item:
                        out.append(f'{indent}  - {item}')
            else:
                out.append(f'{indent}{key}: {value}')
        elif value.startswith('{') and value.endswith('}'):
            out.append(f'{indent}{key}:')
            out.extend(_expand_braced_content(value[1:-1], indent + '  '))
        elif value.startswith('[') and value.endswith(']'):
            out.append(f'{indent}{key}:')
            for item in _split_balanced(value[1:-1], ','):
                item = item.strip()
                if item:
                    out.append(f'{indent}  - {item}')
        else:
            out.append(f'{indent}{key}: {value}')
    return out


def _parse_semicolon_pairs(s: str) -> list:
    """Parse 'key1: val1; key2: {nested}; key3: val3' into (key, value) pairs."""
    pairs = []
    pos = 0
    s = s.strip()
    while pos < len(s):
        m = re.match(r'(\w+)\s*:\s*', s[pos:])
        if not m:
            break
        key = m.group(1)
        pos += m.end()

        if pos >= len(s):
            pairs.append((key, ''))
            break

        if s[pos] in '{[':
            # Nested structure — find matching close
            open_ch = s[pos]
            close_ch = '}' if open_ch == '{' else ']'
            depth = 1
            start = pos
            pos += 1
            while pos < len(s) and depth > 0:
                if s[pos] == open_ch:
                    depth += 1
                elif s[pos] == close_ch:
                    depth -= 1
                pos += 1
            value = s[start:pos]
            pairs.append((key, value))
        else:
            # Simple value — scan to next '; ' or end
            semi = s.find('; ', pos)
            if semi == -1:
                value = s[pos:].strip()
                pairs.append((key, value))
                break
            else:
                value = s[pos:semi].strip()
                pairs.append((key, value))
                pos = semi + 2
                continue

        # Skip delimiter after nested value
        if pos < len(s) and s[pos:pos+2] == '; ':
            pos += 2
        elif pos < len(s) and s[pos] == ';':
            pos += 1

    return pairs


def _expand_braced_content(s: str, indent: str) -> list:
    """Expand '{key1: val1, key2: {inner}, ...}' into indented lines."""
    out = []
    pairs = _parse_comma_pairs(s)
    for key, value in pairs:
        if value.startswith('{') and value.endswith('}'):
            out.append(f'{indent}{key}:')
            out.extend(_expand_braced_content(value[1:-1], indent + '  '))
        elif value.startswith('[') and value.endswith(']'):
            out.append(f'{indent}{key}:')
            for item in _split_balanced(value[1:-1], ','):
                item = item.strip()
                if item:
                    out.append(f'{indent}  - {item}')
        else:
            out.append(f'{indent}{key}: {value}')
    return out


def _parse_comma_pairs(s: str) -> list:
    """Parse 'key1: val1, key2: val2, ...' (comma-separated within braces)."""
    pairs = []
    pos = 0
    s = s.strip()
    while pos < len(s):
        m = re.match(r'(\w+)\s*:\s*', s[pos:])
        if not m:
            break
        key = m.group(1)
        pos += m.end()

        if pos >= len(s):
            pairs.append((key, ''))
            break

        if s[pos] in '{[':
            open_ch = s[pos]
            close_ch = '}' if open_ch == '{' else ']'
            depth = 1
            start = pos
            pos += 1
            while pos < len(s) and depth > 0:
                if s[pos] == open_ch:
                    depth += 1
                elif s[pos] == close_ch:
                    depth -= 1
                pos += 1
            value = s[start:pos]
            pairs.append((key, value))
        else:
            comma = s.find(', ', pos)
            if comma == -1:
                value = s[pos:].strip().rstrip(',')
                pairs.append((key, value))
                break
            else:
                value = s[pos:comma].strip()
                pairs.append((key, value))
                pos = comma + 2
                continue

        if pos < len(s) and s[pos:pos+2] == ', ':
            pos += 2
        elif pos < len(s) and s[pos] == ',':
            pos += 1

    return pairs


def _expand_responses_compact(s: str, indent: str) -> list:
    """Expand '200: {description: ...}, 400: {description: ...}'."""
    out = []
    # Find all status code blocks: NNN: {description: ...}
    for m in re.finditer(r'(\d+)\s*:\s*\{([^}]*)\}', s):
        code = m.group(1)
        inner = m.group(2).strip()
        out.append(f'{indent}{code}:')
        # Parse description and optional content
        desc_m = re.match(r'description:\s*(.+)$', inner)
        if desc_m:
            out.append(f'{indent}  description: {desc_m.group(1).strip()}')
        else:
            # More complex inner — try key-value pairs
            inner_pairs = _parse_semicolon_pairs(inner) or _parse_comma_pairs(inner)
            for k, v in inner_pairs:
                out.append(f'{indent}  {k}: {v}')
    return out


def _split_balanced(s: str, delim: str, braces_only: bool = False) -> list:
    """Split string by delimiter, respecting brace/bracket nesting."""
    parts = []
    depth = 0
    current = ''
    for ch in s:
        if ch in '{[':
            depth += 1
            current += ch
        elif ch in '}]':
            depth -= 1
            current += ch
        elif ch == delim and depth == 0:
            parts.append(current.strip())
            current = ''
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def fix_file(filepath: str) -> int:
    """Fix YAML docstrings in a Python file. Returns number of docstrings fixed."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    fixed_count = 0

    # Find all function docstrings containing YAML blocks (---)
    # Strategy: use regex to find def ... """...---..."""
    def replace_docstring(match):
        nonlocal fixed_count
        full = match.group(0)
        def_line = match.group(1)
        rest = match.group(2)

        # Find the YAML separator --- within the docstring
        # Docstring format: """...\n    ---\n    yaml content\n    """
        yaml_sep_match = re.search(r'\n(\s*)---\n', rest)
        if not yaml_sep_match:
            return full

        yaml_indent = yaml_sep_match.group(1)
        sep_pos = rest.index(yaml_sep_match.group(0))

        before_yaml = rest[:sep_pos + len(yaml_sep_match.group(0))]
        after_yaml_start = sep_pos + len(yaml_sep_match.group(0))

        # Find end of docstring: """ with same or less indentation
        end_match = re.search(r'\n(\s*)"""', rest[after_yaml_start:])
        if not end_match:
            return full

        yaml_body = rest[after_yaml_start:after_yaml_start + end_match.start()]
        closing = rest[after_yaml_start + end_match.start():]

        # Only process if there's compact format
        if ';' not in yaml_body and 'responses: {' not in yaml_body:
            return full

        try:
            fixed_yaml = fix_compact_yaml(yaml_body)

            # Re-indent the fixed YAML
            fixed_lines = []
            for line in fixed_yaml.split('\n'):
                if line.strip():
                    fixed_lines.append(yaml_indent + line)
                else:
                    fixed_lines.append('')

            new_rest = before_yaml + '\n'.join(fixed_lines) + closing
            fixed_count += 1
            return def_line + new_rest
        except Exception as e:
            print(f'  WARNING: Failed to fix docstring: {e}')
            return full

    # Pattern: def func_name(...):\n    """...---
    content = re.sub(
        r'(def\s+\w+\([^)]*\):\s*\n)(\s*""".*?---.*?""")',
        replace_docstring,
        content,
        flags=re.DOTALL,
    )

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return fixed_count

    return fixed_count


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    total = 0
    for relpath in FILES:
        filepath = os.path.join(base_dir, relpath)
        if os.path.exists(filepath):
            n = fix_file(filepath)
            print(f'{relpath}: {n} docstrings fixed')
            total += n
        else:
            print(f'{relpath}: NOT FOUND')
    print(f'\nTotal: {total} docstrings fixed across {len(FILES)} files')


if __name__ == '__main__':
    main()
