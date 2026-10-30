"""Opt-in repairs for exported Recovery BDDL, written only to new run artifacts."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import re


def repair_text(text):
    changes = []
    for section in ('fixtures', 'objects'):
        pattern = re.compile(r'\(:' + section + r'\s+(.*?)\)', re.S)
        match = pattern.search(text)
        if match is None:
            continue
        groups = OrderedDict()
        declarations = re.findall(r'([^\n]+?)\s+-\s+(\w+)\s*(?:\n|$)', match.group(1))
        for names, category in declarations:
            groups.setdefault(category, []).extend(names.split())
        if len(groups) != len(declarations):
            body = '\n' + '\n'.join('    ' + ' '.join(names) + ' - ' + category
                                     for category, names in groups.items()) + '\n  '
            text = text[:match.start(1)] + body + text[match.end(1):]
            changes.append('merge_repeated_' + section + '_categories')
    # Older exports of floor tasks used the nonexistent main_table alias.
    fixtures = re.search(r'\(:fixtures\s+(.*?)\)', text, re.S)
    if ('(problem libero_floor_manipulation)' in text and fixtures
            and re.search(r'\bfloor\s+-\s+floor\b', fixtures.group(1))
            and re.search(r'\bmain_table(?:_|\b)', text)):
        text = re.sub(r'\bmain_table(?=_|\b)', 'floor', text)
        changes.append('floor_workspace_alias')
    return text, changes


def prepare_bddl(source, output):
    source = Path(source)
    original = source.read_text()
    repaired, changes = repair_text(original)
    original_sha = hashlib.sha256(original.encode()).hexdigest()
    if not changes:
        return str(source), None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f'{original_sha}.bddl'
    if not destination.exists():
        with destination.open('x') as stream:
            stream.write(repaired)
    metadata = {'source': str(source), 'source_sha256': original_sha,
                'runtime_bddl': str(destination.resolve()),
                'runtime_sha256': hashlib.sha256(repaired.encode()).hexdigest(), 'changes': changes}
    return str(destination.resolve()), metadata
