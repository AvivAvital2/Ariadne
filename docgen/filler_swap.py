"""The §18.4 filler swap — structural source rendering for Spool builds.

A generation prompt = a structural spine + one intent-filler slot. For
user repos the slot holds raw source (tier 3 fills, the only occupied
rung). For Spool pack builds tier 2 exists, so the swap: render the
spine from the bundle's structure (signatures, imports — no comments,
no docstrings) and let a symbol-anchored official-docs provider fill
intent. Design: designs/spool-environment-plugin.md §18.4.
"""


def structural_source(bundle, intent_filler=None) -> str:
    """Render the prompt's ``source_code`` slot without raw source prose.

    ``intent_filler`` is an optional callable(bundle) -> str returning
    symbol-anchored official-doc excerpts (tier 2); its output is the
    only prose admitted, under an explicit certified-docs banner.
    """
    lines = [
        f'# module: {bundle.module_name} '
        f'({bundle.language}, {bundle.line_count} lines; '
        f'structural rendering — source prose withheld)',
    ]
    for imp in bundle.imports:
        lines.append(f'import {imp.module}')
    for enriched in bundle.elements:
        lines.append(enriched.element.signature)
    if intent_filler is not None:
        excerpt = intent_filler(bundle)
        if excerpt:
            lines += [
                '',
                '# --- certified official documentation (tier 2) ---',
                excerpt,
            ]
    return '\n'.join(lines)
