"""Content validation for generated documentation.

This module provides quality checks for LLM-generated documentation
to ensure it meets standards before being added to the library.
"""

from __future__ import annotations

import re
from typing import Literal

from attrs import define, field, frozen

from docgen.prompts import DocType

ValidationLevel = Literal['error', 'warning', 'info']


@frozen
class ValidationIssue:
    """A validation issue found in content.

    Attributes:
        level: Severity level (error, warning, info).
        code: Short code identifying the issue type.
        message: Human-readable description.
        line: Line number where issue was found (1-indexed), if applicable.
        context: Snippet of content around the issue.
    """

    level: ValidationLevel
    code: str
    message: str
    line: int | None = None
    context: str | None = None


@frozen
class ValidationResult:
    """Result of content validation.

    Attributes:
        is_valid: Whether the content passes validation.
        issues: List of validation issues found.
        title: Title of the validated document.
        errors: Count of error-level issues.
        warnings: Count of warning-level issues.
    """

    is_valid: bool
    issues: tuple[ValidationIssue, ...]
    title: str | None = None

    @property
    def errors(self) -> int:
        return sum(1 for i in self.issues if i.level == 'error')

    @property
    def warnings(self) -> int:
        return sum(1 for i in self.issues if i.level == 'warning')

    @property
    def infos(self) -> int:
        return sum(1 for i in self.issues if i.level == 'info')


# Common LLM artifacts to detect
LLM_ARTIFACTS = [
    (r'\bAs an AI\b', 'LLM self-reference'),
    (r'\bI cannot\b', 'LLM capability statement'),
    (r"\bI don't have access\b", 'LLM access statement'),
    (r'\bAs a language model\b', 'LLM self-reference'),
    (r"\bI'm unable to\b", 'LLM capability statement'),
    (r'\bI apologize\b', 'LLM apology'),
    (r"\bI'd be happy to\b", 'LLM courtesy phrase'),
    (r'\bCertainly!\b', 'LLM courtesy phrase'),
    (r'\bSure!\b', 'LLM courtesy phrase'),
    (r'\bAbsolutely!\b', 'LLM courtesy phrase'),
    (r'\bGreat question\b', 'LLM courtesy phrase'),
]

# Required sections by doc type
REQUIRED_SECTIONS: dict[DocType, list[str]] = {
    'explanation': ['Overview', 'How It Works'],
    'architecture': ['Overview', 'Design'],
    'qa': ['Question', 'Answer'],
    'diagram': [],  # Just needs valid Mermaid
}

# Minimum content length by doc type (characters)
MIN_LENGTH: dict[DocType, int] = {
    'explanation': 200,
    'architecture': 200,
    'qa': 150,
    'diagram': 30,
}


@define
class ContentValidator:
    """Validates generated documentation content.

    This class performs various quality checks on documentation
    to ensure it meets standards before being added to the library.

    Attributes:
        strict: If True, warnings are treated as errors.
        check_mermaid: If True, validate Mermaid diagram syntax.
    """

    strict: bool = False
    check_mermaid: bool = True
    _mermaid_patterns: list[re.Pattern] = field(init=False, factory=list)

    def __attrs_post_init__(self) -> None:
        """Compile regex patterns."""
        # Basic Mermaid diagram type patterns
        self._mermaid_patterns = [
            re.compile(r'^\s*(graph|flowchart)\s+(TB|BT|RL|LR|TD)', re.MULTILINE),
            re.compile(r'^\s*classDiagram', re.MULTILINE),
            re.compile(r'^\s*sequenceDiagram', re.MULTILINE),
            re.compile(r'^\s*stateDiagram', re.MULTILINE),
            re.compile(r'^\s*erDiagram', re.MULTILINE),
            re.compile(r'^\s*gantt', re.MULTILINE),
            re.compile(r'^\s*pie', re.MULTILINE),
            re.compile(r'^\s*journey', re.MULTILINE),
        ]

    def validate(
        self,
        content: str,
        doc_type: DocType,
        title: str | None = None,
    ) -> ValidationResult:
        """Validate documentation content.

        Args:
            content: The content to validate.
            doc_type: Type of documentation.
            title: Optional title for context.

        Returns:
            ValidationResult with any issues found.
        """
        issues: list[ValidationIssue] = []

        # Check minimum length
        issues.extend(self._check_length(content, doc_type))

        # Check for required sections
        issues.extend(self._check_sections(content, doc_type))

        # Check for LLM artifacts
        issues.extend(self._check_llm_artifacts(content))

        # Check code blocks
        issues.extend(self._check_code_blocks(content))

        # Check Mermaid diagrams
        if self.check_mermaid:
            issues.extend(self._check_mermaid(content, doc_type))

        # Check for empty sections
        issues.extend(self._check_empty_sections(content))

        # Check heading structure
        issues.extend(self._check_headings(content))

        # Determine if valid
        error_count = sum(1 for i in issues if i.level == 'error')
        warning_count = sum(1 for i in issues if i.level == 'warning')

        is_valid = error_count == 0
        if self.strict:
            is_valid = is_valid and warning_count == 0

        return ValidationResult(is_valid=is_valid, issues=tuple(issues), title=title)

    def _check_length(self, content: str, doc_type: DocType) -> list[ValidationIssue]:
        """Check minimum content length."""
        issues = []
        min_len = MIN_LENGTH.get(doc_type, 100)

        if len(content) < min_len:
            issues.append(
                ValidationIssue(
                    level='error',
                    code='TOO_SHORT',
                    message=f'Content too short ({len(content)} chars, minimum {min_len})',
                )
            )
        elif len(content) < min_len * 1.5:
            issues.append(
                ValidationIssue(
                    level='warning',
                    code='SHORT_CONTENT',
                    message=f'Content is relatively short ({len(content)} chars)',
                )
            )

        return issues

    def _check_sections(self, content: str, doc_type: DocType) -> list[ValidationIssue]:
        """Check for required sections."""
        issues = []
        required = REQUIRED_SECTIONS.get(doc_type, [])

        for section in required:
            # Look for heading with section name (case-insensitive)
            pattern = rf'^#+\s*{re.escape(section)}'
            if not re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                issues.append(
                    ValidationIssue(
                        level='warning',
                        code='MISSING_SECTION',
                        message=f'Missing recommended section: {section}',
                    )
                )

        return issues

    def _check_llm_artifacts(self, content: str) -> list[ValidationIssue]:
        """Check for LLM-generated artifacts."""
        issues = []

        for pattern, description in LLM_ARTIFACTS:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                # Find line number
                line_num = content[: match.start()].count('\n') + 1
                # Get context
                start = max(0, match.start() - 20)
                end = min(len(content), match.end() + 20)
                context = content[start:end]

                issues.append(
                    ValidationIssue(
                        level='warning',
                        code='LLM_ARTIFACT',
                        message=f'Detected {description}',
                        line=line_num,
                        context=context,
                    )
                )

        return issues

    def _check_code_blocks(self, content: str) -> list[ValidationIssue]:
        """Check code block formatting."""
        issues = []

        # Find code blocks
        code_block_pattern = r'```(\w*)\n(.*?)```'
        matches = list(re.finditer(code_block_pattern, content, re.DOTALL))

        # Check for unclosed code blocks
        open_count = content.count('```')
        if open_count % 2 != 0:
            issues.append(
                ValidationIssue(
                    level='error',
                    code='UNCLOSED_CODE_BLOCK',
                    message='Unclosed code block detected',
                )
            )

        # Check for empty code blocks
        for match in matches:
            code_content = match.group(2).strip()
            if not code_content:
                line_num = content[: match.start()].count('\n') + 1
                issues.append(
                    ValidationIssue(
                        level='warning',
                        code='EMPTY_CODE_BLOCK',
                        message='Empty code block',
                        line=line_num,
                    )
                )

        return issues

    def _check_mermaid(self, content: str, doc_type: DocType) -> list[ValidationIssue]:
        """Check Mermaid diagram syntax."""
        issues = []

        # Find Mermaid blocks
        mermaid_pattern = r'```mermaid\n(.*?)```'
        matches = list(re.finditer(mermaid_pattern, content, re.DOTALL))

        # For diagram type, we expect at least one Mermaid block
        if doc_type == 'diagram' and not matches:
            issues.append(
                ValidationIssue(
                    level='error',
                    code='NO_MERMAID',
                    message='Diagram document has no Mermaid code blocks',
                )
            )
            return issues

        for match in matches:
            mermaid_content = match.group(1).strip()
            line_num = content[: match.start()].count('\n') + 1

            # Check for valid diagram type
            has_valid_type = any(p.search(mermaid_content) for p in self._mermaid_patterns)
            if not has_valid_type:
                issues.append(
                    ValidationIssue(
                        level='error',
                        code='INVALID_MERMAID_TYPE',
                        message='Mermaid block has no recognized diagram type',
                        line=line_num,
                        context=mermaid_content[:100],
                    )
                )
                continue

            # Check for common syntax errors
            mermaid_issues = self._validate_mermaid_syntax(mermaid_content, line_num)
            issues.extend(mermaid_issues)

        return issues

    def _validate_mermaid_syntax(self, content: str, base_line: int) -> list[ValidationIssue]:
        """Validate basic Mermaid syntax."""
        issues = []

        # Check for unbalanced brackets
        brackets = {'[': ']', '(': ')', '{': '}', '"': '"'}
        for open_char, close_char in brackets.items():
            open_count = content.count(open_char)
            close_count = content.count(close_char)
            if open_count != close_count and open_char != close_char:
                issues.append(
                    ValidationIssue(
                        level='warning',
                        code='UNBALANCED_BRACKETS',
                        message=f'Unbalanced {open_char}{close_char} in Mermaid diagram',
                        line=base_line,
                    )
                )

        # Check for extremely long lines (likely truncated)
        for i, line in enumerate(content.split('\n')):
            if len(line) > 200:
                issues.append(
                    ValidationIssue(
                        level='warning',
                        code='LONG_MERMAID_LINE',
                        message='Very long line in Mermaid diagram',
                        line=base_line + i,
                    )
                )

        return issues

    def _check_empty_sections(self, content: str) -> list[ValidationIssue]:
        """Check for empty sections."""
        issues = []

        # Find headings followed immediately by another heading
        pattern = r'^(#+\s+.+)\n\s*\n(#+\s+)'
        matches = list(re.finditer(pattern, content, re.MULTILINE))

        for match in matches:
            line_num = content[: match.start()].count('\n') + 1
            heading = match.group(1).strip()
            issues.append(
                ValidationIssue(
                    level='warning',
                    code='EMPTY_SECTION',
                    message=f'Section appears empty: {heading}',
                    line=line_num,
                )
            )

        return issues

    def _check_headings(self, content: str) -> list[ValidationIssue]:
        """Check heading structure."""
        issues = []

        # Find all headings
        heading_pattern = r'^(#+)\s+(.+)$'
        headings = list(re.finditer(heading_pattern, content, re.MULTILINE))

        if not headings:
            issues.append(
                ValidationIssue(
                    level='warning',
                    code='NO_HEADINGS',
                    message='Content has no headings',
                )
            )
            return issues

        # Check for skipped heading levels
        prev_level = 0
        for match in headings:
            level = len(match.group(1))
            line_num = content[: match.start()].count('\n') + 1

            if level > prev_level + 1 and prev_level > 0:
                issues.append(
                    ValidationIssue(
                        level='info',
                        code='SKIPPED_HEADING_LEVEL',
                        message=f'Heading level skipped (from {prev_level} to {level})',
                        line=line_num,
                    )
                )
            prev_level = level

        return issues

    def validate_batch(
        self,
        contents: list[tuple[str, DocType, str | None]],
    ) -> list[ValidationResult]:
        """Validate multiple documents.

        Args:
            contents: List of (content, doc_type, title) tuples.

        Returns:
            List of ValidationResult objects.
        """
        return [self.validate(c, dt, t) for c, dt, t in contents]


def format_validation_report(result: ValidationResult, title: str = 'Validation') -> str:
    """Format a validation result as a readable report.

    Args:
        result: The validation result.
        title: Title for the report.

    Returns:
        Formatted report string.
    """
    lines = [f'## {title}']
    lines.append('')

    if result.is_valid:
        lines.append('Status: **PASSED**')
    else:
        lines.append('Status: **FAILED**')

    lines.append(f'Errors: {result.errors}, Warnings: {result.warnings}, Info: {result.infos}')
    lines.append('')

    if result.issues:
        lines.append('### Issues')
        for issue in result.issues:
            icon = {'error': 'X', 'warning': '!', 'info': 'i'}.get(issue.level, '?')
            line_info = f' (line {issue.line})' if issue.line else ''
            lines.append(f'- [{icon}] **{issue.code}**{line_info}: {issue.message}')
            if issue.context:
                lines.append(f'  Context: `{issue.context[:60]}...`')

    return '\n'.join(lines)
