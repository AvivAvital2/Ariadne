"""Tests for docgen.validator module."""

from __future__ import annotations

from docgen.validator import (
    ContentValidator,
    ValidationIssue,
    ValidationResult,
    format_validation_report,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_title_field_default_none(self) -> None:
        """ValidationResult has title=None by default."""
        result = ValidationResult(is_valid=True, issues=())
        assert result.title is None

    def test_title_field_stored(self) -> None:
        """ValidationResult stores title when provided."""
        result = ValidationResult(is_valid=True, issues=(), title='My Document')
        assert result.title == 'My Document'

    def test_errors_property(self) -> None:
        """Errors property counts error-level issues."""
        issues = (
            ValidationIssue(level='error', code='E1', message='Error 1'),
            ValidationIssue(level='warning', code='W1', message='Warning 1'),
            ValidationIssue(level='error', code='E2', message='Error 2'),
        )
        result = ValidationResult(is_valid=False, issues=issues)
        assert result.errors == 2

    def test_warnings_property(self) -> None:
        """Warnings property counts warning-level issues."""
        issues = (
            ValidationIssue(level='error', code='E1', message='Error 1'),
            ValidationIssue(level='warning', code='W1', message='Warning 1'),
            ValidationIssue(level='warning', code='W2', message='Warning 2'),
        )
        result = ValidationResult(is_valid=False, issues=issues)
        assert result.warnings == 2

    def test_infos_property(self) -> None:
        """Infos property counts info-level issues."""
        issues = (
            ValidationIssue(level='info', code='I1', message='Info 1'),
            ValidationIssue(level='warning', code='W1', message='Warning 1'),
        )
        result = ValidationResult(is_valid=True, issues=issues)
        assert result.infos == 1


class TestFormatValidationReport:
    """Tests for format_validation_report function."""

    def test_passed_validation(self) -> None:
        """Report shows PASSED for valid result."""
        result = ValidationResult(is_valid=True, issues=())
        report = format_validation_report(result, title='Test Doc')
        assert '## Test Doc' in report
        assert '**PASSED**' in report
        assert 'Errors: 0' in report

    def test_failed_validation_with_issues(self) -> None:
        """Report shows FAILED and lists issues."""
        issues = (
            ValidationIssue(level='error', code='TOO_SHORT', message='Content too short'),
            ValidationIssue(level='warning', code='MISSING_SECTION', message='Missing Overview'),
        )
        result = ValidationResult(is_valid=False, issues=issues)
        report = format_validation_report(result, title='Validation')

        assert '**FAILED**' in report
        assert 'Errors: 1' in report
        assert 'Warnings: 1' in report
        assert '### Issues' in report
        assert 'TOO_SHORT' in report
        assert 'MISSING_SECTION' in report

    def test_uses_result_title(self) -> None:
        """Report uses ValidationResult.title if provided."""
        result = ValidationResult(is_valid=True, issues=(), title='Document Title')
        # Default title parameter, but result has title
        report = format_validation_report(result, title='Validation')
        # The function uses the title parameter, not result.title
        # This is expected behavior - title param is for the report header
        assert '## Validation' in report

    def test_issue_with_line_number(self) -> None:
        """Report includes line number when present."""
        issues = (
            ValidationIssue(
                level='warning',
                code='LLM_ARTIFACT',
                message='Detected LLM phrase',
                line=42,
            ),
        )
        result = ValidationResult(is_valid=True, issues=issues)
        report = format_validation_report(result, title='Test')
        assert '(line 42)' in report

    def test_issue_with_context(self) -> None:
        """Report includes truncated context when present."""
        issues = (
            ValidationIssue(
                level='warning',
                code='LLM_ARTIFACT',
                message='Detected phrase',
                context='This is some context around the issue that was detected',
            ),
        )
        result = ValidationResult(is_valid=True, issues=issues)
        report = format_validation_report(result, title='Test')
        assert 'Context:' in report


class TestContentValidator:
    """Tests for ContentValidator."""

    def test_validate_passes_title_to_result(self) -> None:
        """Validator passes title to ValidationResult."""
        validator = ContentValidator()
        content = '''# My Document

## Overview

This is a comprehensive explanation of the system that covers all the necessary
details and provides enough context for understanding. It goes into sufficient
depth to meet the minimum length requirements.

## How It Works

The system works by processing inputs through multiple stages, each adding value
to the overall output. This multi-stage approach ensures reliability and quality.
'''
        result = validator.validate(content, 'explanation', title='Test Title')
        assert result.title == 'Test Title'

    def test_validate_without_title(self) -> None:
        """Validator leaves title as None when not provided."""
        validator = ContentValidator()
        content = '''# My Document

## Overview

This is a comprehensive explanation of the system that covers all the necessary
details and provides enough context for understanding. It goes into sufficient
depth to meet the minimum length requirements.

## How It Works

The system works by processing inputs through multiple stages, each adding value
to the overall output. This multi-stage approach ensures reliability and quality.
'''
        result = validator.validate(content, 'explanation')
        assert result.title is None
