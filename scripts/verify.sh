#!/bin/bash
# Verification script for Ariadne installation
set -e

echo "=== Ariadne Verification ==="
echo

echo "Running tests..."
uv run pytest tests/ -v
echo

echo "Testing CLI..."
uv run ariadne --help
echo

echo "Testing config command..."
uv run ariadne config
echo

echo "=== All checks passed! ==="
