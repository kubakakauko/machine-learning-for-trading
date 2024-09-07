#!/bin/bash
set -e

# Install/update dependencies
poetry install --no-interaction --no-ansi

# Execute CMD
exec "$@"
