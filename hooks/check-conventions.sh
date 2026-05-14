#!/bin/bash
# Read the edited file path from stdin
INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only check Rust source files
[[ "$FILE" == *.rs ]] || exit 0

# Read the agentdocs convention files
STYLE=$(cat "$CLAUDE_PROJECT_DIR/agentdocs/style.md" 2>/dev/null)
ARCH=$(cat "$CLAUDE_PROJECT_DIR/agentdocs/architecture.md" 2>/dev/null)

cat <<EOF
Review the edit you just made to $FILE against the project conventions below. If you introduced a violation, fix it immediately.

--- style.md ---
$STYLE

--- architecture.md ---
$ARCH
EOF
