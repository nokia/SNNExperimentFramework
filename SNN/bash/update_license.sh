#!/bin/bash

# Script to update license headers in Python files
# Usage: ./update_license.sh

# Check if license files exist
if [[ ! -f "old_licence.txt" ]]; then
    echo "Error: old_licence.txt not found in current directory"
    exit 1
fi

if [[ ! -f "new_licence.txt" ]]; then
    echo "Error: new_licence.txt not found in current directory"
    exit 1
fi

# Read the old and new license content
OLD_LICENSE=$(cat old_licence.txt)
NEW_LICENSE=$(cat new_licence.txt)

# Count total lines in old license
OLD_LICENSE_LINES=$(($(wc -l < old_licence.txt)+1))
echo "Old license has $OLD_LICENSE_LINES lines."

# Find all Python files recursively
echo "Searching for Python files..."
echo "================================"

mkdir -p temp_backup
CHANGED_FILES=0

# Use find to get all .py files
find . -name "*.py" -type f | while read -r file; do
    # Extract first 14 lines (or number of lines in old license)
    LINES_TO_CHECK=$OLD_LICENSE_LINES
    if [[ $LINES_TO_CHECK -gt 14 ]]; then
        LINES_TO_CHECK=14
    fi

    # Get the first N lines of the file
    FIRST_LINES=$(head -n "$LINES_TO_CHECK" "$file")

    # Compare with old license (ignoring trailing whitespace)
    if [[ "$FIRST_LINES" == "$OLD_LICENSE" ]]; then
        echo "Updating license in: $file"

        # store a backup of the file for safety
        cp "$file" "temp_backup/$(basename "$file").bak"

        # Create temporary file
        TEMP_FILE=$(mktemp)

        # Write new license to temp file
        echo "$NEW_LICENSE" > "$TEMP_FILE"

        # Append the rest of the original file (skip old license lines)
        tail -n +$((LINES_TO_CHECK + 1)) "$file" >> "$TEMP_FILE"

        # Replace original file with updated content
        mv "$TEMP_FILE" "$file"

        CHANGED_FILES=$((CHANGED_FILES + 1))
    fi
done

echo "================================"
echo "License update completed."
echo "Total files changed: $CHANGED_FILES"