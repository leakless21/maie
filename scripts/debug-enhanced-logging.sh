#!/bin/bash
# Debug script to verify enhanced logging is working

set -euo pipefail

echo "🔍 Enhanced Logging Debug Script"
echo "================================"
echo

# Check if logs directory exists
if [ ! -d "logs" ]; then
    echo "❌ logs directory not found"
    exit 1
fi

echo "✅ logs directory exists"

# Check if errors.log exists
if [ ! -f "logs/errors.log" ]; then
    echo "❌ logs/errors.log not found"
    exit 1
fi

echo "✅ logs/errors.log exists"

# Check recent log entries for enhanced logging
echo
echo "📋 Recent error log entries (last 10):"
echo "--------------------------------------"
tail -n 10 logs/errors.log | grep -E "(JSON parsing failed|Schema validation failed)" || echo "No recent JSON/schema validation errors found"

echo
echo "🔍 Checking for enhanced logging data..."
echo "--------------------------------------"

# Look for STRUCTURED data in recent logs
RECENT_STRUCTURED=$(tail -n 50 logs/errors.log | grep "STRUCTURED:" | wc -l)

if [ "$RECENT_STRUCTURED" -gt 0 ]; then
    echo "✅ Found $RECENT_STRUCTURED recent entries with enhanced logging data"
    echo
    echo "📄 Sample enhanced logging entry:"
    echo "--------------------------------"
    tail -n 50 logs/errors.log | grep "STRUCTURED:" | head -n 1
else
    echo "❌ No recent enhanced logging data found"
    echo
    echo "💡 To test enhanced logging, run:"
    echo "   python -c \""
    echo "   import sys; sys.path.insert(0, 'src')"
    echo "   from src.config.logging import configure_logging, bind_correlation_id"
    echo "   from src.processors.llm.schema_validator import validate_llm_output"
    echo "   configure_logging(); bind_correlation_id('debug-test')"
    echo "   validate_llm_output('{\"broken\": json', {'type': 'object', 'properties': {}})"
    echo "   \""
fi

echo
echo "📊 Enhanced Logging Status:"
echo "---------------------------"

# Check if the log format includes STRUCTURED data
if grep -q "STRUCTURED:" logs/errors.log; then
    echo "✅ Enhanced logging is ACTIVE"
    echo "   - Structured error data is being captured"
    echo "   - Raw LLM output is included in logs"
    echo "   - Error details (line, column, position) are available"
else
    echo "❌ Enhanced logging is INACTIVE"
    echo "   - Only basic error messages are being logged"
    echo "   - No structured data is available"
fi

echo
echo "🎯 Next Steps:"
echo "--------------"
echo "1. Run your LLM processing task"
echo "2. Check logs/errors.log for entries with 'STRUCTURED:'"
echo "3. Look for 'raw_output' field to see the actual LLM output"
echo "4. Use 'error_type' field to filter specific error types"
echo
echo "📖 For more details, see: docs/SCHEMA_VALIDATION_DEBUGGING.md"