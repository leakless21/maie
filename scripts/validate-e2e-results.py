#!/usr/bin/env python3
"""
E2E Results Validation Script

Validates E2E test results against expected criteria.
Usage: python scripts/validate-e2e-results.py <result_json_file>
"""

import json
import sys
from pathlib import Path
from jsonschema import validate, ValidationError

def validate_result_structure(result: dict) -> bool:
    """Validate basic result structure"""
    required_fields = ['task_id', 'status', 'versions', 'metrics', 'results']

    for field in required_fields:
        if field not in result:
            print(f"❌ Missing required field: {field}")
            return False

    if result['status'] != 'COMPLETE':
        print(f"❌ Status not COMPLETE: {result['status']}")
        return False

    return True

def validate_versions(result: dict) -> bool:
    """Validate version metadata completeness"""
    versions = result['versions']

    required_version_fields = [
        'pipeline_version',
        'asr_backend.name',
        'asr_backend.model_variant',
        'summarization_llm.name'
    ]

    for field_path in required_version_fields:
        keys = field_path.split('.')
        value = versions
        try:
            for key in keys:
                value = value[key]
            if not value:
                print(f"❌ Empty version field: {field_path}")
                return False
        except KeyError:
            print(f"❌ Missing version field: {field_path}")
            return False

    return True

def validate_metrics(result: dict) -> bool:
    """Validate metrics are within reasonable bounds"""
    metrics = result['metrics']

    # Define reasonable ranges
    ranges = {
        'rtf': (0.01, 10.0),
        'vad_coverage': (0.0, 1.0),
        'asr_confidence_avg': (0.0, 1.0),
        'edit_rate_cleaning': (0.0, 1.0)
    }

    for metric, (min_val, max_val) in ranges.items():
        if metric in metrics:
            value = metrics[metric]
            if not (min_val <= value <= max_val):
                print(f"❌ Metric {metric} out of range: {value} (expected {min_val}-{max_val})")
                return False

    return True

def validate_summary_schema(result: dict) -> bool:
    """Validate summary against JSON schema"""
    if 'summary' not in result['results']:
        return True  # Summary not requested

    summary = result['results']['summary']

    # Basic schema check
    required_summary_fields = ['title', 'abstract', 'main_points', 'tags']

    for field in required_summary_fields:
        if field not in summary:
            print(f"❌ Missing summary field: {field}")
            return False

    # Validate tags
    tags = summary['tags']
    if not isinstance(tags, list) or len(tags) < 1 or len(tags) > 5:
        print(f"❌ Invalid tags: {tags} (must be list of 1-5 items)")
        return False

    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate-e2e-results.py <result_json_file>")
        sys.exit(1)

    result_file = Path(sys.argv[1])
    if not result_file.exists():
        print(f"❌ Result file not found: {result_file}")
        sys.exit(1)

    with open(result_file) as f:
        result = json.load(f)

    print(f"🔍 Validating E2E result: {result_file}")

    validations = [
        ("Result Structure", validate_result_structure),
        ("Version Metadata", validate_versions),
        ("Metrics Ranges", validate_metrics),
        ("Summary Schema", validate_summary_schema),
    ]

    all_passed = True
    for name, validator in validations:
        print(f"  Checking {name}...")
        if validator(result):
            print(f"  ✅ {name} passed")
        else:
            print(f"  ❌ {name} failed")
            all_passed = False

    if all_passed:
        print("🎉 All validations passed!")
        sys.exit(0)
    else:
        print("💥 Some validations failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()