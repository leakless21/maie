#!/usr/bin/env python3
"""Debug script to test the template validation issue."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pydantic import ValidationError
from src.api.schemas import ProcessRequestSchema, Feature


def test_template_validation():
    """Test the template validation issue."""
    print("=== Testing Template Validation ===")

    # Test 1: Should work - has template_id
    print("\n1. Testing valid case with template_id...")
    try:
        req = ProcessRequestSchema(file="audio.wav", template_id="meeting_notes_v1")
        print(f"✓ SUCCESS: {req.features}")
        print(f"  template_id: {req.template_id}")
    except ValidationError as e:
        print(f"✗ FAILED: {e}")

    # Test 2: Should work - has template_id with SUMMARY feature
    print("\n2. Testing valid case with SUMMARY feature and template_id...")
    try:
        req2 = ProcessRequestSchema(
            file="audio.wav",
            features=[Feature.SUMMARY],
            template_id="nonexistent_template",
        )
        print(f"✓ SUCCESS: {req2.features}")
        print(f"  template_id: {req2.template_id}")
    except ValidationError as e:
        print(f"✗ FAILED: {e}")

    # Test 3: Should fail - SUMMARY feature without template_id
    print("\n3. Testing invalid case with SUMMARY feature but no template_id...")
    try:
        req3 = ProcessRequestSchema(file="audio.wav", features=[Feature.SUMMARY])
        print(f"✗ UNEXPECTED SUCCESS: {req3.features}")
        print(f"  template_id: {req3.template_id}")
        print("  This should have failed!")
    except ValidationError as e:
        print(f"✓ EXPECTED FAILURE: {e}")

    # Test 4: Check default features
    print("\n4. Testing default features...")
    try:
        req4 = ProcessRequestSchema(file="audio.wav")
        print(f"Default features: {req4.features}")
        has_summary = Feature.SUMMARY in req4.features
        print(f"Has SUMMARY in defaults: {has_summary}")
        if has_summary and not req4.template_id:
            print(
                "✗ PROBLEM: Default features include SUMMARY but no template_id required!"
            )
    except ValidationError as e:
        print(f"✗ FAILED: {e}")


if __name__ == "__main__":
    test_template_validation()
