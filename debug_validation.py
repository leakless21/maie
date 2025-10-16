#!/usr/bin/env python3
"""Debug script to test the template validation issue."""

import os
import sys

from loguru import logger
from pydantic import ValidationError

from src.api.schemas import Feature, ProcessRequestSchema


def test_template_validation():
    """Test the template validation issue."""
    logger.info("=== Testing Template Validation ===")

    # Test 1: Should work - has template_id
    logger.info("\n1. Testing valid case with template_id...")
    try:
        req = ProcessRequestSchema(file="audio.wav", template_id="meeting_notes_v1")
        logger.info("✓ SUCCESS: {}", req.features)
        logger.info("  template_id: {}", req.template_id)
    except ValidationError:
        logger.exception("✗ FAILED while validating request with template_id")

    # Test 2: Should work - has template_id with SUMMARY feature
    logger.info("\n2. Testing valid case with SUMMARY feature and template_id...")
    try:
        req2 = ProcessRequestSchema(
            file="audio.wav",
            features=[Feature.SUMMARY],
            template_id="nonexistent_template",
        )
        logger.info("✓ SUCCESS: {}", req2.features)
        logger.info("  template_id: {}", req2.template_id)
    except ValidationError:
        logger.exception(
            "✗ FAILED while validating request2 with SUMMARY + template_id"
        )

    # Test 3: Should fail - SUMMARY feature without template_id
    logger.info("\n3. Testing invalid case with SUMMARY feature but no template_id...")
    try:
        req3 = ProcessRequestSchema(file="audio.wav", features=[Feature.SUMMARY])
        logger.warning("✗ UNEXPECTED SUCCESS: {}", req3.features)
        logger.warning("  template_id: {}", req3.template_id)
        logger.warning("  This should have failed!")
    except ValidationError as e:
        logger.info("✓ EXPECTED FAILURE: {}", e)

    # Test 4: Check default features
    logger.info("\n4. Testing default features...")
    try:
        req4 = ProcessRequestSchema(file="audio.wav")
        logger.info("Default features: {}", req4.features)
        has_summary = Feature.SUMMARY in req4.features
        logger.info("Has SUMMARY in defaults: {}", has_summary)
        if has_summary and not req4.template_id:
            logger.error(
                "✗ PROBLEM: Default features include SUMMARY but no template_id required!"
            )
    except ValidationError:
        logger.exception("✗ FAILED while validating default features")


if __name__ == "__main__":
    # Ensure src is importable when running as a script from repo root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    test_template_validation()
