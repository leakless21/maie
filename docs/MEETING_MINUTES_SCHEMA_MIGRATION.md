# Meeting Minutes Schema Migration Guide

## Overview

This document provides a comprehensive guide for migrating from the basic `meeting_notes_v1.json` schema to the enhanced professional meeting minutes schema. The new schema addresses all identified gaps while maintaining backward compatibility.

## Key Changes Summary

### Before (Old Schema)

```json
{
  "title": "string",
  "abstract": "string",
  "main_points": ["string"],
  "tags": ["string"]
}
```

### After (New Schema)

The new schema includes 6 major components:

1. **Meeting Identification** - Enhanced with date, time, type, location
2. **Participants** - Detailed attendee information with roles
3. **Meeting Administration** - Quorum, previous meeting follow-up
4. **Substantive Content** - Enhanced discussions, decisions, and HIGH PRIORITY action items
5. **Meeting Conclusion** - Adjournment, next meeting, follow-ups
6. **Administrative Tracking** - Document metadata and processing info

## Backward Compatibility

The new schema maintains full backward compatibility by preserving the original fields:

- `title` - Unchanged
- `abstract` - Unchanged
- `main_points` - Unchanged
- `tags` - Unchanged

All new fields are optional in the sense that they have sensible default values, but they are required in the schema definition to ensure completeness.

## Vietnamese Context Adaptations

### 1. Inferred Information Handling

- Fields default to "unknown" when not extractable from transcripts
- Confidence scores track reliability of extracted information
- Special handling for limited temporal information in Vietnamese transcripts

### 2. Consensus-Based Decision Tracking

- Support for various consensus levels: unanimous, majority, general_agreement, conditional_approval, disagreement, deferred
- Vietnamese hierarchical respect considerations in role identification

### 3. Language Processing Notes

- Processing information includes language detection (vi, en, other)
- Special notes for Vietnamese context considerations

## Priority Field Implementations

### HIGH PRIORITY: Structured Action Items

```json
"action_items": [
  {
    "item_description": "Detailed description of the action item",
    "assigned_to": "Person or department responsible",
    "assigned_by": "Person who assigned the action item",
    "due_date": "YYYY-MM-DD format",
    "priority": "high|medium|low|unknown",
    "status": "not_started|in_progress|completed|on_hold|cancelled|unknown",
    "follow_up_required": true|false
  }
]
```

### HIGH PRIORITY: Enhanced Discussion Topic Categorization

```json
"discussion_topics": [
  {
    "topic": "Main discussion topic",
    "category": "administrative|strategic|operational|financial|personnel|project|policy|other",
    "summary": "Brief summary of the discussion",
    "participants_involved": ["names of participants"],
    "duration_minutes": "integer"
  }
]
```

### MEDIUM PRIORITY: Participant Role Identification

```json
"attendees": [
  {
    "name": "Participant name",
    "role": "chairperson|secretary|participant|guest|observer|unknown",
    "department": "Department or organization"
  }
]
```

### MEDIUM PRIORITY: Meeting Type Classification

- board, committee, team, department, executive, project, planning, review, other

## Validation Rules and Constraints

### Date/Time Validation

- Pattern: `"^(\\d{4}-\\d{2}-\\d{2}|unknown)$"` for dates
- Pattern: `"^(\\d{2}:\\d{2}|unknown)$"` for times

### Confidence Score Validation

- Range: 0.0 to 1.0
- Required for all inferred information

### Enum Validation

- Strict enum values for all categorical fields
- Default values for each enum to handle missing data

### Array Constraints

- `minItems` and `maxItems` where appropriate
- Nested validation for complex objects

## Default Values for Optional/Inferred Fields

| Field                    | Default Value | Purpose                        |
| ------------------------ | ------------- | ------------------------------ |
| date                     | "unknown"     | Handle missing temporal info   |
| time                     | "unknown"     | Handle missing temporal info   |
| location                 | "unknown"     | Handle missing location info   |
| meeting_duration_minutes | -1            | Handle missing duration info   |
| confidence scores        | 0.0           | Track reliability of inference |
| quorum.achieved          | false         | Default when not specified     |
| action_item.status       | "not_started" | Default status                 |
| action_item.priority     | "medium"      | Default priority               |

## Migration Implementation Notes

### 1. Schema Loading

The new schema maintains compatibility with the existing [`schema_validator.py`](src/processors/llm/schema_validator.py:1) by:

- Keeping the same filename (`meeting_notes_v1.json`)
- Maintaining the required `tags` field validation
- Following JSON Schema Draft 07 specification

### 2. Data Migration Strategy

For existing data, the system should:

- Preserve all original fields (title, abstract, main_points, tags)
- Fill new fields with appropriate default values
- Process existing transcripts with the new schema to extract enhanced information

### 3. Template Update

The corresponding Jinja template (`meeting_notes_v1.jinja`) will need to be updated to:

- Request all new fields from the LLM
- Provide Vietnamese language instructions for new fields
- Include examples demonstrating the new structure
- Add validation logic for new field requirements

## Testing Considerations

### Schema Validation

- All existing tests should pass with the new schema
- New tests needed for enhanced validation rules
- Confidence score validation tests

### Vietnamese Processing

- Tests for "unknown" default handling
- Tests for confidence scoring
- Tests for language detection

## Implementation Timeline

### Phase 1: Schema Deployment

- Deploy new schema file
- Update validation system
- Update documentation

### Phase 2: Template Enhancement

- Update Jinja template with new instructions
- Add Vietnamese language guidance
- Include comprehensive examples

### Phase 3: Processing Pipeline

- Update LLM prompts for enhanced extraction
- Add confidence scoring logic
- Implement new field processing

## Error Handling

### Missing Information

- Use "unknown" for non-extractable fields
- Apply confidence scoring to indicate reliability
- Log fields with low confidence for review

### Validation Failures

- Clear error messages for each validation failure
- Path information to identify problematic fields
- Graceful degradation for partially valid data

## Professional Standards Compliance

The new schema aligns with corporate governance best practices by including:

- Comprehensive participant tracking
- Formal decision recording with voting details
- Structured action item management
- Administrative tracking and metadata
- Meeting continuity with previous/follow-up meetings
