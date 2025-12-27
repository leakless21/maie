Things to watch if MAIE returns already-translated content (server-side translation on MAIE):

Payload shape remains consistent

Current parsing assumes maieStatus.results.summary exists and is an object; required subfields are optional. Keep field names stable: title, summary/content/text/body, tags (array), key_topics (or keyTopics), and clean_transcript/raw_transcript.
If translation alters keys or nests content, update the server selectors in process.ts:569-763 accordingly.
Encoding/normalization

Server normalizes strings with .normalize("NFC") before encrypting. Ensure MAIE returns UTF-8 JSON; avoid mixed normalization to prevent preview/title corruption.
Transcript and summary are encrypted byte-for-byte; any encoding mismatch will break decryption consistency.
Metrics unaffected by translation

metrics.input_duration_seconds, processing_time_seconds, rtf, asr_confidence_avg, etc., are stored as-is. Ensure translation doesn’t drop or rename these.
Optional fields tolerance

If translated templates omit summary fields, server already falls back to "" preview and empty keyTopics. No server change needed if keys remain absent rather than renamed.
Socket payload expectations

Real-time emit sends structuredSummary: summary and summary preview string. If clients display localized text, fine; but if clients expect English for downstream processing (e.g., tags, key topics), ensure they can handle Vietnamese.
Tags/key topics localization

Tags and key topics are stored exactly as returned. If MAIE translates tags, DB will store Vietnamese strings and tag matching will become locale-specific. Decide if tags should stay canonical/English or be translated.
Transcript language

If MAIE returns translated transcript, the stored transcript becomes Vietnamese. Any downstream comparison or search that assumes source language will be impacted.
API versioning / docs

If MAIE payload shape changes, update docs and swagger only when server parsing changes. If shape stays the same with translated content, no server doc change needed.
Error fields

Failures store error/error_code as-is. If those become localized, operational debugging may be harder. Consider keeping error codes stable/English even if messages are translated.
Testing considerations

Verify COMPLETE path with translated summary/transcript: encryption storage, preview slicing (first 200 chars) still sensible with Vietnamese.
Ensure NFC normalization is safe for Vietnamese accents (it is) and doesn’t alter content meaning.
In short: keep the response schema stable; ensure UTF-8/NFC; be intentional about tags/key topics/transcript localization effects; keep metrics and error codes untouched; no code changes needed if only the text content is translated and field names stay the same.
