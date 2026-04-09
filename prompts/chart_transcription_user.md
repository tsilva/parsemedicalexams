Extract clinically meaningful data from this `{chart_type}` page.

Use these rules:
- If this is an audiogram, output exactly:
  [AUDIOGRAM]
  Right ear thresholds (dB HL):
  - 125 Hz: value
  - 250 Hz: value
  - 500 Hz: value
  - 1K Hz: value
  - 2K Hz: value
  - 4K Hz: value
  - 8K Hz: value
  Left ear thresholds (dB HL):
  - 125 Hz: value
  - 250 Hz: value
  - 500 Hz: value
  - 1K Hz: value
  - 2K Hz: value
  - 4K Hz: value
  - 8K Hz: value
  Speech audiogram:
  - Right ear SRT: value
  - Left ear SRT: value
  - Right ear discrimination: value
  - Left ear discrimination: value
- If this is a tympanometry page, output exactly:
  [TYMPANOMETRY]
  Right ear:
  - Probe tone: 226 Hz
  - Volume: value
  - Pressure: value
  - Compliance: value
  - Gradient: value
  Left ear:
  - Probe tone: 226 Hz
  - Volume: value
  - Pressure: value
  - Compliance: value
  - Gradient: value
  Audiologist: value
- Include only frequencies or scores that are actually readable
- Use `unreadable` for any requested field that cannot be read
- If the page has no uniquely readable discrete plotted values beyond chart scaffold labels, output exactly:
  [NON_DISCRETE_VISUAL_CHART]
  chart_type: {chart_type}
  chart_data_status: non_discrete_visual
  visible_labels: comma-separated labels

Embedded PDF text for context:
{embedded_text}
