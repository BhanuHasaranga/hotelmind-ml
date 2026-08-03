# HotelMind Group — Standard Operating Procedures (SOP)

## Housekeeping

### Daily Room Turnover
1. Housekeeping receives the day's checkout list from the Front Office
   system by 09:00.
2. Departed rooms are prioritized for cleaning ahead of stay-over rooms so
   that new check-ins are not delayed past the 14:00 standard check-in
   time.
3. A room is marked `AVAILABLE` in `dim_room.current_status` only after
   passing a supervisor inspection; rooms failing inspection are routed
   back to `CLEANING` status, not `AVAILABLE`.
4. Rooms requiring repair are marked `MAINTENANCE` and excluded from the
   available-room count used by the occupancy forecasting model
   (`fact_occupancy_daily.total_rooms` reflects only active, non-maintenance
   rooms).

### Staffing Ratios
Standard housekeeping staffing follows an attendant-to-room ratio of
approximately 1:14 for a full turnover shift; this ratio is the basis for
the synthetic staffing baseline used by the Staffing forecasting model
(see `src/pipelines/synthetic_data.py::DEPARTMENT_STAFF_RATIO`). Actual
required staff for a given day is produced by `POST /predict/staff`, which
should be treated as the authoritative recommendation over the static
ratio when the two disagree.

## Front Office

### Overbooking Protocol
Branches may overbook by up to 3% of total room inventory to offset
expected no-shows, based on the branch's trailing 90-day no-show rate. If
walk situations occur (a confirmed guest cannot be accommodated), Front
Office must: (1) offer a comparable or better room at a partner property
at no additional cost to the guest, (2) cover reasonable transportation to
the alternate property, and (3) log the walk in the incident register for
Revenue Management review.

### Handling Guest Complaints
1. Acknowledge the complaint within 5 minutes of it being raised.
2. Classify the complaint using the standard taxonomy (cleanliness, food,
   staff, price, location, noise, maintenance, wifi, parking, other) —
   the same taxonomy used by the automated guest-review analysis pipeline
   (`genai/reviews/pipeline.py`), so that manually logged complaints and
   review-derived complaints roll up into the same operational dashboards.
3. Resolve at the lowest possible staff level; escalate to the Duty
   Manager only if resolution requires a rate adjustment, comp, or
   relocation.
4. Log the resolution and any compensation issued within the same shift.

## Restaurant & F&B

### Demand Planning
Kitchen prep quantities for breakfast, lunch, and dinner are informed by
`POST /predict/restaurant`, which forecasts expected quantity and revenue
per meal period from recent order volume and occupancy. Kitchen managers
should treat the forecast as a planning guide, not a hard cap — a 10%
prep buffer above the point forecast is standard practice to avoid
stockouts during unexpectedly high demand.

### Waste Tracking
Any unsold, spoiled, or over-prepped food must be logged by category and
approximate value at the end of each shift. Recurring waste patterns
(e.g. consistent over-prep at a specific meal period) should be flagged to
Revenue/Operations for forecast recalibration — this is one of the inputs
the AI Insights Generator's restaurant-waste rules
(`genai/insights/rules/restaurant_waste.py`) are designed to surface
automatically from mart data.

## Escalation Matrix

| Situation | First responder | Escalate to | Escalation SLA |
|---|---|---|---|
| Guest complaint (non-urgent) | Front desk agent | Duty Manager | 30 min |
| Safety/security incident | Security | General Manager | Immediate |
| Rate parity violation | Revenue analyst | Revenue Manager | 24 hours |
| Forecasted occupancy anomaly (>15pp deviation) | Ops dashboard alert | Regional Ops Director | Same business day |
| Churn-risk guest (High risk, Platinum tier) | CRM system alert | Guest Relations Manager | Same business day |
