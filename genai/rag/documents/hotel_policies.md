# HotelMind Group — Guest & Operations Policies

## Check-in / Check-out

Standard check-in time is 14:00 local time; standard check-out is 11:00
local time. Early check-in and late check-out are offered on a
space-available basis:
- Early check-in before 12:00 is complimentary when the room was vacant
  the previous night; otherwise a half-day rate (50% of the nightly rate)
  applies.
- Late check-out up to 15:00 is complimentary for loyalty members (Silver
  tier and above); all other guests are charged a half-day rate for
  check-out after 13:00 and a full-day rate after 18:00.
- Guests must present a valid government-issued photo ID and the credit
  card used for booking (or an equivalent form of guaranteed payment) at
  check-in.

## Cancellation Policy

- Standard rate bookings: free cancellation up to 48 hours before the
  check-in date. Cancellations within 48 hours are charged one night's
  stay plus applicable taxes.
- Non-refundable rate bookings: no refund at any point after booking
  confirmation, except where required by local consumer-protection law.
- Group bookings (10+ rooms): governed by a separate group contract with
  its own cancellation schedule; contact the Sales department.
- No-shows are charged the full first-night rate and the remaining nights
  of the reservation are released for resale.

## Pricing and Rate Changes

- Room rates are dynamically adjusted based on forecast occupancy, day of
  week, local events, and booking lead time, per the HotelMind pricing
  model (`src/prediction/predict_pricing.py`).
- Rate changes never apply retroactively to confirmed, paid reservations.
- Rate parity is maintained across direct and OTA (online travel agency)
  channels; a mismatch of more than 2% must be escalated to Revenue
  Management within 24 hours of discovery.

## Guest Conduct and Room Policies

- Smoking is prohibited in all guest rooms and indoor public areas; a
  cleaning fee of the equivalent of one night's rate applies to violations.
- Pets are welcome in designated pet-friendly room categories only,
  subject to a non-refundable pet fee and a maximum of two pets per room.
- Maximum occupancy per room type is enforced strictly for fire-safety
  compliance (see `dim_room.max_occupancy` in the warehouse schema);
  additional guests beyond the limit require a room upgrade or rollaway
  bed surcharge, subject to availability.
- Quiet hours are 22:00–07:00; repeated noise complaints against a room
  may result in relocation or, in serious cases, removal without refund.

## Loyalty Program Tiers

| Tier | Qualifying nights/year | Key benefits |
|---|---|---|
| Member | 0 | Member rates, birthday reward |
| Silver | 10 | Late check-out, welcome amenity |
| Gold | 25 | Room upgrade (space-available), lounge access |
| Platinum | 50 | Guaranteed room availability (48h), suite upgrade priority |

## Data Privacy

Guest personal data (name, contact details, payment information, stay
history) is processed only for reservation fulfillment, loyalty program
administration, and — in aggregated/anonymized form — for demand
forecasting and operational analytics (the ML models described in this
repository). Guests may request data export or deletion through the
front-desk manager or the corporate Data Protection Officer, subject to
retention requirements for financial and tax records.
