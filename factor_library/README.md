# Factor library map

`factors.json` is the machine-readable source of truth. It contains 31 factor
definitions linked to 22 expert mechanisms and 16 traceable sources. `status`
describes implementation or experiment state; `evidence_level` describes the
strength of support. A literature-backed `K2` factor is still not accepted for
this project until fixed-model experiments produce the required `E*` evidence.

## Priority for the confirmed local data contract

The baseline has 96 aligned `Power` and `GHI_real` history steps, station
capacity/longitude/latitude, Beijing time, and 16-step issued future arrays for
GHI, temperature, wind speed, and wind direction. Executable families now
include:

1. `factor.power.multiscale-ramp`, `multiscale-slope`, `variability`, and
   `acceleration` as one historical-dynamics family. The baseline already has
   all 96 raw power lags, so `multiscale-lags` is a grouped representation, not
   new information.
2. Capacity, previous-day, solar-position, daylight, clear-sky, and recent
   power clear-sky-index factors.
3. Module-temperature, temperature-corrected irradiance, low-irradiance,
   irradiance regime, joint ramp, clipping-score, and weather-power-residual
   factors.
4. `factor.quality.future-weather-coverage` and
   `factor.quality.stuck-shift-score` to expose input reliability.
5. `factor.weather.future-change` for target-prefix GHI, temperature, and wind
   trajectory summaries, and `factor.weather.clear-sky-index-forecast` for a
   bounded target-time cloud/irradiance state normalized across season and site.

Do not add all groups at once. Run a paired family ablation in this order,
then retain only evidence-supported groups before testing interactions.

## Conditional tiers

- Orientation, tilt, altitude, and time-versioned capacity remain unavailable;
  factors requiring them beyond the confirmed static metadata stay conditional.
- NWP issue time and archived forecast cycles unlock forecast age, revision,
  causal bias correction, and trustworthy future-weather availability audits.
- Availability, curtailment, clipping, maintenance, or alarm metadata unlock
  operational-state factors. Without labels, residual heuristics remain weak
  hypotheses and must not be treated as truth.

## Output shapes

- `static`: one value per forecast origin.
- `history_sequence`: a causal sequence ending at the origin.
- `future_known_sequence`: one value per future horizon, derived only from an
  issued forecast or deterministic metadata available at the origin.

The private adapter may flatten sequences for TabM, but it must keep deterministic
column order and include that mapping in the preprocessing fingerprint.

## Recommended first office experiment

Before any factor trial, decide whether the target is all 16 future steps or
only the four-hour endpoint. Establish baseline parity. Test
`factor.weather.future-change` and
`factor.weather.clear-sky-index-forecast` separately before their combination,
using identical seeds and rows and reporting every selected horizon, monthly
distribution, coverage, and runtime.
