# Station metadata and historical GHI contract

Read this reference before enabling any capacity, solar-geometry, irradiance,
PV-physics, weather-regime, clipping, or weather-power residual factor.

## Confirmed private inputs

- `station_info.csv` identifies a station with `plantid` and supplies installed
  capacity `GCCAPACITY`, longitude `LONGITUDE`, and latitude `LATITUDE`.
- Every station uses the `Asia/Shanghai` civil timezone (UTC+8). Timestamps are
  interpreted as local civil time unless the private configuration explicitly
  converts them before loading.
- `GHI_real` is a historical observed-GHI array. It has exactly the same length
  as `Power`; element *i* of both arrays refers to the same observation time,
  including the final element.
- The bundled baseline treats `timestamp_win` as the forecast origin and the
  final history element as lag 1 (15 minutes before the origin). Solar factors
  reconstruct historical times with that convention. If the private dataset
  uses a different origin convention, stop and correct the data contract before
  enabling historical clear-sky factors.
- Issued future GHI, air temperature, wind speed, and wind direction remain
  forecast inputs. `GHI_real` is never shifted forward or used after the
  forecast origin.
- Future-weather trajectory and forecast clear-sky-index factors use only the
  issued array prefix ending at the scalar target horizon. They inherit the
  baseline availability contract; when `nwp_issue_time` is absent, record that
  availability as contract-assumed rather than timestamp-verified.

## Private configuration

Set `data.site_metadata.path` to the private CSV. Use `aliases` to map raw
parquet-row station names to the canonical `plantid`. Use `overrides` only for an
explicitly reviewed correction; an override replaces the CSV record and is
fingerprinted in the run manifest.

The loader rejects missing station metadata, non-positive capacity, invalid
coordinates, conflicting duplicates, non-Beijing timezones, and unequal
`Power`/`GHI_real` history lengths. Do not bypass these checks in a factor.

## Executable expert families

- Capacity and recurrence: normalized recent power and previous-day target-time
  power.
- Solar geometry: target-time solar position, daylight boundary, and Haurwitz
  clear-sky GHI.
- PV physics: recent power clear-sky index, Faiman module-temperature proxy,
  temperature-corrected irradiance, and low-irradiance state.
- Joint irradiance-power state: causal clear/variable/overcast descriptors,
  observed-to-forecast GHI ramp, clipping likelihood, and weather-power
  residual.
- Issued-weather dynamics: target-prefix GHI/temperature/wind changes and
  target-time forecast GHI normalized by deterministic clear-sky GHI.

The clear-sky and thermal formulas are bounded engineering proxies, not site
calibrations. Test each family separately before combining them. Clipping and
weather-power residual outputs are weak diagnostic scores; without operational
labels they must not be interpreted as confirmed curtailment or outage states.

## Protected distinction

Capacity-normalized input factors do not change the prediction target. A full
generation-coefficient target transformation changes training labels, inverse
transformation, clipping, and metrics; it requires a separate protected
protocol and must not be smuggled into an ordinary factor ablation.
