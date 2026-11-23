# Data layout

The reactor data are **not distributed with this repository**. They were collected in
laboratory-scale CSTR campaigns at the Leibniz Institute for Agricultural Engineering and
Bioeconomy (ATB) and are available from the corresponding author on request
(twollschlaeg@uos.de).

Copy the data package here, or anywhere else and point `ADM1_DATA_DIR` at it. It has the
layout below. Once it is in place, the tests marked `requires_data` run instead of
skipping (`pytest -m requires_data`).

```
data/
  rice-bran/                       # dataset A: defatted rice bran residues
    rice-bran-parameters.json      # mechanistic parameters and bounds of the campaign
    calibration.json               # the forcing they were fitted to
    input_R10.csv ... input_R90.csv        # three replicates per setup (R10-R30,
    output_R10.csv ... output_R90.csv      # R40-R60, R70-R90), averaged on load
    base-IVs_R10.csv ...           # calibrated initial states, full ADM1
    R2-IVs_R10.csv ...             # ... and per reduced variant
    R3-IVs_R10.csv ...
    R4-IVs_R10.csv ...
    calibration-backup/<time>/report.json  # the report of each calibration run
  LaRA-temp/                       # dataset B: temperature and loading changes
    LaRA-temp-parameters.json
    calibration.json
    input_F1.csv ... input_F6.csv
    output_F1.csv ... output_F6.csv
    {base,R2,R3,R4}-IVs_F1.csv ... _F6.csv
    calibration-backup/<time>/report.json
  LaRA-substrate/                  # dataset C: substrate substitution
    LaRA-substrate-parameters.json
    calibration.json
    input_F1.csv ... output_F6.csv, {base,R2,R3,R4}-IVs_F*.csv
    calibration-backup/<time>/report.json
```

The paper's names of the reactors are in the main [README](../README.md#names-in-the-paper-and-in-the-code).

## File conventions

**Sample identity.** A reactor is identified by the suffix of its `input_*.csv` file, and a
sample label is `<campaign>_<suffix>`, e.g. `LaRA-temp_F3`. Labels, not positional
indices, are what the split file records. A rice-bran setup is labelled by its last
replicate (`rice-bran_R30` for R10 to R30).

**Ordering matters.** Files are enumerated with a natural sort
(`adm1_package/utils/datasets.list_input_files`), never with raw `os.listdir`. For the
rice-bran campaign this is load-bearing: the loader averages each block of three
consecutive files into one replicate mean, so `R10, R20, R30` must arrive in that order.

**`input_*.csv`**: one row per day. Required columns:

| Column | Unit | Meaning |
|---|---|---|
| `time` | d | days since the start of the experiment |
| `inflow` | L/d | volumetric feed rate |
| `temperature` | °C | reactor temperature (converted to K on load) |
| `S_va`, `S_bu`, `S_pro`, `S_ac` | g/L | volatile fatty acids in the feed |
| `S_TAN` | g/L | total ammoniacal nitrogen in the feed |
| `S_h2o` | g/L | water content of the feed |
| `X_ch`, `X_pr`, `X_li` | g/L | carbohydrates, proteins and lipids in the feed |

Each variant reads the subset it needs: R4 takes `S_TAN`…`X_li`, R3 additionally `S_ac`,
R2 additionally `S_va`, `S_bu` and `S_pro`.

**`output_*.csv`**: measurements, one row per day. `time` plus any of `q_gasout` (NL/d),
`ch4_out` (NL/d), `S_ac`, `S_va`, `S_bu`, `S_pro`, `S_TAN`, `S_h2o` (g/L), `pH`, and the
total and volatile solids `TS` and `VS` (g/kg; `VS` = `TS (g/kg)` × `oTS (g/kg TS)` / 1000).
The gas flows are norm volumes, dry gas at 0 °C and 1013.25 hPa, as the campaigns'
workbooks report them; the models report the same. A day without a gas reading, on which
the reactor was not fed either, carries the rate of the next reading, and a reading
removed as faulty is blank. The scores skip both
(`adm1_package/utils/datasets.measured_points`). Gaps between weekly laboratory analyses
are filled on load with cubic (backward Hermite) interpolation, which is not monotone and
can overshoot between analyses; the loaders clip it at zero.

`TS` and `VS` are scored and drawn but never fitted, and training does not load them
(`ModelSpec.diagnostics`).

**`<variant>-IVs_<reactor>.csv`**: calibrated initial states, index in the first column
and values in the second. Missing files fall back to zeros. Measured components are
overwritten from the first row of `output_*.csv` regardless, and so are the gas phase
(from the first biogas and methane flows), the dissolved methane and CO2, which start in
equilibrium with it, and in R2/R3 the acid-base states and ions, set from the first
measured pH. What a file contributes is its biomass and particulates.

**`<campaign>-parameters.json`**: mechanistic parameters of the campaign, in the
container format of `adm1_package.utils.parameters.Parameters` (value, vary, min,
max per parameter), shared by the variants.

**`calibration-backup/<time>/report.json`**: the report of a calibration run, written by
`scripts/calibrate/calibrate_benchmarks.py` together with a backup of the files it
replaced: the losses, the bounds and the parameters before and after.
`scripts/benchmark_numbers.py` reads the reports. The models do not, and the backups
beside them are not needed.

**`calibration.json`**: written by `scripts/calibrate/calibrate_benchmarks.py` beside the
parameters: the forcing they were fitted to, `{"forcing": "smoothstep"}`. A model built
on the parameters is given the same forcing, and calibrations fitted to another forcing
are refused.

The forcing in `input_*.csv` is recorded once a day. Between two records the models are
given a smoothstep from one record to the next, through each at its time stamp
(`adm1_package/utils/forcing.forcing_coefficients`).
