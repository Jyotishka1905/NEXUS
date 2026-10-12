from pathlib import Path

import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "raw"
    / "forecast_history"
)


files = sorted(
    list(DATA_DIR.glob("event_*.nc"))
    + list(DATA_DIR.glob("control_*.nc"))
)


print("NEXUS Forecast History Verification")
print("------------------------------------")
print(f"Files found: {len(files)}")
print()


valid = 0
invalid = 0


for path in files:

    try:

        with xr.open_dataset(path) as ds:

            if "valid_time" not in ds.sizes:
                raise RuntimeError(
                    "missing valid_time dimension"
                )

            hours = ds.sizes["valid_time"]

            if hours != 168:
                raise RuntimeError(
                    f"expected 168 hours, found {hours}"
                )

            if "t2m" not in ds.data_vars:
                raise RuntimeError(
                    "missing t2m"
                )

            if "tp" not in ds.data_vars:
                raise RuntimeError(
                    "missing tp"
                )

            if ds["t2m"].isnull().any():
                raise RuntimeError(
                    "t2m contains NaN"
                )

            if ds["tp"].isnull().any():
                raise RuntimeError(
                    "tp contains NaN"
                )

            valid += 1

    except Exception as exc:

        invalid += 1

        print(
            f"INVALID: {path.name} -> {exc}"
        )


print()
print("Verification summary")
print("--------------------")
print(f"Valid:   {valid}")
print(f"Invalid: {invalid}")
print(f"Total:   {len(files)}")