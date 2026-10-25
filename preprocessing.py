
"""Clean BRFSS feature values using the codebook.

This module only applies fixed cleaning rules:
- Convert missing/refused codes to NaN.
- Convert encoded frequencies, durations, heights, weights and decimal scales.
- Remove invalid values.

Missing values are NOT filled here.
"""

import re
import numpy as np

from feature_rules import (
    FOOD_FREQUENCY,
    EXERCISE_FREQUENCY,
    DAILY_FREQUENCY,
    ALCOHOL_FREQUENCY,
    ENCODED_DURATION,
    EXERCISE_DURATION,
    DECIMAL_SCALES,
    SCALED_MISSING_FIELDS,
)


def codebook_codes(info):
    """Find valid ranges, missing codes and zero codes in the codebook."""

    ranges = []
    missing = []
    zeros = []

    for code, meaning in info["codes"]:
        # Codebook entries can be single values ("7")
        # or ranges ("101-199").
        match = re.fullmatch(
            r"\s*(\d+)\s*(?:-\s*(\d+))?\s*",
            code
        )

        if not match:
            continue

        lo = int(match[1])
        hi = int(match[2] or match[1])
        label = meaning.lower()

        # Unknown, refused and inapplicable responses are not measurements.
        if any(word in label for word in (
            "don't know", "not sure", "refused", "missing",
            "not applicable", "never heard", "no feet",
        )):
            if lo == hi:
                missing.append(lo)

        # Some numerical questions use special codes for zero
        elif (
            lo == hi
            and info["type"].startswith("numerical")
            and label.startswith((
                "none", "never", "zero",
                "no drinks", "did not drink"
            ))
        ):
            zeros.append(lo)

        else:
            ranges.append((lo, hi))

    return ranges, missing, zeros


def frequency_to_number(x, name):
    """Convert encoded frequencies to a common numerical unit.

    Food and daily-frequency features become occurrences per day.
    Exercise-frequency features become occurrences per week.
    """

    # Start with NaN so unrecognized codes remain missing.
    out = np.full(len(x), np.nan)

    if name in FOOD_FREQUENCY or name in DAILY_FREQUENCY:
        # Codes 101-199: times per day
        # Codes 201-299: times per week
        # Codes 301-399: times per month
        masks = [
            (101, 199, 100, 1),
            (201, 299, 200, 7),
            (301, 399, 300, 30),
        ]

        if name in DAILY_FREQUENCY:
            # Codes 401-499: times per year
            masks.append((401, 499, 400, 365))

        for lo, hi, base, divisor in masks:
            mask = (x >= lo) & (x <= hi)
            out[mask] = (x[mask] - base) / divisor

        # Food code 300 means less than once per month.
        # Approximate it as half an occurrence per month.
        if name in FOOD_FREQUENCY:
            out[x == 300] = 0.5 / 30

    else:
        # Exercise codes use weekly or monthly frequencies.
        weekly = (x >= 101) & (x <= 199)
        monthly = (x >= 201) & (x <= 299)

        out[weekly] = x[weekly] - 100
        out[monthly] = (x[monthly] - 200) * 12 / 52

    return out


def alcohol_days_per_week(x):
    """Decode ALCDAY5 responses to days per week."""
    out = np.full(len(x), np.nan)
    weekly = (x >= 101) & (x <= 107)
    monthly = (x >= 201) & (x <= 230)
    out[weekly] = x[weekly] - 100
    out[monthly] = (x[monthly] - 200) * 7 / 30
    return out


def duration_years(x):
    """Decode LONGWTCH day/week/month/year responses to years."""
    out = np.full(len(x), np.nan)
    for lo, hi, base, divisor in (
        (101, 199, 100, 365),
        (201, 299, 200, 52),
        (301, 399, 300, 12),
        (401, 499, 400, 1),
    ):
        mask = (x >= lo) & (x <= hi)
        out[mask] = (x[mask] - base) / divisor
    # 555 means "all my life"; its duration cannot be known from this column.
    return out


def exercise_minutes(x):
    """Decode EXERHMM1/2 HHMM responses to minutes."""
    out = np.full(len(x), np.nan)
    hours = x // 100
    minutes = x % 100
    valid = (x >= 1) & (x <= 959) & (minutes < 60)
    out[valid] = hours[valid] * 60 + minutes[valid]
    return out


def clean_column(raw, name, info):
    """Clean one feature using its BRFSS codebook definition.

    Returns a NumPy array of numerical values.
    Missing or invalid responses remain NaN.
    """

    # Work on a copy so the original data is not modified
    x = np.asarray(raw, dtype=float).copy()

    ranges, missing, zeros = codebook_codes(info)

    # Step 1: Replace unknown, refused and nonfinite values with NaN
    invalid = ~np.isfinite(x) | np.isin(x, missing)
    x[invalid] = np.nan

    # Step 2: Convert features requiring special interpretation
    if name in ALCOHOL_FREQUENCY:
        original = x.copy()
        x = alcohol_days_per_week(original)
        x[np.isin(original, zeros)] = 0

    elif name in ENCODED_DURATION:
        x = duration_years(x)

    elif name in EXERCISE_DURATION:
        x = exercise_minutes(x)

    elif name in (
        FOOD_FREQUENCY
        | EXERCISE_FREQUENCY
        | DAILY_FREQUENCY
    ):
        original = x.copy()
        x = frequency_to_number(original, name)
        x[np.isin(original, zeros)] = 0

    elif name == "WEIGHT2":
        # Standard values are pounds
        # Values starting with 9 encode kilograms
        pounds = (x >= 50) & (x <= 999)
        kilograms = (x >= 9000) & (x <= 9998)

        out = np.full(len(x), np.nan)
        out[pounds] = x[pounds] * 0.45359237
        out[kilograms] = x[kilograms] - 9000

        x = out

    elif name == "HEIGHT3":
        # BRFSS stores height in two possible formats:
        # 506 = 5 feet 6 inches
        # 9160 = 160 centimeters
        # Convert both formats to meters.

        feet = x // 100
        inches = x % 100

        imperial = (
            (x >= 200) & (x <= 711)
            & (inches < 12)
        )
        metric = (x >= 9000) & (x <= 9998)

        # Initialize the output array
        out = np.full(len(x), np.nan)

        # Feet/inches -> meters
        out[imperial] = (
            feet[imperial] * 12 + inches[imperial]
        ) * 0.0254

        # Centimeters -> meters
        out[metric] = (x[metric] - 9000) / 100

        # Remove implausible heights.
        out[(out < 0.9) | (out > 2.5)] = np.nan

        x = out

    else:
        # Step 3: Handle ordinary categorical/numerical features
        x[np.isin(x, zeros)] = 0

        if ranges:
            valid = np.zeros(len(x), dtype=bool)

            # Some numerical values have implied decimal places.
            factor = DECIMAL_SCALES.get(name, 1)

            if factor == 1:
                units = info.get("units", "").lower()

                for digits in (1, 2, 3):
                    if f"{digits} implied decimal" in units:
                        factor = 10 ** digits
                        break

            if name in SCALED_MISSING_FIELDS and factor != 1 and missing:
                # These CSV values are already scaled, including sentinels.
                x[np.isin(x, np.asarray(missing) / factor)] = np.nan

            # Accept valid codebook values.
            # For scaled variables, also accept already-scaled values.
            for lo, hi in ranges:
                valid |= (x >= lo) & (x <= hi)

                if factor != 1:
                    valid |= (
                        (x * factor >= lo)
                        & (x * factor <= hi)
                    )

            if zeros:
                valid |= x == 0

            x[~valid] = np.nan

            # Convert raw values only when their scale is clear.
            if factor != 1 and name in {
                "_BMI5", "HTM4", "WTKG3", "_DRNKWEK", "DROCDY3_"
            }:
                thresholds = {
                    "_BMI5": 100,
                    "HTM4": 10,
                    "WTKG3": 300,
                    "_DRNKWEK": 0,
                    "DROCDY3_": 0,
                }

                x[x > thresholds[name]] /= factor

        if name == "ASTHMAGE":
            # 97 is the special response "age 10 or younger".
            x[x == 97] = 10

    return x


def clean_data(data, feature_names, codebook):
    """Apply BRFSS cleaning rules to every selected feature.

    Input and output have the same shape and column order.
    No columns are added or removed.
    """

    cleaned = np.empty(data.shape, dtype=float)

    for j, name in enumerate(feature_names):
        cleaned[:, j] = clean_column(
            data[:, j],
            name,
            codebook[name]
        )

    return cleaned

    
def fit_imputation(x):
    """Learn column medians from training data only."""
    medians = np.nanmedian(x, axis=0)

    # Columns containing only NaN cannot have a median.
    medians = np.where(np.isfinite(medians), medians, 0.0)

    return medians


def apply_imputation(x, medians):
    """Replace missing values using previously learned medians."""
    return np.where(np.isfinite(x), x, medians)

