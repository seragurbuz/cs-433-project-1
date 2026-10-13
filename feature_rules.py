
"""Feature selection and special preprocessing rules for BRFSS data."""

# Features that we do not want to use for prediction.
# These include survey identifiers, dates, sampling weights,
# redundant variables, and fields that may cause target leakage.
#
# All features not in this set will be kept.
EXCLUDED_FIELDS = set("""
_STATE FMONTH IDATE IMONTH IDAY IYEAR DISPCODE SEQNO _PSU QSTVER QSTLANG
MSCODE _STSTR _STRWT _RAWRAKE _WT2RAKE _LLCPWT _DUALUSE _DUALCOR
_CLLCPWT HAREHAB1 CTELENUM PVTRESD1 COLGHOUS STATERES CELLFON3
LADULT CTELNUM1 CELLFON2 CADULT PVTRESD2 CCLGHOUS CSTATE LANDLINE
NUMHHOL2 NUMPHON2 CPDEMO1 RCSGENDR RCSRLTN2 CASTHDX2 CASTHNO2
_CHISPNC _CRACE1 _CPRACE _RFHLTH _RFHYPE5 _CHOLCHK _RFCHOL
_LTASTH1 _CASTHM1 _ASTHMS1 _DRDXAR1 _CHLDCNT _EDUCAG _INCOMG
_RFSMOK3 DRNKANY5 _RFBMI5 _TOTINDA _RFSEAT2 _RFSEAT3 _AIDTST3
_MISFRTN _MISVEGN _FRTRESP _VEGRESP _FRT16 _VEG23 _FRUITEX
_VEGETEX PAMISS1_
""".split())


# Some BRFSS questions store both frequency and time unit in one number.
# For example, FRUIT1 = 101 means once per day,
# while FRUIT1 = 201 means once per week.
# These features need conversion to a common time unit.
FOOD_FREQUENCY = set(
    "FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1".split()
)

EXERCISE_FREQUENCY = set(
    "EXEROFT1 EXEROFT2 STRENGTH".split()
)

DAILY_FREQUENCY = set(
    "BLDSUGAR FEETCHK2".split()
)

# 1xx means days per week; 2xx means days in the past 30 days.
ALCOHOL_FREQUENCY = {"ALCDAY5"}

# 1xx/2xx/3xx/4xx mean days/weeks/months/years.
ENCODED_DURATION = {"LONGWTCH"}

# Hours and minutes are stored together as HHMM.
EXERCISE_DURATION = {"EXERHMM1", "EXERHMM2"}


# These features contain encoded dates.
# Their raw values should not be treated as continuous numerical values.
# For now, we treat them as categorical features.
FORCE_CATEGORICAL = {
    "FLSHTMY2",
    "HIVTSTD3",
}


# Some BRFSS numerical variables use implied decimal places.
# For example, a raw BMI value of 2534 represents 25.34.
#
# Some columns may already be scaled in our dataset.
# The preprocessing code must check the actual values before dividing.
DECIMAL_SCALES = {
    "_BMI5": 100,
    "HTM4": 100,
    "WTKG3": 100,
    "_DRNKWEK": 100,
    "DROCDY3_": 100,
}


def selected_features(codebook):
    """Select the 250 predictors that are not excluded."""

    names = [
        name for name in codebook
        if name not in EXCLUDED_FIELDS
    ]

    # Make sure the feature selection has not changed unexpectedly.
    if len(names) != 250:
        raise ValueError(
            f"Expected 250 predictors, found {len(names)}"
        )

    return names
