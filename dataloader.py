from helpers import *
import csv
import os
REPLACEMENTS = {
    "GENHLTH": {7: -1, 8: -1, 9: -1, np.nan: -1},
    "PHYSHLTH": {88: 0, 77: "median", 99: "median", np.nan: "median"},
    "MENTHLTH": {88: 0, 77: "median", 99: "median", np.nan: "median"},
    "POORHLTH": {88: 0, 77: "median", 99: "median", np.nan: "median"},
    "BPHIGH4": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "BPMEDS": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "TOLDHI2": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "CVDSTRK3": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "CHCSCNCR": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "CHCCOPD1": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "HAVARTH3": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "CHCKIDNY": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "DIABETE3": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "INCOME2": {77: "median", 88: "median", 99: "median", np.nan: "median"},
    "EXEROFT1": {777: "median", 888: "median", 999: "median", np.nan: "median"},
    "_ASTHMS1": {9: -1},
    "_RACE": {9: -1},
    "_BMI5": {np.nan: "median"},
    "_SMOKER3": {9: -1},
    "_DRNKWEK": {99900: -1},
    "_FRUTSUM": {np.nan: "median"},
    "_VEGESUM": {np.nan: "median"},
    "_AGEG5YR": {14: "median"},
    "DIFFDRES": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "DIFFALON": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "DIFFWALK": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "EXERANY2": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "RDUCHART": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "RDUCSTRK": {7: 0, 8: 0, 9: 0, np.nan: 0},
    "FLUSHOT6": {7: 0, 8: 0, 9: 0, np.nan: 0},
}

LIST_KEEP = [
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH", "BPHIGH4", "BPMEDS",
    "TOLDHI2", "CVDSTRK3", "CHCSCNCR", "CHCCOPD1", "HAVARTH3", "CHCKIDNY",
    "DIABETE3", "SEX", "INCOME2", "EXEROFT1", "_ASTHMS1", "_RACE", "_BMI5",
    "_SMOKER3", "_DRNKWEK", "_FRUTSUM", "_VEGESUM", "_AGEG5YR", "DIFFDRES",
    "DIFFALON", "DIFFWALK", "EXERANY2", "RDUCHART", "RDUCSTRK", "FLUSHOT6",
]

def get_delete_indices(csv_path, keep_list):
    with open(csv_path) as f:
        header = next(csv.reader(f))[1:]
    missing = [n for n in keep_list if n not in header]
    if missing:
        raise ValueError(f"Not in header: {missing}")
    return [i for i, name in enumerate(header) if name not in keep_list]

def clean_data(data, columns):
    """
    This function removes certain columns from the given data, 
    cleaning it up for use.
    """
    return np.delete(data, columns, axis=1)

def prepare_input_data(x_train, x_test, columns):
    """
    This function cleans and prepares the input data
    example use:
    
    data_path = "dataset"
    to_clean = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    x_train_raw, x_test_raw, y_train_raw, train_ids, test_ids = load_csv_data(data_path, sub_sample=False)
    x_train, x_test = prepare_input_data(x_train_raw, x_test_raw, to_clean)
    """
    return clean_data(x_train, columns), clean_data(x_test, columns)

def fit_replacements(reference_data, header, replacements):
    """
    Learn replacement values from training rows only.
    Apply fixed replacements first; exclude missing codes from means/medians.
    """
    header = list(header)
    fitted = {}
    for name, rules in replacements.items():
        if name not in header:
            continue
        col = reference_data[:, header.index(name)].astype(float).copy()
        masks = {old: np.isnan(col) if np.isnan(old) else col == old
                 for old in rules}
        for old, new in rules.items():
            col[masks[old]] = np.nan if isinstance(new, str) else new
        fitted[name] = {}
        for old, new in rules.items():
            if isinstance(new, str):
                if new not in ("mean", "median"):
                    raise ValueError(f"Unknown replacement statistic: {new}")
                valid = col[np.isfinite(col)]
                if not valid.size:
                    raise ValueError(f"No observed fitting values for {name}")
                new = float(np.mean(valid) if new == "mean" else np.median(valid))
            fitted[name][old] = new
    return fitted


def replace_values(data, header, replacements, reference_data=None):
    """
    Replace values using the given rules.
    For validation, reuse rules learned from training rows.
    """
    header = list(header)
    if any(isinstance(new, str) for rules in replacements.values()
           for new in rules.values()):
        replacements = fit_replacements(
            data if reference_data is None else reference_data, header, replacements
        )
    result = data.astype(float).copy()
    for name, rules in replacements.items():
        if name not in header:
            continue
        index = header.index(name)
        original = data[:, index]
        for old, new in rules.items():
            mask = np.isnan(original) if np.isnan(old) else original == old
            result[mask, index] = new
    return result

def count_values(data, keeplist, dictionnary):
    """
    This fonction counts the number of occurrences of each value
    in the specified columns of the data, according to the given 
    dictionary of replacements.
    """
    for i, name in enumerate(keeplist):
        if name in dictionnary:
            for value, replacement in dictionnary[name].items():
                if np.isnan(value):
                    count = np.sum(np.isnan(data[:, i]))
                else:
                    count = np.sum(data[:, i] == value)
                print(f"{name}: {value}, count: {count}")

def save_csv_data(savedatapath, data, filename, keepList):
    """
    This function saves the given data to a CSV file with the specified filename.
    It adds a header line with the column names from the keeplist.
    """
    with open(os.path.join(savedatapath, filename), 'w') as f:
        f.write(','.join(keepList) + '\n')
        np.savetxt(f, data, delimiter=',', fmt='%s')

if __name__ == "__main__":
    # Load the data
    data_path = "dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

    # Get the indices of the columns to delete based on the LIST_KEEP
    indices_to_delete = get_delete_indices(os.path.join(data_path, "x_train.csv"), LIST_KEEP)

    # Get the data of interest
    x_train_clean, x_test_clean = prepare_input_data(x_train, x_test, indices_to_delete)
    with open(os.path.join(data_path, "x_train.csv")) as source:
        header = next(csv.reader(source))[1:]
    # Deleting columns preserves the CSV order, not necessarily LIST_KEEP order.
    LIST_KEEP = [name for i, name in enumerate(header) if i not in indices_to_delete]
    print("x_train_clean shape:", x_train_clean.shape)

    # Check for NaN values in the cleaned training data
    mask = np.isnan(x_train_clean)
    print(mask.sum(axis=0))

    # Replace the values in the dictionnary with the specified replacements
    replaced_data = replace_values(x_train_clean, LIST_KEEP, REPLACEMENTS)
    replaced_test_data = replace_values(
        x_test_clean, LIST_KEEP, REPLACEMENTS, reference_data=x_train_clean
    )

    count_values(x_train_clean, LIST_KEEP, REPLACEMENTS)
    count_values(replaced_data, LIST_KEEP, REPLACEMENTS)
    print(x_train_clean[:10,:])
    print(replaced_data[:10,:])

    # Save the cleaned and replaced data to CSV files
    data_path = "dataset/cleaned_data"
    os.makedirs(data_path, exist_ok=True)
    save_csv_data(data_path, x_train_clean, "x_train_clean.csv", LIST_KEEP)
    save_csv_data(data_path, replaced_data, "x_train_replaced.csv", LIST_KEEP)
    save_csv_data(data_path, x_test_clean, "x_test_clean.csv", LIST_KEEP)
    save_csv_data(data_path, replaced_test_data, "x_test_replaced.csv", LIST_KEEP)
