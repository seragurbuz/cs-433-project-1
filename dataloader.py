from helpers import *
import csv
import os

REPLACEMENTS = {
    "PHYSHLTH": {88: 0, 77: 0, 99: 0},
    "MENTHLTH": {88: 0, 77: 0, 99: 0},
}

LIST_KEEP = [
    # General health
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    # Health care access
    "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1",
    # Classic cardiac risk factors
    "_RFHYPE5", "BPMEDS", "_CHOLCHK", "_RFCHOL", "DIABETE3",
    # Other chronic conditions
    "CVDSTRK3", "_ASTHMS1", "CHCSCNCR", "CHCOCNCR", "CHCCOPD1",
    "_DRDXAR1", "ADDEPEV2", "CHCKIDNY",
    # Demographics
    "SEX", "_AGE80", "MARITAL", "_EDUCAG", "_INCOMG", "RENTHOM1",
    "VETERAN3", "EMPLOY1", "_CHLDCNT", "INTERNET",
    # Body mass
    "_BMI5",
    # Functional limitations
    "QLACTLM2", "USEEQUIP", "BLIND", "DECIDE",
    "DIFFWALK", "DIFFDRES", "DIFFALON",
    # Lifestyle
    "_SMOKER3", "USENOW3", "_DRNKWEK", "_RFBING5",
    "_FRUTSUM", "_VEGESUM",
    "_TOTINDA", "PA1MIN_", "_PACAT1", "_PASTRNG",
    # Prevention
    "FLUSHOT6", "PNEUVAC3", "HIVTST6", "_RFSEAT2",
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
    This function removes certain columns from the given data, cleaning it up for use
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

def replace_values(data, replacements):
    pass

def count_values(data, keeplist, dictionnary):
    for i, name in enumerate(keeplist):
        if name in dictionnary:
            for value, replacement in dictionnary[name].items():
                count = np.sum(data[:, i] == value)
                print(f"{name}: {value} -> {replacement}, count: {count}")

if __name__ == "__main__":
    data_path = "dataset"
    LIST_KEEP = ["CELLFON3", "MENTHLTH"]
    indices_to_delete = get_delete_indices(os.path.join(data_path, "x_train.csv"), LIST_KEEP)
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
    x_train_clean, x_test_clean = prepare_input_data(x_train, x_test, indices_to_delete)
    print("x_train_clean shape:", x_train_clean.shape)
    print(x_train_clean[:10,:])
    mask = np.isnan(x_train_clean)
    print(mask.sum())
    print(mask.sum(axis=0))
    rows = np.where(mask[:, 0])[0]
    count_values(x_train_clean, LIST_KEEP, REPLACEMENTS)