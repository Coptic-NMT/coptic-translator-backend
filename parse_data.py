import json
import regex as re
import pandas as pd
import random
import os
import string
import datasets

UROMAN_PATH = "./uroman/bin/uroman.pl"

illegal = set(string.ascii_lowercase + string.ascii_uppercase)

def clean_empty_norms(df):
    return df[~df["norm"].str.contains("_warn:empty_norm_")]

def first_illegal(s):
    for i, c in enumerate(s):
        if c in illegal:
            return i, c

def invalid(s):
    return any(c in illegal for c in s)

def filter_no_text_cols(df, cols):
    def has_no_text(string):
        return not any(c.isalnum() for c in string)
    for col in cols:
        df = df[~df[col].apply(has_no_text)]
    return df

# get dataframe of all rows with invalid norm
def get_invalid(df):
    return df[df["norm"].apply(invalid)]

def find_invalid(string: str):
    for index, c in enumerate(string):
        if c in illegal:
            return index, c

def parse_data_to_csv(data_path: str, out_path: str):
    data_dict = {"translation": [], "norm_group": [], "norm": [], "func": [], "pos": [], "arabic": [], "meta::translation": [], "meta::title": [], "meta::source": [], "meta::corpus": []}

    with open(data_path) as f:
        data = f.read()
        data = re.sub(r"^\d+\.", '', data, flags=re.MULTILINE)
        chunks = data.split("\n\n\n")
        for chunk in chunks:
            lines = chunk.split("\n")
            row = {}
            for line in lines:
                header, entry = line.strip().split("\t")
                print(line)
                row[header] = entry.strip()
                
            for column in data_dict:
                data_dict[column].append(row.get(column, None))
            
    df = pd.DataFrame(data_dict)
    df.to_csv(out_path)

def filter_bracketed_periods(df: pd.DataFrame):
    for col in ["norm", "norm_group", "translation"]:
        df = df[~df[col].str.contains(r"\[\.*\]", regex=True)]
    return df

def filter_dots_and_ellipsis(df: pd.DataFrame, columns=["translation"]):
    pattern = r"(…)|(\.\.+)"
    for col in columns:
        df = df[~df[col].str.contains(pattern, regex=True)]
    return df

def replace_dots_and_ellipsis(df: pd.DataFrame):
    pattern = r"(\. [\. ]+)|(…)"
    df = df.copy(deep=True)
    df["norm"] = df["norm"].str.replace(pattern, '', regex=True)
    df["norm_group"] = df["norm_group"].str.replace(pattern, '', regex=True)
    return df

def filter_parenthesis(df: pd.DataFrame):
    df = df[~df["translation"].str.contains("\(|\)")]
    return df

def filter_verses(df: pd.DataFrame):
    pattern = r'\([^)]*\d+:\d+[^)]*\)'
    df = df.copy(deep=True)
    df['translation'] = df['translation'].str.replace(pattern, '', regex=True)
    return df

def fix_periods(df: pd.DataFrame):
    df = df.copy(deep=True)
    df["norm"] = df["norm"].str.replace("·", '.')
    df["norm_group"] = df["norm_group"].str.replace("·", '.')
    return df

def remove_bullshit(df: pd.DataFrame, columns: list[str]):
    # Remove bullshit special characters (like …, etc.)
    allowed_special = [" ", "."]
    df = df.copy(deep=True)
    for col in columns:
        df[col] = df[col].apply(lambda x: ''.join(c for c in x if c.isalnum() or c in allowed_special))
    return df

def drop_empty(df: pd.DataFrame, columns: list[str]):
    df = df.copy(deep=True)
    for col in columns:
        df = df[~df[col].str.contains(r"^\s*$", regex=True)]
    return df

def unnormalize(df: pd.DataFrame):
    def unnormalize_string(string):
        return ''.join(c for c in string if c.isalnum())
    df = df.copy(deep=True)
    df["unnormalized"] = df["norm"].apply(unnormalize_string)
    return df

def align_periods(df: pd.DataFrame, columns: list[str]):
    df = df.copy(deep=True)
    for col in columns:
        df[col] = df[col].str.replace(r" \. ", ". ", regex=True)
    return df

def to_translation_json(df: pd.DataFrame, src_column, tgt_column, src_language, tgt_language):
    translations = []
    # loop through rows
    for _, row in df.iterrows():
        entry = {}
        for column in row.keys():
            if column in [src_column, tgt_column]:
                continue
            entry[column] = row[column]

        src = row[src_column]
        tgt = row[tgt_column]

        entry["translation"] = {src_language: row[src_column], tgt_language: row[tgt_column]}
        translations.append(entry)
    with open('translation.json', 'w') as f:
        json.dump(translations, f, indent=2)

def only_sentences_without_translations(df: pd.DataFrame):
    df = df[(df["translation"] == "…") | (df["translation"] == "\.\.\.")]
    return df

def romanize_columns(df: pd.DataFrame, columns: list[str], tmpDir="./tmp"):
    if set(columns) & set(df.columns) != set(columns):
        raise ValueError("Not all columns in dataframe")
    
    if not os.path.exists(UROMAN_PATH):
        raise ValueError("Uroman not found")

    tmpDir = "./tmp"
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)
    
    df = df.copy()
    df = df.reset_index()
    for col in columns:
        current = df[col]
        current.to_csv(os.path.join(tmpDir, "col.csv"), index=False)
        os.system(f"{UROMAN_PATH} < {tmpDir}/col.csv > {tmpDir}/rom.csv")
        # Set column to romanized column
        df[col + "_romanized"] = pd.read_csv(os.path.join(tmpDir, "rom.csv"))
        os.remove(os.path.join(tmpDir, "rom.csv"))
        os.remove(os.path.join(tmpDir, "col.csv"))
    return df

def greekify_columns(df: pd.DataFrame, columns: list[str]):
    df = df.copy()
    for col in columns:
        df[col + "_greekified"] = df[col].apply(greekify)
    return df



COPTIC_TO_GREEK = {
    "ⲁ": "α",
    "ⲃ": "β",
    "ⲅ": "γ",
    "ⲇ": "δ",
    "ⲉ": "ε",
    "ⲋ": "ϛ",
    "ⲍ": "ζ",
    "ⲏ": "η",
    "ⲑ": "θ",
    "ⲓ": "ι",
    "ⲕ": "κ",
    "ⲗ": "λ",
    "ⲙ": "μ",
    "ⲛ": "ν",
    "ⲝ": "ξ",
    "ⲟ": "ο",
    "ⲡ": "π",
    "ⲣ": "ρ",
    "ⲥ": "σ",
    "ⲧ": "τ",
    "ⲩ": "υ",
    "ⲫ": "φ",
    "ⲭ": "χ",
    "ⲯ": "ψ",
    "ⲱ": "ω",
    "ϣ": "s",
    "ϥ": "f",
    "ϧ": "k",
    "ϩ": "h",
    "ϫ": "j",
    "ϭ": "c",
    "ϯ": "t",   
}

GREEK_TO_COPTIC = {
    greek: coptic for coptic, greek in COPTIC_TO_GREEK.items()
}

COPTIC_TO_ROMAN = {
    'ⲁ': 'a',
    'ⲃ': 'v',
    'ⲅ': 'g',
    'ⲇ': 'd',
    'ⲉ': 'eie',
    'ⲋ': 's',
    'ⲍ': 'z',
    'ⲏ': 'h',
    'ⲑ': 'th',
    'ⲓ': 'iau',
    'ⲕ': 'k',
    'ⲗ': 'l',
    'ⲙ': 'm',
    'ⲛ': 'n',
    'ⲝ': 'ks',
    'ⲟ': 'o',
    'ⲡ': 'p',
    'ⲣ': 'r',
    'ⲥ': 's',
    'ⲧ': 't',
    'ⲩ': 'ua',
    'ⲫ': 'f',
    'ⲭ': 'kh',
    'ⲯ': 'ps',
    'ⲱ': 'oou',
    'ϣ': 'sh',
    'ϥ': 'f',
    'ϧ': 'kh',
    'ϩ': 'h',
    'ϫ': 'g',
    'ϭ': 'sh',
    'ϯ': 'd'
 }

def greekify(coptic_text: str):
    chars = []
    for c in coptic_text:
        chars.append(COPTIC_TO_GREEK.get(c, c))
    return "".join(chars)

def coptify(greek_text):
    chars = []
    for c in greek_text:
        chars.append(GREEK_TO_COPTIC.get(c, c))
    return "".join(chars)

def romanize(coptic_text):
    chars = []
    for c in coptic_text:
        l_c = c.lower()
        chars.append(COPTIC_TO_ROMAN.get(l_c, l_c))
    return "".join(chars)

# def save_dataset(data: pd.DataFrame, src_lang, tgt_lang, src_col, tgt_col, data_path):
#     if os.path.exists(data_path):
#         raise FileExistsError("Data path already exists")
#     dataset = datasets.Dataset.from_pandas(data)
#     # add translation column
#     dataset = dataset.map(lambda x: {"translation": {src_lang: x[src_col], tgt_lang: x[tgt_col]}})
#     dataset.shuffle(seed=42)
#     temp_data_dict = dataset.train_test_split(test_size=128, seed=42)
#     data_dict = temp_data_dict['train'].train_test_split(test_size=500, seed=42)
#     data_dict['validation'] = temp_data_dict['test']
#     data_dict.save_to_disk(data_path)


def load_dataset_for_translation(dataset_path: str, src_lang: str, tgt_lang: str, src_col: str, tgt_col: str):
    data_dict = datasets.load_from_disk(dataset_path)
    for split in data_dict:
       data_dict[split] = data_dict[split].map(lambda x: {"translation": {src_lang: x[src_col], tgt_lang: x[tgt_col]}})
    return data_dict

def load_test_set(dataset_path: str, src_lang: str, tgt_lang: str, src_col: str, tgt_col: str):
    dataset = datasets.Dataset.from_csv(dataset_path)
    dataset = dataset.map(lambda x: {"translation": {src_lang: x[src_col], tgt_lang: x[tgt_col]}})
    dataset.data_dir = dataset_path
    return dataset