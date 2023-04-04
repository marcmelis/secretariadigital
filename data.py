import csv
import re
from typing import List, Any

import tiktoken
import inflect
import pandas as pd


def get_exams_text() -> List[str]:

    FILE = "data/exams_schedule_1sem.csv"
    GRADE = "Computer Engineering"
    COLUMN_NAMES = ['subject', 'classroom', 'weekday', 'start time', 'end time', 'course', 'semester']
    phrase="The exam for {} will be held in the classroom {} during the exams week on {}. It will take place from {} to {} and it is part of the {} course and the {} semester of " + GRADE + "."
    engine = inflect.engine()


    with open(FILE, 'r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter='%')
        text = []
        for row in csv_reader:
            if 'course' in COLUMN_NAMES:
                index = COLUMN_NAMES.index('course')
                row[index] = engine.ordinal(row[index])
            if 'semester' in COLUMN_NAMES:
                index = COLUMN_NAMES.index('semester')
                row[index] = engine.ordinal(row[index])
            text.append(phrase.format(*row))

    return text

def get_professors_text() -> List[str]:

    FILE = "data/profs.csv"
    phrase="Professor {} works in the {} department. You can reach him via email at {}. His office is located at {}."

    with open(FILE, 'r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter='%')
        text = []
        for row in csv_reader:
            # Replace periods with commas in the desk parameter if it's not a number
            row[2]= re.sub(r'(?<=\D)\.(?=\D)', ',', row[2])
            sentence = phrase.format(row[0], row[3], row[1], row[2])
            # Delete double spaces
            sentence = re.sub(r'\s{2,}', ' ', sentence)
            text.append(sentence)

    return text

def get_schedules_text() -> List[str]:
    FILE = "data/timetables_1sem.csv"
    GRADE = "Computer Engineering"
    COLUMN_NAMES = ['subject', 'classroom', 'professor', 'weekday', 'start time', 'end time', 'course', 'semester']
    phrase="The class of {} will be held in the classroom {} and be lectured by the professor {} every {}. It will take place from {} to {} and it is part of the {} course and the {} semester of " + GRADE + "."
    engine = inflect.engine()

    with open(FILE, 'r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter='%')
        text = []
        for row in csv_reader:
            if 'course' in COLUMN_NAMES:
                index = COLUMN_NAMES.index('course')
                row[index] = engine.ordinal(row[index])
            if 'semester' in COLUMN_NAMES:
                index = COLUMN_NAMES.index('semester')
                row[index] = engine.ordinal(row[index])
            text.append(phrase.format(*row))
    return text

def get_text() -> List[str]:
    return get_professors_text() + get_schedules_text() + get_exams_text()

def list_to_dataframe(anylist: List[Any]) -> pd.DataFrame:
    df = pd.DataFrame(anylist, columns=['content'])
    df['tokens'] = df['content'].apply(num_tokens_from_string)
    return df

def list_to_csv(anylist: List[Any], filename: str):
    df = list_to_dataframe (anylist)
    df.to_csv(filename, index=False)

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('gpt2') #encoding for curie model and other openai models
    num_tokens = len(encoding.encode(string))
    return num_tokens
