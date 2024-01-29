import json
from tqdm import tqdm

def open_txt_file(file):
    entities = []

    for line in open(file).readlines():
        entities.append(line)

    return entities


def open_file(file):
    entities = []

    for line in open(file).readlines():
        entities.append(json.loads(line))

    return entities


def save_file(data, file):
    file_writer = open(file, 'w')

    for line in data:
        file_writer.write(json.dumps(line) + "\n")


def get_prompt(flesch_summary):
    if flesch_summary >= 80:
        prompt = 'Write highlights for this article for a 11 years old student:\n\n'
    elif 80 > flesch_summary >= 60:
        prompt = 'Write highlights for this article for a middle school student:\n\n'
    elif 60 > flesch_summary >= 40:
        prompt = 'Write highlights for this article for a high school student:\n\n'
    else:
        prompt = 'Write highlights for this article for a college student:\n\n'
    return prompt


def transform_data(split):
    data = open_file('../data/' + split + '.json')
    new_data = []

    for entry in tqdm(data):

        flesch_summary = entry["summary_metrics"]["flesch"]

        prompt = get_prompt(flesch_summary)
        entry["prompt"] = prompt
        entry["input_noprompt"] = entry["input"]
        entry["input"] = prompt + entry["input"]
        new_data.append(entry)

    save_file(new_data, '../data/' + split + '_prompt_category.json')


transform_data('train')
transform_data('validation')
transform_data('test')







