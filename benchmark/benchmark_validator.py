import json


def get_first_n_data_sets(json_data, n):
    if n <= 0:
        return []

    if isinstance(json_data, list):
        return json_data[:n]

    if isinstance(json_data, dict):
        if "data" in json_data and isinstance(json_data["data"], list):
            return json_data["data"][:n]
        return [{key: value} for key, value in list(json_data.items())[:n]]

    return []


def extract_key_structure(item):
    if isinstance(item, dict):
        return {key: extract_key_structure(value) for key, value in item.items()}

    if isinstance(item, list):
        if not item:
            return []
        return [extract_key_structure(item[0])]

    return None


with open("benchmark/train_separate_questions.json") as f:
    data = json.load(f)


limit = 2

print(type(data))
first_n_data = get_first_n_data_sets(data, limit)
print(f"First {limit} data sets (keys only):")
for i, item in enumerate(first_n_data, start=1):
    print(f"{i}.")
    print(json.dumps(extract_key_structure(item), ensure_ascii=False, indent=2))
