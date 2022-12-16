import yaml


def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            loaded_line = json.loads(line)
            data.append(loaded_line)
    return data