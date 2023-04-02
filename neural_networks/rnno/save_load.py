import joblib


def save_params(params: dict, path):
    joblib.dump(params, path)


def load_params(path) -> dict:
    return joblib.load(path)
