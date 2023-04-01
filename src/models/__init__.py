from models.main_model import Model
models = {}


def register_model(name, factory):
    models[name] = factory


def model_create(config):
    name = config['name']
    if name in models:
        return models[name](config)
    else:
        raise BaseException("no such model:", name)

register_model('model', Model)
