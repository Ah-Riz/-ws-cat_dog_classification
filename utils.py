def convert_params(params):
    for key, value in params.items():
        try:
            params[key] = int(value)
        except ValueError:
            pass
        except TypeError:
            pass
    return params