def poly(weights):
    return lambda x: sum([i * (x**id) for id, i in enumerate(weights)])
