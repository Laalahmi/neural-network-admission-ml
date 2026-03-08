from sklearn.neural_network import MLPClassifier


def get_candidate_models():
    """
    Return candidate neural network models for comparison.
    """
    models = {
        "mlp_small_relu": MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        ),
        "mlp_medium_relu": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        ),
        "mlp_medium_tanh": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="tanh",
            solver="adam",
            max_iter=500,
            random_state=42,
        ),
    }
    return models