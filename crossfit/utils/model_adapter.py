from typing import Any, Callable


def adapt_model_input(model: Callable, encoded_input: dict) -> Any:
    """
    Adapt the encoded input to the model, handling both single and multiple argument cases.

    This function allows flexible calling of different model types:
    - Models expecting keyword arguments (e.g., Hugging Face models)
    - Models expecting a single dictionary input (e.g., Sentence Transformers)

    :param model: The model function to apply
    :param encoded_input: The encoded input to pass to the model
    :return: The output of the model
    """
    try:
        # First, try to call the model with keyword arguments
        # For standard Hugging Face models
        return model(**encoded_input)
    except TypeError:
        # If that fails, try calling it with a single argument
        # This is useful for models like Sentence Transformers
        return model(encoded_input)
