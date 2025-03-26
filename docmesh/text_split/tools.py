import tiktoken


def default_length_function(text: str, encoding_name: str = "gpt2") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    num_tokens = len(tokens)

    return num_tokens
