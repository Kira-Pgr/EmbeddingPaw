# EmbeddingPaw

EmbeddingPaw is a Python library for playing with text embeddings. It provides a simple and intuitive interface for creating, manipulating, and visualizing embeddings using an OpenAI-like API.

## Installation

To install EmbeddingPaw, you can use pip:

```bash
python setup.py sdist bdist_wheel
pip install .
```

## Getting Started

To get started with EmbeddingPaw, you need to create an instance of the `EmbeddingPaw` class with your API configuration:

```python
from embeddingpaw import EmbeddingPaw

config = EmbeddingPaw(
    base_url="http://localhost:1234/v1",
    api_key="sk-xxxx",
    embedding_db_path="embeddings_db.pkl"
)
```
You can use LMStudio to create your own local embedding server. You can also use the OpenAI API by providing your API key.

## Creating Tokens

You can create a `Token` object by providing the text you want to embed:

```python
from embeddingpaw import Token

token = Token("Hello, world!")
```

The `Token` class automatically retrieves the embedding for the given text using the configured API.

Sure! Here's the updated part:

## Token Operations

EmbeddingPaw provides various operations that you can perform on `Token` objects:

- `get_similarity(token)`: Calculate the cosine similarity between two tokens.
- `get_closest_token(num=1)`: Find the closest token(s) in the embedding database.

Users can perform arithmetic operations on token embeddings using the following operators:
- Addition (`+`): Add the embeddings of two tokens.
- Subtraction (`-`): Subtract the embeddings of two tokens.
- Multiplication (`*`): Multiply the embeddings of two tokens.
- Division (`/`): Divide the embeddings of two tokens.
- Matrix Multiplication (`@`): Perform matrix multiplication on the embeddings of two tokens.

These arithmetic operations allow users to manipulate and combine token embeddings in meaningful ways.

Additionally, users can calculate the mean of two token embeddings using the `mean(other)` method.

## Token Arrays

You can create a `TokenArray` object to work with multiple tokens:

```python
from embeddingpaw import TokenArray

token_array = TokenArray([token1, token2, token3])
```

The `TokenArray` class provides methods for manipulating and analyzing the array of tokens:

- `append(token)`: Append a token to the array.
- `pop()`: Remove the last token from the array.
- `delete(text)`: Delete a token from the array based on its text.
- `pca(n_components=3)`: Apply Principal Component Analysis (PCA) to reduce the dimensionality of the embeddings.

## Visualizing Embeddings

EmbeddingPaw includes a `TokenVisualizer` class for visualizing token embeddings in a 3D scatter plot:

```python
from embeddingpaw import TokenVisualizer

visualizer = TokenVisualizer(token_array)
visualizer.show_web()  # Render the visualization in a web browser
visualizer.show_notebook()  # Render the visualization in a Jupyter notebook
```

## Embedding Database

The `EmbeddingPawDatabase` class allows you to manage and interact with an embedding database:

```python
from embeddingpaw import EmbeddingPawDatabase

db = EmbeddingPawDatabase()
```

The database provides methods for adding, deleting, and loading tokens:

- `add_token(token)`: Add a token to the database.
- `delete_token(text)`: Delete a token from the database based on its text.
- `load_token_from_txt(path)`: Load tokens from a text file.
- `load_token_from_json(path)`: Load tokens from a JSON file.
- `load_token_from_excel(path)`: Load tokens from an Excel file.

## Contributing

Contributions to EmbeddingPaw are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/Kira-Pgr/EmbeddingPaw).

## License

EmbeddingPaw is released under the [MIT License](https://opensource.org/licenses/MIT).