#%%
# import libraries
import numpy as np
from sklearn.decomposition import PCA
from openai import OpenAI
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from pyecharts import options as opts
from pyecharts.charts import Scatter3D
from pyecharts.globals import ThemeType
import os
import http.server
import socketserver
import webbrowser
import threading
import json
import pandas as pd


class EmbeddingPaw:
    """
    Initialize the configuration for the EmbeddingPaw library
    Example Usage:
    config = EmbeddingPaw(
        base_url="http://localhost:1234/v1",
        api_key="sk-xxxx",
        embedding_db_path="embeddings_db.pkl"
    )
    :param base_url: The base URL of OpenAI-like embedding API
    :param api_key: The API key
    :param embedding_db_path: The path to the embedding database (pickle file)
    """
    def __init__(self, base_url: str, api_key: str, embedding_db_path: str):
        self.base_url = base_url
        self.api_key = api_key
        self.embedding_db_path = embedding_db_path
        self.set_config()
        
    def set_config(self):
        Token.set_config(self)
        EmbeddingPawDatabase.set_config(self)


class Token:
    _config = None

    @classmethod
    def set_config(cls, config: EmbeddingPaw):
        cls._config = config

    def __init__(self, text: str, embedding: np.array = None):
        if Token._config is None:
            raise ValueError("Config must be set before creating a TokenPaw instance.")
        
        self.text = text
        self.client = OpenAI(base_url=Token._config.base_url, api_key=Token._config.api_key)
        self.embedding = embedding if embedding is not None else self.__get_embedding(text)
        self.db = self.__load_embedding_db(Token._config.embedding_db_path)

    
    def __get_embedding(self, text: str, model="nomic-ai/nomic-embed-text-v1.5-GGUF") -> np.array:
        """
        Get the embedding of the text using API
        :param text: The target text
        :param model: The model to use
        :return: The embedding of the text
        """
        text = text.replace("\n", " ")
        embedding = self.client.embeddings.create(input=[text], model=model).data[0].embedding
        return np.array(embedding)
    
    @staticmethod
    def __load_embedding_db(path: str) -> dict:
        """
        Load the embedding database from a file
        :param path: The path to the embedding database
        :return: The embedding database
        """
        path = os.path.expanduser(path)
        with open(path, 'rb') as f:
            embeddings_db = pickle.load(f)
        return embeddings_db
    
    def __repr__(self):
        return f"<Token(text='{self.text}')>"
    
    def get_similarity(self, token) -> float:
        """
        Get the similarity between two tokens
        :param token: The target token
        :return: The similarity between two tokens
        """
        return cosine_similarity([self.embedding], [token.embedding])[0][0]
        
    def get_closest_token(self, num=1):
        """
        Get the closest token to the given embedding
        :param num: The number of closest tokens to return
        :return: The closest n tokens
        """
        tok_list = []
        for word, embedding in self.db.items():
            similarity = cosine_similarity([self.embedding], [embedding])[0][0]
            tok_list.append((word, similarity))
        tok_list.sort(key=lambda x: x[1], reverse=True)
        closest_tokens = [Token(word, self.db[word]) for word, _ in tok_list[:num]]
        return closest_tokens
    
    def __add__(self, other):
        new_embedding = self.embedding + other.embedding
        new_token = Token(self.text, new_embedding)
        closest_tokens = new_token.get_closest_token(5)
        for token in closest_tokens:
            if token.text != self.text and token.text != other.text:
                return token
    
    def __sub__(self, other):
        new_embedding = self.embedding - other.embedding
        new_token = Token(self.text, new_embedding)
        closest_tokens = new_token.get_closest_token(5)
        for token in closest_tokens:
            if token.text != self.text and token.text != other.text:
                return token
            
    def __mul__(self, other):
        new_embedding = self.embedding * other.embedding
        new_token = Token(self.text, new_embedding)
        closest_tokens = new_token.get_closest_token(5)
        for token in closest_tokens:
            if token.text != self.text and token.text != other.text:
                return token
            
    def __truediv__(self, other):
        new_embedding = self.embedding / other.embedding
        new_token = Token(self.text, new_embedding)
        closest_tokens = new_token.get_closest_token(5)
        for token in closest_tokens:
            if token.text != self.text and token.text != other.text:
                return token
              
    def __matmul__(self, other):
        """
        Define the behavior of the `@` operator for matrix multiplication (dot product).
        :param other: The target token
        :return: A new token closest to the resulting embedding
        """
        result = np.dot(self.embedding, other.embedding)
        new_embedding = (self.embedding + other.embedding) / 2 * result
        new_token = Token(self.text, new_embedding)
        closest_tokens = new_token.get_closest_token(5)
        for token in closest_tokens:
            if token.text != self.text and token.text != other.text:
                return token
            
            
    def mean(self, other):
        new_embedding = (self.embedding + other.embedding) / 2
        new_token = Token(self.text, new_embedding)
        closest_tokens = new_token.get_closest_token(5)
        for token in closest_tokens:
            if token.text != self.text and token.text != other.text:
                return token


class TokenArray:
    def __init__(self, tokens: list[Token]):
        self.length = len(tokens)
        self.tokens = tokens
        self.text = [token.text for token in tokens]
        self.embeddings = np.array([token.embedding for token in tokens])
        self.pca_res = None
    
    def __getitem__(self, index):
        return self.tokens[index]
    
    def append(self, token: Token):
        """
        Append a token to the array.
        :param token: The token to append.
        """
        self.tokens.append(token)
        self.text.append(token.text)
        self.embeddings = np.append(self.embeddings, [token.embedding], axis=0)
        self.length += 1
    
    def pop(self):
        """
        Remove the last token from the array.
        """
        self.tokens.pop()
        self.text.pop()
        self.embeddings = self.embeddings[:-1]
        self.length -= 1
    
    def delete(self, text):
        """
        Delete a token from the array.
        :param text: The text of the token to delete.
        """
        index = self.text.index(text)
        self.tokens.pop(index)
        self.text.pop(index)
        self.embeddings = np.delete(self.embeddings, index, axis=0)
        self.length -= 1
    
    def pca(self, n_components=3):
        """
        Apply PCA to reduce the embeddings to n_components dimensions.
        :param n_components: Number of dimensions to reduce to.
        :return: Reduced embeddings as a numpy array.
        """
        if self.length < n_components:
            raise ValueError("PCA Failed: Please add more tokens.")
        else:
            pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(self.embeddings)
        return reduced_embeddings
    
    def __repr__(self):
        class_name = self.__class__.__name__
        num_tokens = len(self.tokens)
        tokens_preview = ', '.join(self.text[:3]) + ('...' if num_tokens > 3 else '')
        return f"<{class_name}(num_tokens={num_tokens}, tokens=[{tokens_preview}])>"


class TokenVisualizer:
    def __init__(self, token_array: TokenArray):
        token_array.pca_res = token_array.pca()
        self.token_array = token_array
        self.data = self._prepare_data()

    def _prepare_data(self):
        """
        Prepare the data for the scatter plot.
        """
        reduced_embeddings = self.token_array.pca_res
        text_labels = self.token_array.text
        data = []
        for i, txt in enumerate(text_labels):
            data.append([reduced_embeddings[i][0], reduced_embeddings[i][1], reduced_embeddings[i][2], txt])
        return data

    def _create_scatter3d(self, width="100%", height="800%", theme=ThemeType.DARK, title="3D PCA Visualization of Embeddings"):
        """
        Create the 3D scatter plot.
        """
        scatter3d = (
            Scatter3D(init_opts=opts.InitOpts(theme=theme, width=width, height=height, page_title=title))
            .add(
                series_name="Embeddings",
                data=[[d[0], d[1], d[2]] for d in self.data],
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="3D PCA Visualization of Embeddings", pos_top='2%', pos_left='center'),
                visualmap_opts=opts.VisualMapOpts(max_=1, min_=-1),
                legend_opts=opts.LegendOpts(is_show=False),  # Hide the legend
            )
        )

        for d in self.data:
            scatter3d.add(
                series_name=d[3],
                data=[[d[0], d[1], d[2]]],
                label_opts=opts.LabelOpts(is_show=True, formatter=d[3], position="right"),
            )

        return scatter3d

    def show_web(self, output_file="scatter3d.html", port=8000):
        """
        Render the scatter plot to an HTML file, serve it via a web server, and open it in a web browser.
        """
        scatter3d = self._create_scatter3d()
        scatter3d.render(output_file)

        def cleanup():
            if os.path.exists(output_file):
                os.remove(output_file)
            print("Cleaned up the generated HTML file.")

        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", port), handler)
        
        def start_server():
            print(f"Serving at port {port}")
            httpd.serve_forever()

        # Open the web browser
        url = f"http://localhost:{port}/{output_file}"
        webbrowser.open(url)

        # Start the server in a separate thread
        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True
        server_thread.start()

        try:
            # Wait for the user to press Control-C
            while True:
                pass
        except KeyboardInterrupt:
            print("Shutting down the server...")
            httpd.shutdown()
            cleanup()

    def show_notebook(self):
        """
        Render the scatter plot directly in a Jupyter notebook.
        """
        scatter3d = self._create_scatter3d(width="900px", height="500px", theme=ThemeType.CHALK)
        return scatter3d.render_notebook()


class EmbeddingPawDatabase:
    _config = None
    
    @classmethod
    def set_config(cls, conf: EmbeddingPaw):
        cls._config = conf
    
    @staticmethod
    def __load_embedding_db(path: str) -> dict:
        """
        Load the embedding database from a file
        :param path: The path to the embedding database
        :return: The embedding database
        """
        path = os.path.expanduser(path)
        with open(path, 'rb') as f:
            embeddings_db = pickle.load(f)
        return embeddings_db
    
    def __init__(self):
        if EmbeddingPawDatabase._config is None:
            raise ValueError("Please set the config first.")
        self.filepath = EmbeddingPawDatabase._config.embedding_db_path
        self.data = self.__load_embedding_db(self.filepath)
        self.vocab_size = len(self.data)
        
    def __repr__(self):
        return f"<EmbeddingPawDatabase(file='{self.filepath}', vocab_size='{self.vocab_size}')>"
    
    def __getitem__(self, key):
        return self.data[key]
    
    def save_db_to_disk(self, path):
        """
        Save the database to a pickle file
        :param path: The file path to save the database
        :return: None
        """
        path = os.path.expanduser(path)
        if not path:
            path = self.filepath
        else:
            self.filepath = path
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
            
    def add_token(self, token: Token):
        """
        Add a token to the database
        :param token: The token to add
        :return: None
        """
        if token.text in self.data:
            print(f"Token '{token.text}' is already present.")
        else:
            self.data[token.text] = token.embedding
            self.vocab_size += 1
            print(f"Token '{token.text}' added successfully.")
    
    def load_token_from_txt(self, path: str):
        """
        Load tokens from a text file, where each line is a token, 
        we will calculate the embeddings for you.
        :param path: The path to the text file
        :return: None
        """
        path = os.path.expanduser(path)
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            elif line.strip() in self.data:
                print(f"Token '{line.strip()}' is already present.")
            else:
                token = Token(text=line.strip())
                self.add_token(token)
    
    def load_token_from_json(self, path: str):
        """
        Load tokens from a JSON file, The file should be in the format:
        {
            "token1": [0.1, 0.2, ... ,0.3],
            "token2": [0.4, 0.5, ... ,0.6],
            "token": embedding,
            ...
        }
        :param path: The path to the JSON file
        :return: None
        """
        path = os.path.expanduser(path)
        with open(path, 'r') as f:
            data = json.load(f)
        for token, embedding in data.items():
            if token in self.data:
                print(f"Token '{token}' is already present.")
            else:
                self.data[token] = embedding
                self.vocab_size += 1
    
    def load_token_from_excel(self, path: str):
        """
        Load tokens from an Excel file, where each row is a token, 
        If your Excel file has two columns, we will use the first column for token,
        the second column for embedding.
        If your Excel file has only one column, we will calculate the embeddings for you.
        :param path: The path to the Excel file
        :return: None
        """
        path = os.path.expanduser(path)
        df = pd.read_excel(path)
        if len(df.columns) == 1:
            for token in df.iloc[:, 0]: # used to access a group of rows and columns by label(s) or a boolean array.
                if token in self.data:
                    print(f"Token '{token}' is already present.")
                else:
                    token = Token(text=token)
                    self.add_token(token)
        elif len(df.columns) == 2:
            for token, embedding in df.itertuples(index=False):
                if token in self.data:
                    print(f"Token '{token}' is already present.")
                else:
                    self.data[token] = embedding
                    self.vocab_size += 1
        else:
            raise ValueError("Excel file must have one or two columns.")
    
    def delete_token(self, text: str):
        """
        Delete a token from the database
        :param text: The text of the token to delete
        :return: None
        """
        if text in self.data:
            del self.data[text]
            self.vocab_size -= 1
            print(f"Token '{text}' deleted successfully.")
        else:
            print(f"Token '{text}' not found.")
