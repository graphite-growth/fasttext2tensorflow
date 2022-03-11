import sys

import fasttext
import tensorflow as tf


class HashWord(tf.Module):  # type: ignore
    """
    Computes a hash value for the given string
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    @tf.function()
    def __call__(self, inputs):
        # Split string into unicode characters
        characters = tf.cast(tf.io.decode_raw(inputs, out_type=tf.uint8), tf.uint32)
        # Apply hashing function recursively to each character of string
        h = tf.foldl(
            lambda a, x: tf.bitwise.bitwise_xor(a, x)
            * tf.constant(16777619, dtype="uint32"),
            characters,
            initializer=tf.constant(2166136261, dtype="uint32"),
        )
        return h


class GetSubwordIndex(tf.Module):  # type: ignore
    """
    Returns the input_matrix index for the given subword string
    """

    def __init__(self, vocab_size, bucket_size, name=None):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.bucket_size = bucket_size

    @tf.function()
    def __call__(self, inputs):
        # Get hashes ffor given subword
        hash = HashWord()(inputs)
        # Modulo % the hash by the bucket size
        modulo_buckets = tf.math.floormod(
            hash, tf.constant(self.bucket_size, dtype="uint32")
        )
        # Add vocabulary length to find subword index
        return modulo_buckets + self.vocab_size
            
        
class GetWordNgramIndex(tf.Module):  # type: ignore
    """
    Given a wordngram in string format, return the index of input_matrix for
    the wordngram
    """

    def __init__(self, vocab_size, bucket_size, name=None):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.bucket_size = bucket_size

    @tf.function()
    def __call__(self, inputs):
        words = tf.strings.split(inputs)
        # Map every wordngram tensor to a hashes tensor
        hashes = tf.map_fn(
            HashWord(),
            words,
            fn_output_signature=tf.uint32,
        )
        hashes = tf.cast(tf.cast(hashes, "int32"), "uint64")
        h = tf.foldl(
            lambda a, x: tf.cast(tf.constant(116049371, dtype="uint64")
            * a + tf.cast(tf.cast(x, "int32"), "uint64"), "uint64"),
            hashes[1:],
            initializer=tf.gather(hashes, 0),
        )
        wordngram_index = tf.math.floormod(h, tf.constant(self.bucket_size, dtype="uint64"))
        return wordngram_index + self.vocab_size


class FilterInvalidWords(tf.Module):  # type: ignore
    """
    Returns the valid subset of token indexes for input_matrix (token >= 0)
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    @tf.function()
    def __call__(self, inputs):
        valid_positions = tf.reshape(tf.where(tf.math.greater_equal(inputs, 0)), [-1])
        return tf.gather(inputs, valid_positions)


class GetSubwordsIndexes(tf.Module):  # type: ignore
    """
    Return the subword indexes in input_matrix for each of the input subwords
    """

    def __init__(self, vocab_size, bucket_size, name=None):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.bucket_size = bucket_size

    @tf.function()
    def __call__(self, inputs):
        # Map every subword to an input_matrix index
        subwords_indexes = tf.map_fn(
            GetSubwordIndex(self.vocab_size, self.bucket_size),
            inputs,
            fn_output_signature=tf.uint32,
        )
        return subwords_indexes
        
        
class GetWordNgramIndexes(tf.Module):  # type: ignore
    """
    Return the subword indexes in input_matrix for each of the input subwords
    """

    def __init__(self, vocab_size, bucket_size, name=None):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.bucket_size = bucket_size

    @tf.function()
    def __call__(self, inputs):
        indexes = tf.map_fn(
            GetWordNgramIndex(self.vocab_size, self.bucket_size),
            inputs,
            fn_output_signature=tf.uint64,
        )
        return indexes


class WordTokenizerLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Given a tensor of input sentences this layer returns a ragged tensor with
    the list of valid word tokens as indexes of input_matrix, for every sentence
    """

    def __init__(self, vocabulary):
        super(WordTokenizerLayer, self).__init__()
        # Map word strings to word indexes in vocabulary
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(vocabulary, range(len(vocabulary))),
            default_value=-1,
        )

    def call(self, inputs):
        # Split sentences into words
        words = tf.strings.split(inputs)
        # Find word indexes in vocabulary
        indexes = self.table.lookup(words)
        # Filter OOV word indexes = -1 (not found)
        valid_indexes = tf.map_fn(
            FilterInvalidWords(),
            indexes,
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[None], dtype=tf.int32, ragged_rank=0
            ),
        )
        return valid_indexes


class AddEOLTokenLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Add a FastText EOL token to each sentence in the tensor
    """

    def __init__(self):
        super(AddEOLTokenLayer, self).__init__()

    def call(self, inputs):
        sentences = tf.strings.reduce_join(
            [inputs, tf.broadcast_to(tf.constant("</s>"), shape=tf.shape(inputs))],
            separator=" ",
            axis=0,
        )
        return sentences


class SubwordTokenizerLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Given a tensor of sentences this layer will return a ragged tensor with
    all the tokens per sentence as indexes of input_matrix
    """

    def __init__(self, minn, maxn, vocab_size, bucket_size):
        super(SubwordTokenizerLayer, self).__init__()
        self.bow = tf.constant("<")
        self.eow = tf.constant(">")
        self.minn = minn
        self.maxn = maxn
        self.vocab_size = vocab_size
        self.bucket_size = bucket_size

    def call(self, inputs):
        # Split sentences into a ragged tensor with words per sentence
        words = tf.strings.split(inputs)
        # Add BOW and EOW chars to all words
        words_ = tf.ragged.map_flat_values(
            tf.strings.reduce_join,
            [
                tf.broadcast_to(self.bow, shape=(tf.size(words),)),
                words,
                tf.broadcast_to(self.eow, shape=(tf.size(words),)),
            ],
            separator="",
            axis=0,
        )
        # Make ngrams of characters
        subwords = tf.strings.ngrams(
            tf.strings.unicode_split(words_, "UTF-8"),  # Split into characters
            [i for i in range(self.minn, self.maxn + 1)],  # For all sizes
            separator="",
        )
        # Map each ngram to an index of input_matrix
        subwords_indexes = tf.ragged.map_flat_values(
            GetSubwordsIndexes(self.vocab_size, self.bucket_size), subwords
        )
        # Cast needed for embedding layer
        return tf.cast(subwords_indexes, dtype="int64")


class WordNgramsTokenizerLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Returns a ragged tensor with the tokens as indexes of input_matrix, for
    every sentence in the input tensor
    """
    def __init__(self, max_ngram_length, vocab_size, bucket_size):
        super(WordNgramsTokenizerLayer, self).__init__()
        self.max_ngram_length = max_ngram_length
        self.vocab_size = vocab_size
        self.bucket_size = bucket_size
        
    def call(self, inputs):
        # Create ragged tensor with words per sentence
        sentence_tokens = tf.strings.split(inputs)
        # Make ragged tensor of word ngrams per sentence
        sentence_word_ngrams = tf.strings.ngrams(
            sentence_tokens,
            [i for i in range(2, self.max_ngram_length+1)]
        )
        # Get the indexes for each of the ngrams
        ngram_indexes = tf.ragged.map_flat_values(
            GetWordNgramIndexes(self.vocab_size, self.bucket_size),
            sentence_word_ngrams,
        )
        return tf.cast(ngram_indexes, "int64")


class EmbeddingLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Return embeddings for the given indexes of input_matrix
    """

    def __init__(self, embedding_matrix):
        super(EmbeddingLayer, self).__init__()
        self.embedding_matrix = tf.constant(embedding_matrix)

    def call(self, inputs):
        return tf.ragged.map_flat_values(tf.gather, self.embedding_matrix, inputs)


class CombineFeaturesLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Stack ragged tensors of features into a single feature dimension
    """

    def __init__(self):
        super(CombineFeaturesLayer, self).__init__()

    def call(self, inputs):
        results = tf.ragged.stack(inputs, axis=1).merge_dims(1, 2)
        return results


class FlattenLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Reduce redundant dimensions (1, 2)
    """

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def call(self, inputs):
        return inputs.merge_dims(1, 2)


class SentenceVectorLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Returns a sentence vector by mean reduce of all the feature vectors of
    each sentence in the tensor
    """

    def __init__(self):
        super(SentenceVectorLayer, self).__init__()

    def call(self, inputs):
        return tf.math.reduce_mean(inputs, axis=1)


class LinearLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Dense layer implementantion, returns the output of the single layer
    neural network with trained FastText weights
    """

    def __init__(self, output_weights):
        super(LinearLayer, self).__init__()
        self.output_weights = tf.constant(output_weights)

    def call(self, inputs):
        # Multiply class weights by sentence vector
        return tf.linalg.matmul(
            self.output_weights,
            tf.reshape(inputs, (tf.shape(inputs)[0], tf.shape(inputs)[1], 1)),
        )


class PredictionLabelLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Given a tensor of softmaxed output class vectors, return for each the
    corresponding label for the highest probability class
    """

    def __init__(self, indexes, labels):
        super(PredictionLabelLayer, self).__init__()
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(indexes, labels), default_value="NA"
        )

    def call(self, inputs):
        # Get the position in the vector for largest probability
        pred = tf.cast(tf.math.argmax(inputs, axis=-1), tf.int32)
        # Lookup the vector position corresponding label
        return self.table.lookup(pred)
        

class PredictionProbaLayer(tf.keras.layers.Layer):  # type: ignore
    """
    Given a tensor of softmaxed output class vectors, return for each the
    corresponding label for the highest probability class
    """

    def __init__(self):
        super(PredictionProbaLayer, self).__init__()

    def call(self, inputs):
        # Get the largest probability per sentence
        pred = tf.math.reduce_max(inputs, axis=-1)
        return pred


def get_fasttext_parameters(fasttext_model_path):
    """
    Return a dictionary containing the trained parameters and the hyperparameters
    of a binary FastText model
    """
    model = fasttext.load_model(fasttext_model_path)

    loss = model.f.getArgs().loss.name
    if loss != "softmax":
        raise Exception(f"Model has {loss} loss, only softmax is supported")

    return {
        "vocabulary": model.get_words(),
        "embedding_dim": model.get_dimension(),
        "minn": model.f.getArgs().minn,
        "maxn": model.f.getArgs().maxn,
        "wordNgrams": model.f.getArgs().wordNgrams,
        "bucket_size": model.f.getArgs().bucket,
        "input_matrix": model.get_input_matrix(),
        "output_matrix": model.get_output_matrix(),
        "labels": [label.replace("__label__", "") for label in model.labels],
    }


def build_tensorflow_model(fasttext_parameters):
    """
    Build a Keras model that replicate the inference process of a FastText model
    defined by the given parameters and hyperparameters
    """
    inputs = tf.keras.Input((), dtype=tf.string, name="input")
    # Define the embedding layer for all features
    embeddings_layer = EmbeddingLayer(embedding_matrix=fasttext_parameters["input_matrix"])
    # Add the </s> EOL token to the sentence
    words_layer = AddEOLTokenLayer()(inputs)
    # Get the word tokens as indexes of input_matrix
    word_index_layer = WordTokenizerLayer(vocabulary=fasttext_parameters["vocabulary"])(
        words_layer
    )
    # Get embeddings for word tokens
    word_embedding_layer = embeddings_layer(word_index_layer)
    # Append to the sentence feature accumulator
    sentence_features = [word_embedding_layer]

    # Add character Ngram features if available
    if fasttext_parameters["minn"] != 0 and fasttext_parameters["maxn"] != 0:
        # Get all subword tokens
        subword_tokens_layer = SubwordTokenizerLayer(
            fasttext_parameters["minn"],
            fasttext_parameters["maxn"],
            len(fasttext_parameters["vocabulary"]),
            fasttext_parameters["bucket_size"],
        )(inputs)
        # Get embedding vectors for subword features
        subword_embedding_layer = FlattenLayer()(
            embeddings_layer(subword_tokens_layer)
        )
        # Append to the sentence feature accumulator
        sentence_features.append(subword_embedding_layer)

    # Add wordNgram features if available
    if fasttext_parameters["wordNgrams"] > 1:
        # Get the tokens as indexes of input_matrix for every wordNgram
        wordngram_tokens_layer = WordNgramsTokenizerLayer(
            fasttext_parameters["wordNgrams"],
            len(fasttext_parameters["vocabulary"]),
            fasttext_parameters["bucket_size"]
        )(words_layer)
        # Get embeddings for all word ngrams
        wordngram_embedding_layer = embeddings_layer(wordngram_tokens_layer)
        # Append to the sentence feature accumulator
        sentence_features.append(wordngram_embedding_layer)

    # Get all sentence features (word + subword + wordNgram embeddings)
    sentence_features_layer = CombineFeaturesLayer()(sentence_features)
    # Get single sentence vector by aggregating features
    sentence_vector_layer = SentenceVectorLayer()(sentence_features_layer)

    # Multiply sentence vector by output weights
    matmul_layer = LinearLayer(output_weights=fasttext_parameters["output_matrix"])(
        sentence_vector_layer
    )
    # Flatten output vector
    flat_matmul_layer = tf.keras.layers.Reshape((-1,))(matmul_layer)
    # Apply softmax to output vector
    softmax_layer = tf.keras.layers.Softmax()(flat_matmul_layer)
    # Get prediction label
    labels = PredictionLabelLayer(
        indexes=range(len(fasttext_parameters["labels"])),
        labels=fasttext_parameters["labels"],
    )(softmax_layer)
    
    proba = PredictionProbaLayer()(softmax_layer)

    return tf.keras.Model(inputs=inputs, outputs=[labels, proba])


def main(fasttext_model_path, tensorflow_model_path):
    fasttext_parameters = get_fasttext_parameters(fasttext_model_path)
    tensorflow_model = build_tensorflow_model(fasttext_parameters)
    tensorflow_model.save(tensorflow_model_path)


if __name__ == "__main__":
    fasttext_model_path = sys.argv[1]
    tensorflow_model_path = sys.argv[2]
    main(fasttext_model_path, tensorflow_model_path)
