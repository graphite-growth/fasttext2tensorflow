# fasttext2tensorflow

This Python script takes a trained supervised FastText binary model and generates an equivalent serialized TensorFlow model which replicates the inference process of the FastText model. We developed this tool since we had trained FastText models that we needed to run in Google's BigQuery, which provides the option of uploading a serialized TensorFlow model to be run for prediction. Besides the **fasttext2tensorflow.py** script we share a Jupyter Notebook that describes the inference process of a FastText supervised model in plain Python, useful to understand the inner workings of FastText without having to go through the original C++ sources, the notebook is also available on [Google's Colaboratory](https://colab.research.google.com/drive/1fk1iOYwhuXLR2JyA2pL9jXTnnrl1qk4-?usp=sharing).

## Python Dependencies

- fasttext
- tensorflow

## How To Use

`python fasttext2tensorflow.py /path/to/fasttext/model.bin /path/to/save/tensorflow/model`

## How To Upload a TensorFlow model to BigQuery

[Google's Documentation](https://cloud.google.com/bigquery-ml/docs/making-predictions-with-imported-tensorflow-models)

Keep in mind that the TensorFlow model must be less than 250 MB in size to be allowed in BigQuery.

## Contact

Enrique Diaz De León: enrique@graphite.com
<br>
Alan Salinas: alan@graphite.com
