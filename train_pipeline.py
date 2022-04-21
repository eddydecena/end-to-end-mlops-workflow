import os
from typing import NamedTuple

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp import compiler
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Model
)

@component(packages_to_install=['pandas==1.1.4'])
def data_extraction(arxiv_data_filtered_output: Output[Dataset]):
    import pandas as pd
    arxiv_data = pd.read_csv(
        "https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv"
    )
    arxiv_data = arxiv_data[:10000]
    arxiv_data.to_csv(arxiv_data_filtered_output.path, index=False, header=True)

@component(packages_to_install=['pandas==1.1.4'])
def data_analysis(dataset: Input[Dataset]):
    import pandas as pd
    
    arxiv_data = pd.read_csv(dataset.path, )
    
    arxiv_data.head()
    arxiv_data[:]
    print(f"There are {len(arxiv_data)} rows in the dataset.")

    total_duplicate_titles = sum(arxiv_data["titles"].duplicated())
    print(f"There are {total_duplicate_titles} duplicate titles.")

    arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]
    print(f"There are {len(arxiv_data)} rows in the deduplicated dataset.")
    
    print(sum(arxiv_data["terms"].value_counts() == 1))

    print(arxiv_data["terms"].nunique())

@component(packages_to_install=['pandas==1.1.4', 'scikit-learn==1.0.2', 'tensorflow-cpu==2.8.0'])
def data_transformation(
        dataset: Input[Dataset], 
        train_dataset_output: Output[Dataset], 
        validation_dataset_output: Output[Dataset], 
        test_dataset_output: Output[Dataset]
    ) -> NamedTuple(
            'Datasets',
            [('train_dataset_path', str),
            ('validation_dataset_path', str), 
            ('test_dataset_path', str)]):
    
    from ast import literal_eval
    from collections import namedtuple
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    
    arxiv_data = pd.read_csv(dataset.path)
    
    arxiv_data_filtered = arxiv_data.groupby("terms").filter(lambda x: len(x) > 1)
    print(arxiv_data_filtered.shape)
    
    arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(
        lambda x: literal_eval(x)
    )
    arxiv_data_filtered["terms"].values[:5]
    
    test_split = 0.1

    # Initial train and test split.
    train_df, test_df = train_test_split(
        arxiv_data_filtered,
        test_size=test_split,
        stratify=arxiv_data_filtered["terms"].values,
    )

    # Splitting the test set further into validation
    # and new test sets.
    val_df = test_df.sample(frac=0.5)
    test_df.drop(val_df.index, inplace=True)

    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in validation set: {len(val_df)}")
    print(f"Number of rows in test set: {len(test_df)}")
    
    terms = tf.ragged.constant(train_df["terms"].values)
    lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
    lookup.adapt(terms)
    vocab = lookup.get_vocabulary()
    print('Lookup.get_vocabulary return type: ', type(vocab))


    def invert_multi_hot(encoded_labels):
        """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
        hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
        return np.take(vocab, hot_indices)

    print("Vocabulary:\n")
    print(vocab)
    
    sample_label = train_df["terms"].iloc[0]
    print(f"Original label: {sample_label}")

    label_binarized = lookup([sample_label])
    print(f"Label-binarized representation: {label_binarized}")
    
    train_df["summaries"].apply(lambda x: len(x.split(" "))).describe()
    
    max_seqlen = 150
    batch_size = 128
    padding_token = "<pad>"
    auto = tf.data.AUTOTUNE

    def make_dataset(dataframe, is_train=True):
        labels = tf.ragged.constant(dataframe["terms"].values)
        label_binarized = lookup(labels).numpy()
        dataset = tf.data.Dataset.from_tensor_slices(
            (dataframe["summaries"].values, label_binarized)
        )
        dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
        return dataset.batch(batch_size)
    
    train_dataset = make_dataset(train_df, is_train=True)
    validation_dataset = make_dataset(val_df, is_train=False)
    test_dataset = make_dataset(test_df, is_train=False)
    
    text_batch, label_batch = next(iter(train_dataset))

    for i, text in enumerate(text_batch[:5]):
        label = label_batch[i].numpy()[None, ...]
        print(f"Abstract: {text}")
        print(f"Label(s): {invert_multi_hot(label[0])}")
        print(" ")
    
    vocabulary = set()
    train_df["summaries"].str.lower().str.split().apply(vocabulary.update)
    vocabulary_size = len(vocabulary)
    print(vocabulary_size)
    
    text_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
    )

    # `TextVectorization` layer needs to be adapted as per the vocabulary from our
    # training set.
    with tf.device("/CPU:0"):
        text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

    train_dataset = train_dataset.map(
        lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)
    validation_dataset = validation_dataset.map(
        lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)
    test_dataset = test_dataset.map(
        lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)
    
    # save dataset
    tf.data.experimental.save(train_dataset, train_dataset_output.path)
    tf.data.experimental.save(validation_dataset, validation_dataset_output.path)
    tf.data.experimental.save(test_dataset, test_dataset_output.path)
    
    datasets = namedtuple('Datasets', ['train_dataset_path', 'validation_dataset_path', 'test_dataset_path'])
    return datasets(
        train_dataset_path=train_dataset_output.path,
        validation_dataset_path=validation_dataset_output.path,
        test_dataset_path=test_dataset_output.path
    )

@component(packages_to_install=['pandas==1.1.4', 'scikit-learn==1.0.2', 'tensorflow-cpu==2.8.0'])
def train(raw_dataset: Input[Dataset], 
    train_dataset_output: Input[Artifact], 
    validation_dataset_output: Input[Artifact], 
    test_dataset_output: Input[Artifact], 
    model: Output[Model]):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow import keras
    import tensorflow as tf
    from ast import literal_eval
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    arxiv_data = pd.read_csv(raw_dataset.path)
    
    arxiv_data_filtered = arxiv_data.groupby("terms").filter(lambda x: len(x) > 1)
    print(arxiv_data_filtered.shape)
    
    arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(
        lambda x: literal_eval(x)
    )
    arxiv_data_filtered["terms"].values[:5]
    
    test_split = 0.1

    # Initial train and test split.
    train_df, test_df = train_test_split(
        arxiv_data_filtered,
        test_size=test_split,
        stratify=arxiv_data_filtered["terms"].values,
    )

    # Splitting the test set further into validation
    # and new test sets.
    val_df = test_df.sample(frac=0.5)
    test_df.drop(val_df.index, inplace=True)

    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in validation set: {len(val_df)}")
    print(f"Number of rows in test set: {len(test_df)}")
    
    # optimize lookup
    terms = tf.ragged.constant(train_df["terms"].values)
    lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
    lookup.adapt(terms)
    vocab = lookup.get_vocabulary()
    print('Lookup.get_vocabulary return type: ', type(vocab))
    
    # ready dataset
    train_dataset = tf.data.experimental.load(train_dataset_output.path)
    validation_dataset = tf.data.experimental.load(validation_dataset_output.path)
    test_dataset = tf.data.experimental.load(test_dataset_output.path)
    
    def make_model():
        shallow_mlp_model = keras.Sequential(
            [
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
            ]
        )
        return shallow_mlp_model
    
    epochs = 20

    shallow_mlp_model = make_model()
    shallow_mlp_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
    )

    history = shallow_mlp_model.fit(
        train_dataset, validation_data=validation_dataset, epochs=epochs
    )
    
    shallow_mlp_model.save(model.path)
    
    print(history)

@component(packages_to_install=['tensorflow-cpu==2.8.0'])
def validation(test_dataset_input: Input[Dataset], model_input: Input[Model]) -> bool:
    import tensorflow as tf
    
    test_dataset = tf.data.experimental.load(test_dataset_input.path)
    model = tf.keras.models.load_model(model_input.path)
    _, categorical_acc = model.evaluate(test_dataset)
    print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%.")
    if round(categorical_acc * 100, 2) > 70:
        return True
    
    return False

@component
def deploy(model: Input[Model]) -> None:
    print('deploying models...')

@dsl.pipeline(
    name=os.environ.get('KFP_NAME', 'Multi-label classification for arxiv paper abstract'),
    # pipeline_root='gs://artifacts-4029582/kubeflow-templates'
)
def arxiv_pipeline():
    data_extraction_op = data_extraction()
    data_analysis(data_extraction_op.output).after(data_extraction_op)
    data_transformation_op = data_transformation(data_extraction_op.output).after(data_extraction_op)
    train_op = train(
        data_extraction_op.output,
        data_transformation_op.outputs['train_dataset_output'], 
        data_transformation_op.outputs['validation_dataset_output'],
        data_transformation_op.outputs['test_dataset_output']
    ).after(data_transformation_op)
    validation_op = validation(data_transformation_op.outputs['test_dataset_output'], train_op.output)
    
    with dsl.Condition(validation_op.output=="true", 'validation_threshold'):
        deploy(train_op.output)

compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(pipeline_func=arxiv_pipeline, package_path=os.environ.get('KFP_ARTIFACT', 'pipeline.yaml'))

client = kfp.Client(host=os.environ.get('KFP_HOST', 'http://localhost:8080'))
# run the pipeline in v2 compatibility mode
client.create_run_from_pipeline_func(
    arxiv_pipeline,
    arguments={},
    mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
    enable_caching=False
)