import os

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
def data_preprocessing(arxiv_data_filtered_output: Output[Dataset]):
    from ast import literal_eval
    import pandas as pd
    
    arxiv_data = pd.read_csv(
        "https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv"
    )
    arxiv_data.head()

    print(f"There are {len(arxiv_data)} rows in the dataset.")

    total_duplicate_titles = sum(arxiv_data["titles"].duplicated())
    print(f"There are {total_duplicate_titles} duplicate titles.")

    arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]
    print(f"There are {len(arxiv_data)} rows in the deduplicated dataset.")

    # There are some terms with occurrence as low as 1.
    print('Number of terms with 1 occurence: ', sum(arxiv_data["terms"].value_counts() == 1))

    # How many unique terms?
    print('Unique terms: ', arxiv_data["terms"].nunique())

    # Filtering the rare terms.
    arxiv_data_filtered = arxiv_data.groupby("terms").filter(lambda x: len(x) > 1)
    print('filtered: ', arxiv_data_filtered.shape)

    # Convert the string labels to lists of strings
    arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(
        lambda x: literal_eval(x)
    )
    print('Convert label to list of string: ', arxiv_data_filtered["terms"].values[:5])
    
    arxiv_data_filtered.to_csv(arxiv_data_filtered_output.path, index=False, header=False)

@component(packages_to_install=['pandas==1.1.4'])
def test(dataset: Input[Dataset]):
    import pandas as pd
    df = pd.read_csv(dataset.path)
    print(df.head())

@dsl.pipeline(
    name=os.environ.get('KFP_NAME', 'Multi-label classification for arxiv paper abstract'),
#   pipeline_root='gs://my-pipeline-root/example-pipeline'
)
def arxiv_pipeline():
    dataset = data_preprocessing()
    test(dataset.output)

compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(pipeline_func=arxiv_pipeline, package_path=os.environ.get('KFP_ARTIFACT', 'pipeline.yaml'))

client = kfp.Client(host=os.environ.get('KFP_SERVER', 'http://localhost:8080'))
# run the pipeline in v2 compatibility mode
client.create_run_from_pipeline_func(
    arxiv_pipeline,
    arguments={},
    mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
)