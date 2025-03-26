import glob
from docmesh.tools.load_config import load_config
from docmesh.embedding.EmbeddingModelFactory import EmbeddingModelFactory
from docmesh.llm.LLMFactory import LLMFactory
from docmesh.evaluation import RAGTestsetGenerator, load_personas_from_yaml

from ragas.testset.transforms import HeadlineSplitter, HeadlinesExtractor
from ragas.testset.transforms.extractors import NERExtractor

import pandas as pd
from datasets import Dataset

from dotenv import load_dotenv


def generate_dataset(
    path_config: str, path_dir_test_data: str, testset_size: int, path_save: str
):
    load_dotenv()
    config = load_config(path_config)

    list_path_html = glob.glob(path_dir_test_data + "/*.html")
    assert len(list_path_html) > 0

    LLMFactory.create_llm(**config["llm"])
    generator_llm = LLMFactory.create_llm(**config["llm"])
    critic_llm = LLMFactory.create_llm(**config["llm"])
    embeddings = EmbeddingModelFactory.create_embedding_model(
        **config["embedding_model"]
    )

    list_persona = load_personas_from_yaml(path_config)

    generator = RAGTestsetGenerator(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        generator_embeddings=embeddings,
        list_persona=list_persona,
    )

    transforms = [HeadlinesExtractor(), HeadlineSplitter(), NERExtractor()]

    result = generator.create_testset(
        list_path_html, testset_size=testset_size, transforms=transforms
    )
    result_pandas = result.to_pandas()
    result_pandas.to_csv(path_save, index=False)
    return result_pandas


def load_dataset(path_csv: str):
    df = pd.read_csv(path_csv)
    df["reference_contexts"] = df["reference_contexts"].map(eval)
    test_dataset = Dataset.from_pandas(df)
    test_dataset = test_dataset.add_column(
        "retrieved_contexts", list(test_dataset["reference_contexts"])
    )
    return test_dataset
