import glob
from docmesh.tools.load_config import load_config
from docmesh.embedding.EmbeddingModelFactory import EmbeddingModelFactory
from docmesh.llm.LLMFactory import LLMFactory
from docmesh.evaluation.dataset import RAGTestsetGenerator, load_personas_from_yaml

from ragas.testset.transforms import HeadlineSplitter, HeadlinesExtractor
from ragas.testset.transforms.extractors import NERExtractor

from dotenv import load_dotenv

path_config = "../test_config.yaml"


def test_generate_eval_data_pipeline():
    load_dotenv()
    config = load_config(path_config)

    path_dir_test_data = "../test_data"
    list_path_html = glob.glob(path_dir_test_data + "/*.html")

    LLMFactory.create_llm(**config["llm"])
    generator_llm = LLMFactory.create_llm(**config["llm"])
    critic_llm = LLMFactory.create_llm(**config["llm"])
    embeddings = EmbeddingModelFactory.create_embedding_model(
        **config["embedding_model"]
    )

    list_persona = load_personas_from_yaml("../test_persona.yaml")

    generator = RAGTestsetGenerator(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        generator_embeddings=embeddings,
        list_persona=list_persona,
    )

    transforms = [HeadlinesExtractor(), HeadlineSplitter(), NERExtractor()]

    result = generator.create_testset(
        list_path_html, testset_size=len(list_path_html), transforms=transforms
    )
    result_pandas = result.to_pandas()
    assert len(result_pandas) == len(list_path_html)
