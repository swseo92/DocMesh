import pytest
from docmesh.llm.BaseLLM import BaseLLM


def test_base_llm_is_abstract():
    with pytest.raises(TypeError):
        _ = BaseLLM()
