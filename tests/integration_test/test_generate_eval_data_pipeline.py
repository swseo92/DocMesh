from docmesh.evaluation.basic_generate_pipeline import generate_dataset
import os


from dotenv import load_dotenv

path_config = "../test_config.yaml"


def test_generate_eval_data_pipeline():
    load_dotenv()
    path_dir_test_data = "../test_data"
    testset_size = 10
    path_save = "test_data.csv"

    if os.path.exists(path_save):
        os.remove(path_save)

    assert os.path.exists(path_dir_test_data)
    assert not os.path.exists(path_save)

    result = generate_dataset(path_config, path_dir_test_data, testset_size, path_save)
    assert len(result) == testset_size
    assert os.path.exists(path_save)
