import pandas as pd
import pytest

pytest.importorskip("simpy")
pytest.importorskip("mlflow")
torch = pytest.importorskip("torch")

from ml4cps.debta.base import WindowedSequenceDataset


def test_windowed_sequence_dataset_stores_dataframe_time_indices():
    sequence = pd.DataFrame(
        {"x": [10.0, 20.0, 30.0, 40.0]},
        index=[0.0, 0.5, 1.0, 1.5],
    )
    dataset = WindowedSequenceDataset([sequence], window_size=2, window_stride=1)

    assert dataset.time_index_windows.tolist() == [
        [0.0, 0.5],
        [0.5, 1.0],
        [1.0, 1.5],
    ]

    values, time_index = dataset.get_with_time(1)
    assert values.tolist() == [[20.0], [30.0]]
    assert time_index.tolist() == [0.5, 1.0]


def test_windowed_sequence_dataset_uses_range_index_for_tensor_input():
    sequence = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    dataset = WindowedSequenceDataset([sequence], window_size=2, window_stride=2)

    assert dataset.time_index_windows.tolist() == [
        [0, 1],
        [2, 3],
    ]

    values, time_index = dataset.get_with_time(0)
    assert values.tolist() == [[1.0], [2.0]]
    assert time_index.tolist() == [0, 1]
