import json
from typing import Any

import pytest
import responses

from tfbpapi.AbstractAPI import AbstractAPI
from tfbpapi.ParamsDict import ParamsDict


class ConcreteAPI(AbstractAPI):
    """Concrete implementation of AbstractAPI for testing purposes."""

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        pass  # Implement for testing if necessary

    def read(self, **kwargs) -> dict[str, Any]:
        return {"id": id}  # Mock implementation for testing

    def update(self, df: Any, **kwargs) -> Any:
        pass  # Implement for testing if necessary

    def delete(self, id: str, **kwargs) -> Any:
        pass  # Implement for testing if necessary

    def submit(self, post_dict: dict, **kwargs) -> Any:
        pass  # Implement for testing if necessary

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        pass  # Implement for testing if necessary


@pytest.fixture
@responses.activate
def api_client():
    valid_url = "https://valid.url"
    responses.add(responses.HEAD, valid_url, status=200)
    return ConcreteAPI(url=valid_url, token="token")


def test_initialize(snapshot, api_client):
    assert api_client.url == "https://valid.url"
    assert api_client.token == "token"
    assert isinstance(api_client.params, ParamsDict)


def test_push_params(snapshot, api_client):
    params = {"param1": "value1", "param2": "value2"}
    api_client.push_params(params)
    # Serialize the dictionary to a JSON string for comparison
    params_as_json = json.dumps(api_client.params.as_dict(), sort_keys=True)
    snapshot.assert_match(params_as_json, "push_params")


def test_pop_params(snapshot, api_client):
    params = {"param1": "value1", "param2": "value2"}
    api_client.push_params(params)
    api_client.pop_params(["param1"])
    params_as_json1 = json.dumps(api_client.params.as_dict(), sort_keys=True)
    snapshot.assert_match(params_as_json1, "pop_params_after_one_removed")
    api_client.pop_params()
    params_as_json2 = json.dumps(api_client.params.as_dict(), sort_keys=True)
    snapshot.assert_match(params_as_json2, "pop_params_after_all_removed")


@responses.activate
def test_is_valid_url(api_client):
    invalid_url = "https://invalid.url"

    responses.add(responses.HEAD, invalid_url, status=404)

    with pytest.raises(ValueError):
        api_client.url = invalid_url


def test_cache_operations(snapshot, api_client):
    key = "test_key"
    value = "test_value"

    api_client._cache_set(key, value)
    snapshot.assert_match(str(api_client._cache_get(key)), "cache_get_after_set")

    keys = api_client._cache_list()
    snapshot.assert_match(str(keys), "cache_list")

    api_client._cache_delete(key)
    snapshot.assert_match(str(api_client._cache_get(key)), "cache_get_after_delete")
    snapshot.assert_match(str(api_client._cache_get(key)), "cache_get_after_delete")


if __name__ == "__main__":
    pytest.main()
