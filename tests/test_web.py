"""Tests for the Flask front end: host checks and per-request isolation."""

import pytest

pytest.importorskip("flask")

from Web_based_interface import create_app  # noqa: E402
from test_nlp_engine import engine_with  # noqa: E402


def client_for(factory):
    return create_app(engine_factory=factory).test_client()


def post(client, host="localhost:5000", **body):
    return client.post("/generate_code", json=body, headers={"Host": host})


def test_translates_a_request():
    client = client_for(lambda: engine_with("```python\nprint(1)\n```"))
    r = post(client, message="print one")
    assert r.status_code == 200 and r.get_json()["code"] == "print(1)"


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1:5000", "[::1]:5000", "[::1]"])
def test_local_hosts_are_allowed(host):
    client = client_for(lambda: engine_with("x = 1"))
    assert post(client, host=host, message="x").status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:5000", "localhost.evil.example"])
def test_foreign_host_is_rejected(host):
    """DNS rebinding: a hostile domain resolved to 127.0.0.1 keeps its own Host."""
    client = client_for(lambda: engine_with("x = 1"))
    assert post(client, host=host, message="x").status_code == 403


def test_each_request_gets_its_own_conversation():
    """One client's input must never reach another client's context."""
    engines = []

    def factory():
        engines.append(engine_with("x = 1"))
        return engines[-1]

    client = client_for(factory)
    post(client, message="my secret request")
    post(client, message="another user")

    second_call = engines[1].brion.provider.calls[0]["messages"]
    assert [m["content"] for m in second_call] == ["another user"]


def test_empty_message_is_rejected():
    assert post(client_for(lambda: engine_with()), message=" ").status_code == 400


def test_unknown_language_is_a_client_error():
    r = post(client_for(lambda: engine_with("x")), message="x", language="klingon")
    assert r.status_code == 400 and "Unknown language" in r.get_json()["error"]


def test_index_serves_the_front_end():
    r = client_for(lambda: engine_with()).get("/", headers={"Host": "localhost"})
    assert r.status_code == 200 and b"message-input" in r.data
