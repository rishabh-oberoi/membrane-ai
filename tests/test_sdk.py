import os
import pytest
from membrane import TrustLayer
from membrane.config import reset_config, get_config

class TestSDKInterface:
    def setup_method(self):
        reset_config()
        # Clear relevant env vars
        for key in ["TRUST_LLM_PROVIDER", "TRUST_LLM_MODEL", "TRUST_LLM_API_BASE"]:
            if key in os.environ:
                del os.environ[key]

    def test_init_defaults(self):
        layer = TrustLayer()
        assert layer.provider == "mock"
        assert layer.model is None

    def test_init_overrides(self):
        layer = TrustLayer(provider="openai", model="gpt-4", api_key="sk-test")
        assert layer.provider == "openai"
        assert layer.model == "gpt-4"
        assert layer.api_key == "sk-test"

    def test_env_var_sync(self):
        """TrustLayer should sync its settings to env vars for the underlying config singleton."""
        TrustLayer(provider="gemini", model="gemini-pro")
        config = get_config()
        assert config.llm_provider == "gemini"
        assert config.llm_model == "gemini-pro"

    def test_mock_call(self):
        layer = TrustLayer(provider="mock")
        result = layer.call("My name is John.")
        assert "original_prompt" in result
        assert "anonymized_prompt" in result
        assert "final_response" in result
        assert result["metrics"]["status"] == "ok"

    def test_model_parameter_passing(self):
        """Verify that the model string is preserved in the layer object."""
        model_name = "custom-model-v7"
        layer = TrustLayer(provider="openai", model=model_name)
        assert layer.model == model_name
        
        # Check that it propagated to the environment
        assert os.environ.get("TRUST_LLM_MODEL") == model_name
