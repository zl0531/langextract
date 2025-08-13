# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for provider schema discovery and implementations."""

from unittest import mock

from absl.testing import absltest

from langextract import data
from langextract import exceptions
from langextract import factory
from langextract import schema
from langextract.providers import gemini as gemini_provider
from langextract.providers import ollama
from langextract.providers import openai


class ProviderSchemaDiscoveryTest(absltest.TestCase):
  """Tests for provider schema discovery via get_schema_class()."""

  def test_gemini_returns_gemini_schema(self):
    """Test that GeminiLanguageModel returns GeminiSchema."""
    schema_class = gemini_provider.GeminiLanguageModel.get_schema_class()
    self.assertEqual(
        schema_class,
        schema.GeminiSchema,
        msg="GeminiLanguageModel should return GeminiSchema class",
    )

  def test_ollama_returns_format_mode_schema(self):
    """Test that OllamaLanguageModel returns FormatModeSchema."""
    schema_class = ollama.OllamaLanguageModel.get_schema_class()
    self.assertEqual(
        schema_class,
        schema.FormatModeSchema,
        msg="OllamaLanguageModel should return FormatModeSchema class",
    )

  def test_openai_returns_none(self):
    """Test that OpenAILanguageModel returns None (no schema support yet)."""
    # OpenAI imports dependencies in __init__, not at module level
    schema_class = openai.OpenAILanguageModel.get_schema_class()
    self.assertIsNone(
        schema_class,
        msg="OpenAILanguageModel should return None (no schema support)",
    )


class FormatModeSchemaTest(absltest.TestCase):
  """Tests for FormatModeSchema implementation."""

  def test_from_examples_ignores_examples(self):
    """Test that FormatModeSchema ignores examples and returns JSON mode."""
    examples_data = [
        data.ExampleData(
            text="Test text",
            extractions=[
                data.Extraction(
                    extraction_class="test_class",
                    extraction_text="test extraction",
                    attributes={"key": "value"},
                )
            ],
        )
    ]

    test_schema = schema.FormatModeSchema.from_examples(examples_data)
    self.assertEqual(
        test_schema._format,
        "json",
        msg="FormatModeSchema should default to JSON format",
    )

  def test_to_provider_config_returns_format(self):
    """Test that to_provider_config returns format parameter."""
    examples_data = []
    test_schema = schema.FormatModeSchema.from_examples(examples_data)

    provider_config = test_schema.to_provider_config()

    self.assertEqual(
        provider_config,
        {"format": "json"},
        msg="Provider config should contain format: json",
    )

  def test_supports_strict_mode_returns_true(self):
    """Test that FormatModeSchema supports strict mode (valid JSON output)."""
    examples_data = []
    test_schema = schema.FormatModeSchema.from_examples(examples_data)

    self.assertTrue(
        test_schema.supports_strict_mode,
        msg="FormatModeSchema should support strict mode",
    )

  def test_different_examples_same_output(self):
    """Test that different examples produce the same schema for Ollama."""
    examples1 = [
        data.ExampleData(
            text="Text 1",
            extractions=[
                data.Extraction(
                    extraction_class="class1", extraction_text="text1"
                )
            ],
        )
    ]

    examples2 = [
        data.ExampleData(
            text="Text 2",
            extractions=[
                data.Extraction(
                    extraction_class="class2",
                    extraction_text="text2",
                    attributes={"attr": "value"},
                )
            ],
        )
    ]

    schema1 = schema.FormatModeSchema.from_examples(examples1)
    schema2 = schema.FormatModeSchema.from_examples(examples2)

    # Examples are ignored by FormatModeSchema
    self.assertEqual(
        schema1.to_provider_config(),
        schema2.to_provider_config(),
        msg="Different examples should produce same config for Ollama",
    )


class OllamaYAMLOverrideTest(absltest.TestCase):
  """Tests for Ollama YAML format override behavior."""

  def test_ollama_yaml_format_in_request_payload(self):
    """Test that YAML format override appears in Ollama request payload."""
    with mock.patch("requests.post", autospec=True) as mock_post:
      mock_response = mock.Mock(spec=["status_code", "json"])
      mock_response.status_code = 200
      mock_response.json.return_value = {"response": '{"extractions": []}'}
      mock_post.return_value = mock_response

      model = ollama.OllamaLanguageModel(model_id="gemma2:2b", format="yaml")

      list(model.infer(["Test prompt"]))

      mock_post.assert_called_once()
      call_kwargs = mock_post.call_args[1]
      self.assertIn(
          "json", call_kwargs, msg="Request should use json parameter"
      )
      payload = call_kwargs["json"]
      self.assertIn("format", payload, msg="Payload should contain format key")
      self.assertEqual(payload["format"], "yaml", msg="Format should be yaml")

  def test_yaml_override_sets_fence_output_true(self):
    """Test that overriding to YAML format sets fence_output to True."""

    examples_data = [
        data.ExampleData(
            text="Test text",
            extractions=[
                data.Extraction(
                    extraction_class="test_class",
                    extraction_text="test extraction",
                )
            ],
        )
    ]

    with mock.patch("requests.post", autospec=True) as mock_post:
      mock_response = mock.Mock(spec=["status_code", "json"])
      mock_response.status_code = 200
      mock_response.json.return_value = {"response": '{"extractions": []}'}
      mock_post.return_value = mock_response

      with mock.patch("langextract.providers.registry.resolve") as mock_resolve:
        mock_resolve.return_value = ollama.OllamaLanguageModel

        config = factory.ModelConfig(
            model_id="gemma2:2b",
            provider_kwargs={"format": "yaml"},
        )

        model = factory.create_model(
            config=config,
            examples=examples_data,
            use_schema_constraints=True,
            fence_output=None,  # Let it be computed
        )

        self.assertTrue(
            model.requires_fence_output, msg="YAML format should require fences"
        )

  def test_json_format_keeps_fence_output_false(self):
    """Test that JSON format keeps fence_output False."""

    examples_data = [
        data.ExampleData(
            text="Test text",
            extractions=[
                data.Extraction(
                    extraction_class="test_class",
                    extraction_text="test extraction",
                )
            ],
        )
    ]

    with mock.patch("requests.post", autospec=True) as mock_post:
      mock_response = mock.Mock(spec=["status_code", "json"])
      mock_response.status_code = 200
      mock_response.json.return_value = {"response": '{"extractions": []}'}
      mock_post.return_value = mock_response

      with mock.patch("langextract.providers.registry.resolve") as mock_resolve:
        mock_resolve.return_value = ollama.OllamaLanguageModel

        config = factory.ModelConfig(
            model_id="gemma2:2b",
            provider_kwargs={"format": "json"},
        )

        model = factory.create_model(
            config=config,
            examples=examples_data,
            use_schema_constraints=True,
            fence_output=None,  # Let it be computed
        )

        self.assertFalse(
            model.requires_fence_output,
            msg="JSON format should not require fences",
        )


class GeminiSchemaProviderIntegrationTest(absltest.TestCase):
  """Tests for GeminiSchema provider integration."""

  def test_gemini_schema_to_provider_config(self):
    """Test that GeminiSchema.to_provider_config includes response_schema."""
    examples_data = [
        data.ExampleData(
            text="Patient has diabetes",
            extractions=[
                data.Extraction(
                    extraction_class="condition",
                    extraction_text="diabetes",
                    attributes={"severity": "moderate"},
                )
            ],
        )
    ]

    gemini_schema = schema.GeminiSchema.from_examples(examples_data)
    provider_config = gemini_schema.to_provider_config()

    self.assertIn(
        "response_schema",
        provider_config,
        msg="GeminiSchema config should contain response_schema",
    )
    self.assertIsInstance(
        provider_config["response_schema"],
        dict,
        msg="response_schema should be a dictionary",
    )
    self.assertIn(
        "properties",
        provider_config["response_schema"],
        msg="response_schema should contain properties field",
    )

    self.assertIn(
        "response_mime_type",
        provider_config,
        msg="GeminiSchema config should contain response_mime_type",
    )
    self.assertEqual(
        provider_config["response_mime_type"],
        "application/json",
        msg="response_mime_type should be application/json",
    )

  def test_gemini_supports_strict_mode(self):
    """Test that GeminiSchema supports strict mode."""
    examples_data = []
    gemini_schema = schema.GeminiSchema.from_examples(examples_data)
    self.assertTrue(
        gemini_schema.supports_strict_mode,
        msg="GeminiSchema should support strict mode",
    )

  def test_gemini_rejects_yaml_with_schema(self):
    """Test that Gemini raises error when YAML format is used with schema."""

    examples_data = [
        data.ExampleData(
            text="Test",
            extractions=[
                data.Extraction(
                    extraction_class="test",
                    extraction_text="test text",
                )
            ],
        )
    ]
    test_schema = schema.GeminiSchema.from_examples(examples_data)

    with mock.patch("google.genai.Client", autospec=True):
      model = gemini_provider.GeminiLanguageModel(
          model_id="gemini-2.5-flash",
          api_key="test_key",
          format_type=data.FormatType.YAML,
      )
      model.apply_schema(test_schema)

      prompt = "Test prompt"
      config = {"temperature": 0.5}
      with self.assertRaises(exceptions.InferenceRuntimeError) as cm:
        _ = model._process_single_prompt(prompt, config)

      self.assertIn(
          "only supports JSON format",
          str(cm.exception),
          msg="Error should mention JSON-only constraint",
      )

  def test_gemini_forwards_schema_to_genai_client(self):
    """Test that GeminiLanguageModel forwards schema config to genai client."""

    examples_data = [
        data.ExampleData(
            text="Test",
            extractions=[
                data.Extraction(
                    extraction_class="test",
                    extraction_text="test text",
                )
            ],
        )
    ]
    test_schema = schema.GeminiSchema.from_examples(examples_data)

    with mock.patch("google.genai.Client", autospec=True) as mock_client:
      mock_model_instance = mock.Mock(spec=["return_value"])
      mock_client.return_value.models.generate_content = mock_model_instance
      mock_model_instance.return_value.text = '{"extractions": []}'

      model = gemini_provider.GeminiLanguageModel(
          model_id="gemini-2.5-flash",
          api_key="test_key",
          response_schema=test_schema.schema_dict,
          response_mime_type="application/json",
      )

      prompt = "Test prompt"
      config = {"temperature": 0.5}
      _ = model._process_single_prompt(prompt, config)

      mock_model_instance.assert_called_once()
      call_kwargs = mock_model_instance.call_args[1]
      self.assertIn(
          "config",
          call_kwargs,
          msg="genai.generate_content should receive config parameter",
      )
      self.assertIn(
          "response_schema",
          call_kwargs["config"],
          msg="Config should contain response_schema from GeminiSchema",
      )
      self.assertIn(
          "response_mime_type",
          call_kwargs["config"],
          msg="Config should contain response_mime_type",
      )
      self.assertEqual(
          call_kwargs["config"]["response_mime_type"],
          "application/json",
          msg="response_mime_type should be application/json",
      )

  def test_gemini_doesnt_forward_non_api_kwargs(self):
    """Test that GeminiLanguageModel doesn't forward non-API kwargs to genai."""

    with mock.patch("google.genai.Client", autospec=True) as mock_client:
      mock_model_instance = mock.Mock(spec=["return_value"])
      mock_client.return_value.models.generate_content = mock_model_instance
      mock_model_instance.return_value.text = '{"extractions": []}'

      model = gemini_provider.GeminiLanguageModel(
          model_id="gemini-2.5-flash",
          api_key="test_key",
          max_workers=5,
          response_schema={"test": "schema"},  # API parameter
      )

      prompt = "Test prompt"
      config = {"temperature": 0.5}
      _ = model._process_single_prompt(prompt, config)

      mock_model_instance.assert_called_once()
      call_kwargs = mock_model_instance.call_args[1]

      self.assertNotIn(
          "max_workers",
          call_kwargs["config"],
          msg="max_workers should not be forwarded to genai API config",
      )

      self.assertIn(
          "response_schema",
          call_kwargs["config"],
          msg="response_schema should be forwarded to genai API config",
      )


class SchemaShimTest(absltest.TestCase):
  """Tests for backward compatibility shims in schema module."""

  def test_extractions_key_import(self):
    """Test that EXTRACTIONS_KEY can be imported from schema module."""
    from langextract import schema as s  # pylint: disable=reimported,import-outside-toplevel

    self.assertEqual(
        s.EXTRACTIONS_KEY,
        "extractions",
        msg="EXTRACTIONS_KEY should be 'extractions'",
    )

  def test_constraint_types_import(self):
    """Test that Constraint and ConstraintType can be imported."""
    from langextract import schema as s  # pylint: disable=reimported,import-outside-toplevel

    constraint = s.Constraint()
    self.assertEqual(
        constraint.constraint_type,
        s.ConstraintType.NONE,
        msg="Default Constraint should have type NONE",
    )

    self.assertEqual(
        s.ConstraintType.NONE.value,
        "none",
        msg="ConstraintType.NONE should have value 'none'",
    )

  def test_provider_schema_imports(self):
    """Test that provider schemas can be imported from schema module."""
    from langextract import schema as s  # pylint: disable=reimported,import-outside-toplevel

    # Backward compatibility: re-exported from providers.schemas.gemini
    self.assertTrue(
        hasattr(s, "GeminiSchema"),
        msg=(
            "GeminiSchema should be importable from schema module for backward"
            " compatibility"
        ),
    )


if __name__ == "__main__":
  absltest.main()
