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

"""Ollama provider for LangExtract.

This provider enables using local Ollama models with LangExtract's extract() function.
No API key is required since Ollama runs locally on your machine.

Usage with extract():
    import langextract as lx
    from langextract.data import ExampleData, Extraction

    # Create an example for few-shot learning
    example = ExampleData(
        text="Marie Curie was a pioneering physicist and chemist.",
        extractions=[
            Extraction(
                extraction_class="person",
                extraction_text="Marie Curie",
                attributes={"name": "Marie Curie", "field": "physics and chemistry"}
            )
        ]
    )

    # Basic usage with Ollama
    result = lx.extract(
        text_or_documents="Isaac Asimov was a prolific science fiction writer.",
        model_id="gemma2:2b",
        prompt_description="Extract the person's name and field",
        examples=[example],
    )

Direct provider instantiation (when model ID conflicts with other providers):
    from langextract.providers.ollama import OllamaLanguageModel

    # Create Ollama provider directly
    model = OllamaLanguageModel(
        model_id="gemma2:2b",
        model_url="http://localhost:11434",  # optional, uses default if not specified
    )

    # Use with extract by passing the model instance
    result = lx.extract(
        text_or_documents="Your text here",
        model=model,  # Pass the model instance directly
        prompt_description="Extract information",
        examples=[example],
    )

Supported model ID formats:
    - Standard Ollama: llama3.2:1b, gemma2:2b, mistral:7b, qwen2.5:7b, etc.
    - Hugging Face style: meta-llama/Llama-3.2-1B-Instruct, google/gemma-2b, etc.

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull the model: ollama pull gemma2:2b
    3. Ollama server will start automatically when you use extract()
"""
# pylint: disable=cyclic-import,duplicate-code

from __future__ import annotations

import dataclasses
from typing import Any, Iterator, Mapping, Sequence
import warnings

import requests

from langextract import data
from langextract import exceptions
from langextract import inference
from langextract import schema
from langextract.providers import registry

_OLLAMA_DEFAULT_MODEL_URL = 'http://localhost:11434'


@registry.register(
    # Standard Ollama naming patterns
    r'^gemma',  # gemma2:2b, gemma2:9b, gemma2:27b, etc.
    r'^llama',  # llama3.2:1b, llama3.2:3b, llama3.1:8b, llama3.1:70b, etc.
    r'^mistral',  # mistral:7b, mistral-nemo:12b, mistral-large, etc.
    r'^mixtral',  # mixtral:8x7b, mixtral:8x22b, etc.
    r'^phi',  # phi3:mini, phi3:medium, phi3.5, etc.
    r'^qwen',  # qwen3:8b, qwen2.5:7b, qwen2.5:32b, qwen2.5-coder, etc.
    r'^deepseek',  # deepseek-r1:8b, deepseek-v3:671b, deepseek-coder-v2, etc.
    r'^command-r',  # command-r:35b, command-r-plus:104b, etc.
    r'^starcoder',  # starcoder2:3b, starcoder2:7b, starcoder2:15b, etc.
    r'^codellama',  # codellama:7b, codellama:13b, codellama:34b, etc.
    r'^codegemma',  # codegemma:2b, codegemma:7b, etc.
    r'^tinyllama',  # tinyllama:1.1b, etc.
    r'^wizardcoder',  # wizardcoder:7b, wizardcoder:13b, wizardcoder:34b, etc.
    r'^gpt-oss',  # gpt-oss:20b, etc.
    # Hugging Face style model IDs (organization/model-name)
    r'^meta-llama/[Ll]lama',  # meta-llama/Llama-3.2-1B-Instruct, etc.
    r'^google/gemma',  # google/gemma-2b, google/gemma-7b-it, etc.
    r'^mistralai/[Mm]istral',  # mistralai/Mistral-7B-v0.1, etc.
    r'^mistralai/[Mm]ixtral',  # mistralai/Mixtral-8x7B-v0.1, etc.
    r'^microsoft/phi',  # microsoft/phi-2, microsoft/phi-3-mini, etc.
    r'^Qwen/',  # Qwen/Qwen2.5-7B, Qwen/Qwen2.5-Coder, etc.
    r'^deepseek-ai/',  # deepseek-ai/deepseek-coder-v2, etc.
    r'^bigcode/starcoder',  # bigcode/starcoder2-3b, etc.
    r'^codellama/',  # codellama/CodeLlama-7b-Python, etc.
    r'^TinyLlama/',  # TinyLlama/TinyLlama-1.1B-Chat-v1.0, etc.
    r'^WizardLM/',  # WizardLM/WizardCoder-Python-7B-V1.0, etc.
    priority=10,
)
@dataclasses.dataclass(init=False)
class OllamaLanguageModel(inference.BaseLanguageModel):
  """Language model inference class using Ollama based host."""

  _model: str
  _model_url: str
  format_type: data.FormatType = data.FormatType.JSON
  _constraint: schema.Constraint = dataclasses.field(
      default_factory=schema.Constraint, repr=False, compare=False
  )
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  @classmethod
  def get_schema_class(cls) -> type[schema.BaseSchema] | None:
    """Return the FormatModeSchema class for JSON output support.

    Returns:
      The FormatModeSchema class that enables JSON mode (non-strict).
    """
    return schema.FormatModeSchema

  def __init__(
      self,
      model_id: str,
      model_url: str = _OLLAMA_DEFAULT_MODEL_URL,
      base_url: str | None = None,  # Alias for model_url
      format_type: data.FormatType | None = None,
      structured_output_format: str | None = None,  # Deprecated
      constraint: schema.Constraint = schema.Constraint(),
      **kwargs,
  ) -> None:
    """Initialize the Ollama language model.

    Args:
      model_id: The Ollama model ID to use.
      model_url: URL for Ollama server (legacy parameter).
      base_url: Alternative parameter name for Ollama server URL.
      format_type: Output format (JSON or YAML). Defaults to JSON.
      structured_output_format: DEPRECATED - use format_type instead.
      constraint: Schema constraints.
      **kwargs: Additional parameters.
    """
    self._requests = requests

    # Handle deprecated structured_output_format parameter
    if structured_output_format is not None:
      warnings.warn(
          "'structured_output_format' is deprecated and will be removed in "
          "v2.0.0. Use 'format_type' instead.",
          DeprecationWarning,
          stacklevel=2,
      )
      # Only use structured_output_format if format_type wasn't explicitly provided
      if format_type is None:
        format_type = (
            data.FormatType.JSON
            if structured_output_format == 'json'
            else data.FormatType.YAML
        )

    fmt = kwargs.pop('format', None)
    if format_type is None and fmt in ('json', 'yaml'):
      format_type = (
          data.FormatType.JSON if fmt == 'json' else data.FormatType.YAML
      )

    # Default to JSON if neither parameter was provided
    if format_type is None:
      format_type = data.FormatType.JSON

    self._model = model_id
    # Support both model_url and base_url parameters
    self._model_url = base_url or model_url or _OLLAMA_DEFAULT_MODEL_URL
    self.format_type = format_type
    self._constraint = constraint
    self._extra_kwargs = kwargs or {}
    super().__init__(constraint=constraint)

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[inference.ScoredOutput]]:
    """Runs inference on a list of prompts via Ollama's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params.

    Yields:
      Lists of ScoredOutputs.
    """
    for prompt in batch_prompts:
      try:
        response = self._ollama_query(
            prompt=prompt,
            model=self._model,
            structured_output_format='json'
            if self.format_type == data.FormatType.JSON
            else 'yaml',
            model_url=self._model_url,
            **kwargs,
        )
        # No score for Ollama. Default to 1.0
        yield [inference.ScoredOutput(score=1.0, output=response['response'])]
      except Exception as e:
        raise exceptions.InferenceRuntimeError(
            f'Ollama API error: {str(e)}', original=e
        ) from e

  def _ollama_query(
      self,
      prompt: str,
      model: str | None = None,
      temperature: float = 0.8,
      seed: int | None = None,
      top_k: int | None = None,
      max_output_tokens: int | None = None,
      structured_output_format: str | None = None,
      system: str = '',
      raw: bool = False,
      model_url: str | None = None,
      timeout: int = 30,
      keep_alive: int = 5 * 60,
      num_threads: int | None = None,
      num_ctx: int = 2048,
      **kwargs,  # pylint: disable=unused-argument
  ) -> Mapping[str, Any]:
    """Sends a prompt to an Ollama model and returns the generated response.

    This function makes an HTTP POST request to the `/api/generate` endpoint of
    an Ollama server. It can optionally load the specified model first, generate
    a response (with or without streaming), then return a parsed JSON response.

    Args:
      prompt: The text prompt to send to the model.
      model: The name of the model to use. Defaults to self._model.
      temperature: Sampling temperature. Higher values produce more diverse
        output.
      seed: Seed for reproducible generation. If None, random seed is used.
      top_k: The top-K parameter for sampling.
      max_output_tokens: Maximum tokens to generate. If None, the model's
        default is used.
      structured_output_format: If set to "json" or a JSON schema dict, requests
        structured outputs from the model. See Ollama documentation for details.
      system: A system prompt to override any system-level instructions.
      raw: If True, bypasses any internal prompt templating; you provide the
        entire raw prompt.
      model_url: The base URL for the Ollama server. Defaults to self._model_url.
      timeout: Timeout (in seconds) for the HTTP request.
      keep_alive: How long (in seconds) the model remains loaded after
        generation completes.
      num_threads: Number of CPU threads to use. If None, Ollama uses a default
        heuristic.
      num_ctx: Number of context tokens allowed. If None, uses model's default
        or config.
      **kwargs: Additional parameters passed through.

    Returns:
      A mapping (dictionary-like) containing the server's JSON response. For
      non-streaming calls, the `"response"` key typically contains the entire
      generated text.

    Raises:
      InferenceConfigError: If the server returns a 404 (model not found).
      InferenceRuntimeError: For any other HTTP errors, timeouts, or request
        exceptions.
    """
    model = model or self._model
    model_url = model_url or self._model_url
    if structured_output_format is None:
      structured_output_format = (
          'json' if self.format_type == data.FormatType.JSON else 'yaml'
      )

    options: dict[str, Any] = {'keep_alive': keep_alive}
    if seed:
      options['seed'] = seed
    if temperature:
      options['temperature'] = temperature
    if top_k:
      options['top_k'] = top_k
    if num_threads:
      options['num_thread'] = num_threads
    if max_output_tokens:
      options['num_predict'] = max_output_tokens
    if num_ctx:
      options['num_ctx'] = num_ctx

    api_url = model_url + '/api/generate'

    payload = {
        'model': model,
        'prompt': prompt,
        'system': system,
        'stream': False,
        'raw': raw,
        'format': structured_output_format,
        'options': options,
    }

    try:
      response = self._requests.post(
          api_url,
          headers={
              'Content-Type': 'application/json',
              'Accept': 'application/json',
          },
          json=payload,
          timeout=timeout,
      )
    except self._requests.exceptions.RequestException as e:
      if isinstance(e, self._requests.exceptions.ReadTimeout):
        msg = (
            f'Ollama Model timed out (timeout={timeout},'
            f' num_threads={num_threads})'
        )
        raise exceptions.InferenceRuntimeError(
            msg, original=e, provider='Ollama'
        ) from e
      raise exceptions.InferenceRuntimeError(
          f'Ollama request failed: {str(e)}', original=e, provider='Ollama'
      ) from e

    response.encoding = 'utf-8'
    if response.status_code == 200:
      return response.json()
    if response.status_code == 404:
      raise exceptions.InferenceConfigError(
          f"Can't find Ollama {model}. Try: ollama run {model}"
      )
    else:
      msg = f'Bad status code from Ollama: {response.status_code}'
      raise exceptions.InferenceRuntimeError(msg, provider='Ollama')
