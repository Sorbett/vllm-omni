# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import functools
from threading import Lock
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model


@functools.lru_cache(maxsize=1)
def _cosyvoice3_model_and_runner():
    """Defer heavy Omni/vLLM imports until a test runs (avoids duplicate CustomOp init)."""
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    return CosyVoice3Model, GPUARModelRunner


class _DummyCode2Wav:
    def __init__(
        self,
        vocab_size: int,
        num_samples: int = 32,
        outputs: list[tuple[torch.Tensor, dict[str, object] | None]] | None = None,
    ):
        self.input_embedding = SimpleNamespace(num_embeddings=vocab_size)
        self.num_samples = num_samples
        self.outputs = list(outputs or [])
        self.forward_calls: list[dict[str, object]] = []
        self.forward_streaming_calls: list[dict[str, object]] = []
        self.forward_streaming_batch_calls: list[list[dict[str, object]]] = []

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        token = kwargs["token"]
        num_samples = int(token.shape[-1])
        return torch.linspace(-1.0, 1.0, max(num_samples, 1), dtype=torch.float32).reshape(1, 1, -1)

    def forward_streaming(self, **kwargs):
        self.forward_streaming_calls.append(kwargs)
        if self.outputs:
            return self.outputs.pop(0)

        token = kwargs["token"]
        num_samples = int(token.shape[-1])
        audio = torch.linspace(-1.0, 1.0, max(num_samples, 1), dtype=torch.float32).reshape(1, 1, -1)
        new_state = None
        if not kwargs.get("finalize", False):
            new_state = {
                "mel": torch.ones((1, 80, max(num_samples, 1)), dtype=torch.float32),
                "speech_offset": audio.shape[-1],
            }
        return audio, new_state

    def forward_streaming_batch(self, items, *, n_timesteps: int = 10):
        self.forward_streaming_batch_calls.append(items)
        return [
            self.forward_streaming(
                token=item["token"],
                prompt_token=item["prompt_token"],
                prompt_feat=item["prompt_feat"],
                embedding=item["embedding"],
                cache_state=item.get("cache_state"),
                n_timesteps=n_timesteps,
                token_offset_tokens=int(item.get("token_offset_tokens", 0)),
                finalize=bool(item.get("finalize", False)),
            )
            for item in items
        ]


def _make_code2wav_model(
    *,
    with_stride_cfg: bool = False,
    num_samples: int = 32,
    outputs: list[tuple[torch.Tensor, dict[str, object] | None]] | None = None,
) -> CosyVoice3Model:
    CosyVoice3Model, _ = _cosyvoice3_model_and_runner()
    model = object.__new__(CosyVoice3Model)
    nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_code2wav"
    hift_cfg = {} if not with_stride_cfg else {"upsample_rates": [8, 5, 3], "istft_params": {"hop_len": 4}}
    model.config = SimpleNamespace(
        sample_rate=24000,
        hift=hift_cfg,
        token_frame_rate=25 if with_stride_cfg else 0,
        token_mel_ratio=2 if with_stride_cfg else 0,
    )
    model.code2wav = _DummyCode2Wav(vocab_size=4, num_samples=num_samples, outputs=outputs)
    # Short-circuit the lazy TensorRT estimator swap: these tests exercise the
    # forward audio logic, not the TRT path. On a GPU CI runner the swap would
    # otherwise run and dereference ``self.model_dir`` (only set in __init__,
    # which this fixture bypasses via object.__new__).
    model._code2wav_trt_done = True
    model.source_cache_len = 4
    model.speech_window = torch.hamming_window(8, periodic=False)
    model._stream_audio_cache_by_req = {}
    model._stream_audio_cache_lock = Lock()
    model._stream_vocoder_cache_by_req = {}
    return model


def _make_talker_model() -> CosyVoice3Model:
    CosyVoice3Model, _ = _cosyvoice3_model_and_runner()
    model = object.__new__(CosyVoice3Model)
    nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_talker"
    model.config = SimpleNamespace(
        llm={
            "speech_token_size": 6561,
            "eos_token_id": 6562,
            "sampling": {
                "top_p": 0.8,
                "top_k": 25,
                "win_size": 10,
                "tau_r": 0.1,
            },
        },
        vocab_size=151923,
    )
    return model


def _make_sampling_metadata(
    *,
    output_token_ids: list[list[int]],
    repetition_penalty: float = 2.0,
) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.tensor([1.0], dtype=torch.float32),
        all_greedy=False,
        all_random=True,
        top_p=torch.tensor([0.8], dtype=torch.float32),
        top_k=torch.tensor([25], dtype=torch.int32),
        generators={},
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1, dtype=torch.float32),
        presence_penalties=torch.zeros(1, dtype=torch.float32),
        repetition_penalties=torch.tensor([repetition_penalty], dtype=torch.float32),
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


@pytest.fixture
def async_output_vllm_config(tmp_path):
    from vllm.config import VllmConfig

    from vllm_omni.config.model import OmniModelConfig
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    hf_config = CosyVoice3Config()
    hf_config.llm.update(llm_input_size=16, llm_output_size=16, speech_token_size=32)
    # Exercise the real config fields without model downloads or engine setup.
    model_config = object.__new__(OmniModelConfig)
    model_config.hf_config = hf_config
    model_config.model_stage = "cosyvoice3_talker"
    model_config.model = str(tmp_path)
    model_config.async_chunk = True
    model_config.enable_return_routed_experts = False
    model_config.engine_output_type = "latent"
    model_config.stage_connector_config = {"name": "SharedMemoryConnector", "extra": {"role": "sender"}}
    vllm_config = object.__new__(VllmConfig)
    vllm_config.model_config = model_config
    return vllm_config


@pytest.fixture
def async_output_talker(monkeypatch, async_output_vllm_config):
    CosyVoice3Model, _ = _cosyvoice3_model_and_runner()
    from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3_talker

    class DummyEncoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, inputs_embeds, positions):
            return inputs_embeds

    class DummyTalker(nn.Module):
        def __init__(self, *, llm: nn.Module, **kwargs):
            super().__init__()
            self.llm = llm

    monkeypatch.setattr(cosyvoice3_talker, "VLLMQwen2Encoder", DummyEncoder)
    monkeypatch.setattr(cosyvoice3_talker, "CosyVoice3LM", DummyTalker)
    monkeypatch.setattr(CosyVoice3Model, "_create_llm_vllm_config", lambda self, config: config)

    return CosyVoice3Model(vllm_config=async_output_vllm_config)


@pytest.mark.parametrize(
    "async_chunk,async_scheduling,prefix_cache,expected",
    [(True, True, None, True), (False, True, None, False), (True, False, None, False), (True, True, object(), False)],
    ids=["enabled", "sync_chunks", "sync_scheduling", "prefix_cache"],
)
@pytest.mark.parametrize(
    "payload", [None, {}, {"embed": {"embedding": torch.ones(1, 2)}}], ids=["none", "empty", "prefill"]
)
@pytest.mark.parametrize("include_hidden", [False, True], ids=["token_payload", "hidden_payload"])
def test_talker_async_output_runtime_guards(
    async_output_talker,
    async_output_vllm_config,
    async_chunk,
    async_scheduling,
    prefix_cache,
    expected,
    payload,
    include_hidden,
):
    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
    runner = object.__new__(GPUARModelRunner)
    runner.use_async_scheduling = async_scheduling
    runner.omni_prefix_cache = prefix_cache
    runner.speculative_config = None
    runner.model_config = async_output_vllm_config.model_config
    runner.model_config.async_chunk = async_chunk
    runner.model = async_output_talker
    runner.model.omni_pooler_payload_include_hidden = include_hidden

    assert runner._should_use_async_omni_output(payload) is (expected and (include_hidden or bool(payload)))


@pytest.mark.parametrize("is_prefill", [True, False], ids=["prefill", "decode"])
def test_talker_async_output_preserves_conditioning_and_tokens(
    async_output_talker, async_output_vllm_config, monkeypatch, is_prefill
):
    """The deferred path must retain conditioning after reusable buffers change."""
    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

    from vllm_omni.worker.gpu_ar_model_runner import _snapshot_tensor_payload_to_cpu_async

    runner = object.__new__(GPUARModelRunner)
    runner.model = async_output_talker
    runner.vllm_config = async_output_vllm_config
    runner.model_config = async_output_vllm_config.model_config
    runner._async_chunk = True
    runner.omni_prefix_cache = None
    runner.supports_mm_inputs = False
    runner.routed_experts_initialized = False
    runner.model_intermediate_buffer = {}
    # The live batch has already advanced when the background builder runs.
    runner.input_batch = object.__new__(InputBatch)
    runner.input_batch._req_ids = ["next-request"]
    runner.input_batch.req_id_to_index = {"next-request": 0}
    monkeypatch.setattr(GPUARModelRunner, "_resolve_pooler_payload_req_ids", lambda self, req_ids: ("latent", req_ids))
    monkeypatch.setattr(GPUARModelRunner, "get_omni_connector_output", lambda self: None)

    conditioning = {
        "speech_token": torch.tensor([[11, 12], [21, 0]], dtype=torch.long),
        "speech_token_len": torch.tensor([2, 1], dtype=torch.long),
        "speech_feat": torch.arange(16, dtype=torch.float32).reshape(2, 4, 2),
        "embedding": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    expected = {key: value.clone() for key, value in conditioning.items()}
    model_output = async_output_talker.forward(
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.arange(3),
        inputs_embeds=torch.ones(3, 16),
        **(conditioning if is_prefill else {}),
    )
    payload = runner._build_omni_async_snapshot_payload(
        hidden_states=model_output.text_hidden_states,
        staged_hidden_states_cpu=None,
        multimodal_outputs=model_output.multimodal_outputs,
    )
    assert set(payload) == {"multimodal_outputs"}
    # CPU tensors use the same snapshot helper without requiring a CUDA stream.
    snapshot = _snapshot_tensor_payload_to_cpu_async(payload, copy_stream=None, pin_memory=False)
    for tensor in conditioning.values():
        tensor.zero_()
    model_output.text_hidden_states.zero_()
    snapshot.wait()

    scheduler_output = object.__new__(SchedulerOutput)
    scheduler_output.total_num_scheduled_tokens = 3
    scheduler_output.num_scheduled_tokens = {"r1": 2, "r2": 1}
    output = runner._build_omni_model_runner_output_from_snapshot(
        scheduler_output=scheduler_output,
        hidden_states=model_output.text_hidden_states[:0],
        staged_hidden_states_cpu=None,
        multimodal_outputs=snapshot.payload["multimodal_outputs"],
        req_ids_output_copy=["r1", "r2"],
        req_id_to_index_output_copy={"r1": 0, "r2": 1},
        valid_sampled_token_ids=[[101], [102]],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=None,
        num_scheduled_tokens_np=torch.tensor([2, 1], dtype=torch.int32).numpy(),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.long),
    )
    assert output.req_ids == ["r1", "r2"]
    assert output.sampled_token_ids == [[101], [102]]
    assert output.multimodal_outputs is None
    if not is_prefill:
        assert not output.inter_stage_outputs or all(not item for item in output.inter_stage_outputs)
        return

    assert len(output.inter_stage_outputs) == 2
    for idx, prompt_len in enumerate([2, 1]):
        item = output.inter_stage_outputs[idx]
        assert "hidden" not in item
        # The downstream processor removes padding using speech_token_len.
        torch.testing.assert_close(item["embed.speech_token"], expected["speech_token"][idx : idx + 1])
        torch.testing.assert_close(item["embed.speech_feat"], expected["speech_feat"][idx : idx + 1])
        assert item["embed.speech_token_len"].item() == prompt_len
        torch.testing.assert_close(item["embed.embedding"], expected["embedding"][idx : idx + 1])


def test_forward_prefers_token_offset_when_present():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"left_context_size": 2},
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert len(out.multimodal_outputs["audio"]) == 1
    assert out.multimodal_outputs["audio"][0].numel() > 0
    assert len(model.code2wav.forward_streaming_calls) == 1
    call = model.code2wav.forward_streaming_calls[0]
    assert call["token"].shape == (1, 3)
    assert call["token_offset_tokens"] == 2
    assert call["finalize"] is False


def test_forward_falls_back_to_left_context_size_for_backward_compat():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"left_context_size": 2},
        }
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert model.code2wav.forward_streaming_calls[0]["token_offset_tokens"] == 2


def test_forward_ignores_single_request_padded_tail_tokens():
    model = _make_code2wav_model(with_stride_cfg=True)
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"left_context_size": 0},
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2, 3, 3], dtype=torch.long),
        positions=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    # The padded tail must not contribute to code2wav length.
    assert out.multimodal_outputs["audio"][0].numel() == 3
    assert model.code2wav.forward_streaming_calls[0]["token"].tolist() == [[0, 1, 2]]


def test_forward_uses_non_stream_decode_without_chunk_metadata():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "ids": {"prompt": [101, 102]},
            "generated_len": 3,
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert out.multimodal_outputs["audio"][0].numel() == 3
    assert len(model.code2wav.forward_calls) == 1
    assert len(model.code2wav.forward_streaming_calls) == 0
    call = model.code2wav.forward_calls[0]
    assert call["token"].tolist() == [[0, 1, 2]]
    assert call["token_offset_tokens"] == 0


def test_forward_uses_non_stream_talker_prefill_offset():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"talker_prefill_offset": 3},
        }
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert model.code2wav.forward_calls[0]["token_offset_tokens"] == 3


def test_forward_reuses_streaming_cache_state_between_chunks():
    model = _make_code2wav_model(
        outputs=[
            (
                torch.arange(4, dtype=torch.float32).reshape(1, 1, -1),
                {"mel": torch.ones((1, 80, 3), dtype=torch.float32), "speech_offset": 4},
            ),
            (
                torch.full((1, 1, 2), 9.0, dtype=torch.float32),
                {"mel": torch.ones((1, 80, 5), dtype=torch.float32), "speech_offset": 6},
            ),
        ]
    )
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {
                "req_id": ["rid-stream"],
                "stream_finished": torch.tensor(False),
                "left_context_size": 0,
            },
        }
    ]

    out1 = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert out1.multimodal_outputs["audio"][0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert model.code2wav.forward_streaming_calls[0]["cache_state"] is None

    out2 = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert out2.multimodal_outputs["audio"][0].tolist() == [9.0, 9.0]
    cache_state = model.code2wav.forward_streaming_calls[1]["cache_state"]
    assert cache_state is not None
    assert cache_state["speech_offset"] == 4
    assert "rid-stream" in model._stream_vocoder_cache_by_req


def test_forward_clears_streaming_cache_on_terminal_chunk():
    model = _make_code2wav_model(
        outputs=[
            (
                torch.arange(4, dtype=torch.float32).reshape(1, 1, -1),
                {"mel": torch.ones((1, 80, 3), dtype=torch.float32), "speech_offset": 4},
            ),
            (
                torch.full((1, 1, 1), 7.0, dtype=torch.float32),
                None,
            ),
        ]
    )
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {
                "req_id": ["rid-stream"],
                "stream_finished": torch.tensor(False),
                "left_context_size": 0,
            },
        }
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert "rid-stream" in model._stream_vocoder_cache_by_req

    runtime_info[0]["meta"]["stream_finished"] = torch.tensor(True)
    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert out.multimodal_outputs["audio"][0].tolist() == [7.0]
    assert "rid-stream" not in model._stream_vocoder_cache_by_req


def test_forward_batches_streaming_flow_items(monkeypatch):
    monkeypatch.setenv("COSYVOICE3_BATCH_FLOW", "1")
    model = _make_code2wav_model()
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {
                "req_id": ["rid-a"],
                "stream_finished": torch.tensor(False),
                "left_context_size": 0,
            },
        },
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.7, 0.8]], dtype=torch.float32),
            },
            "meta": {
                "req_id": ["rid-b"],
                "stream_finished": torch.tensor(False),
                "left_context_size": 1,
            },
        },
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long),
        positions=torch.arange(6, dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3, 3],
    )

    assert len(out.multimodal_outputs["audio"]) == 2
    assert len(model.code2wav.forward_streaming_batch_calls) == 1
    batch_items = model.code2wav.forward_streaming_batch_calls[0]
    assert [item["index"] for item in batch_items] == [0, 1]
    assert torch.equal(batch_items[0]["token"], torch.tensor([[0, 1, 2]]))
    assert batch_items[1]["token_offset_tokens"] == 1
    assert "rid-a" in model._stream_vocoder_cache_by_req
    assert "rid-b" in model._stream_vocoder_cache_by_req


def test_sample_uses_ras_rejection_for_recent_repetition():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10])
    logits = torch.tensor([[-1e9, 10.0, 0.0]], dtype=torch.float32)

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.tolist() == [[2]]


def test_sample_tolerates_padded_rows_without_history():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10])
    logits = torch.tensor(
        [
            [-1e9, 10.0, 0.0],
            [-1e9, 0.0, 10.0],
        ],
        dtype=torch.float32,
    )

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.shape == (2, 1)


def test_sample_excludes_non_finite_logits():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[]])
    metadata.temperature.fill_(0.5)
    logits = torch.tensor([[float("nan"), 1.0, float("inf"), float("-inf")]], dtype=torch.bfloat16)

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.tolist() == [[1]]


def test_sample_preserves_allowed_token_mask_with_invalid_logits():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[]])
    metadata.allowed_token_ids_mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[float("nan"), 1.0, float("inf")]], dtype=torch.float32)

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.tolist() == [[1]]


def test_sample_rejects_rows_without_finite_logits():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[]])
    metadata.allowed_token_ids_mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[0.0, float("nan"), 0.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="no finite logits"):
        model.sample(logits, metadata)


def test_sample_keeps_only_finite_token_after_ras_rejection():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10])
    metadata.allowed_token_ids_mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.tolist() == [[1]]


def test_gpu_ar_model_runner_prefers_model_sampler_when_opted_in():
    metadata = _make_sampling_metadata(output_token_ids=[[1, 2, 3]])
    expected = SamplerOutput(
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        logprobs_tensors=None,
    )
    calls: list[torch.Tensor] = []

    class _DummyInputBatch:
        def __init__(self):
            self.sampling_metadata = metadata
            self.updated = False

        def update_async_output_token_ids(self):
            # After PR 3681 fix, update_async_output_token_ids is called
            # BEFORE model sampler path to ensure async placeholder repair
            # runs for all sampling paths
            self.updated = True

    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = _DummyInputBatch()

    def model_sample(logits, sampling_metadata):
        calls.append(logits.clone())
        return expected

    runner.model = SimpleNamespace(
        prefer_model_sampler=True,
        sample=model_sample,
    )
    runner.sampler = lambda **_: (_ for _ in ()).throw(AssertionError("fallback sampler should not be used"))

    out = runner._sample(torch.tensor([[0.1, 0.2]], dtype=torch.float32), spec_decode_metadata=None)

    assert out is expected
    assert runner.input_batch.updated is True
    assert len(calls) == 1


def test_gpu_ar_model_runner_supplies_req_output_history_to_model_sampler():
    metadata = _make_sampling_metadata(output_token_ids=[])
    seen_histories: list[list[list[int]]] = []

    class _DummyInputBatch:
        def __init__(self):
            self.sampling_metadata = metadata
            self.req_output_token_ids = [[1, 2, 3]]
            self.req_ids = ["rid-1"]
            self.sampled_token_ids_cpu = None
            self.async_copy_ready_event = None
            self.prev_req_id_to_index = None
            self.update_async_called = False

        def update_async_output_token_ids(self):
            # After PR 3681 fix, update_async_output_token_ids is called
            # BEFORE model sampler path to ensure async placeholder repair
            # runs for all sampling paths
            self.update_async_called = True

    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = _DummyInputBatch()

    def model_sample(logits, sampling_metadata):
        seen_histories.append([list(x) for x in sampling_metadata.output_token_ids])
        return SamplerOutput(sampled_token_ids=torch.tensor([[7]], dtype=torch.int32), logprobs_tensors=None)

    runner.model = SimpleNamespace(
        prefer_model_sampler=True,
        sample=model_sample,
    )
    runner.sampler = lambda **_: (_ for _ in ()).throw(AssertionError("fallback sampler should not be used"))

    runner._sample(torch.tensor([[0.1, 0.2]], dtype=torch.float32), spec_decode_metadata=None)

    assert runner.input_batch.update_async_called is True
    assert seen_histories == [[[1, 2, 3]]]


def test_gpu_ar_model_runner_repairs_async_placeholders_for_model_sampler():
    metadata = _make_sampling_metadata(output_token_ids=[])
    seen_histories: list[list[list[int]]] = []

    class _ReadyEvent:
        def __init__(self):
            self.synced = False

        def synchronize(self):
            self.synced = True

    class _DummyInputBatch:
        def __init__(self):
            self.sampling_metadata = metadata
            self.req_output_token_ids = [[11, -1]]
            self.req_ids = ["rid-1"]
            self.sampled_token_ids_cpu = torch.tensor([[29]], dtype=torch.int32)
            self.async_copy_ready_event = _ReadyEvent()
            self.prev_req_id_to_index = {"rid-1": 0}
            self.update_async_called = False

        def update_async_output_token_ids(self):
            # After PR 3681 fix, update_async_output_token_ids is called
            # BEFORE model sampler path to ensure async placeholder repair
            # runs for all sampling paths (model sampler + fallback sampler)
            self.update_async_called = True

    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = _DummyInputBatch()

    def model_sample(logits, sampling_metadata):
        seen_histories.append([list(x) for x in sampling_metadata.output_token_ids])
        return SamplerOutput(sampled_token_ids=torch.tensor([[7]], dtype=torch.int32), logprobs_tensors=None)

    runner.model = SimpleNamespace(
        prefer_model_sampler=True,
        sample=model_sample,
    )
    runner.sampler = lambda **_: (_ for _ in ()).throw(AssertionError("fallback sampler should not be used"))

    runner._sample(torch.tensor([[0.1, 0.2]], dtype=torch.float32), spec_decode_metadata=None)

    assert runner.input_batch.async_copy_ready_event.synced is True
    assert runner.input_batch.update_async_called is True
    assert seen_histories == [[[11, 29]]]
