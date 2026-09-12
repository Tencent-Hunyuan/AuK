"""Split deployment for AuK: two nodes, one thin orchestrator.

    python -m auk.serve text-encoder --ckpt ckpts/AuK/auk_base.safetensors \
        --qwen_path ckpts/Qwen2.5-Omni-3B --device cuda:1 --port 8001

    python -m auk.serve worker --ckpt ckpts/AuK/auk_base.safetensors \
        --device cuda:0 --port 8002

    python -m auk.infer.infer_cli --text_encoder_url http://host:8001 \
        --worker_url http://host:8002 --instruction "..." -o out.wav

The split follows the two natural boundaries in the pipeline: the layer-fusion weights stay with
Qwen (otherwise 36 layers of hidden states, ~117 MB, cross the wire), and the ODE loop stays with
the DiT (otherwise sampling becomes one RPC per solver step).
"""

from auk.serve.client import AukDistributedInfer, RemoteTextEncoder, RemoteWorker
from auk.serve.nodes import TextEncoderNode, WorkerNode

__all__ = ["AukDistributedInfer", "RemoteTextEncoder", "RemoteWorker", "TextEncoderNode", "WorkerNode"]
