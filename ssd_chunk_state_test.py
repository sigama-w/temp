# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test cases for operators in ssd_chunk_state.py

This test file contains multiple test cases for the following functions:
- _chunk_cumsum_fwd: Compute chunked cumsum of A * dt
- _chunk_state_fwd: Compute the state for each intra-chunk
- chunk_state_varlen: Compute varlen states for continuous batching
"""

import math
import torch
import pytest

from ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
    chunk_state_varlen,
)


def create_cumsum_inputs(
    batch_size: int,
    seqlen: int,
    nheads: int,
    chunk_size: int,
    dtype: torch.dtype = torch.float16,
    device: str = "npu",
):
    """Create input tensors for _chunk_cumsum_fwd."""
    dt = torch.randn(batch_size, seqlen, nheads, dtype=dtype, device=device)
    A = torch.randn(nheads, dtype=torch.float32, device=device)
    return {"dt": dt, "A": A}


def create_state_inputs(
    batch_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    ngroups: int,
    dstate: int,
    chunk_size: int,
    dtype: torch.dtype = torch.float16,
    device: str = "npu",
):
    """Create input tensors for _chunk_state_fwd."""
    nchunks = math.ceil(seqlen / chunk_size)

    B = torch.randn(batch_size, seqlen, ngroups, dstate, dtype=dtype, device=device)
    x = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    dt = torch.randn(batch_size, nheads, nchunks, chunk_size, dtype=torch.float32, device=device)
    dA_cumsum = torch.randn(batch_size, nheads, nchunks, chunk_size, dtype=torch.float32, device=device)

    return {
        "B": B,
        "x": x,
        "dt": dt,
        "dA_cumsum": dA_cumsum,
        "nchunks": nchunks,
    }


class TestChunkCumsumFwd:
    """Test class for _chunk_cumsum_fwd operator."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("seqlen", [64, 128, 256, 512])
    def test_basic_multi_batch(self, batch_size: int, seqlen: int):
        """Test basic functionality with different batch sizes and sequence lengths."""
        nheads = 8
        chunk_size = 64

        inputs = create_cumsum_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
        )

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt=inputs["dt"],
            A=inputs["A"],
            chunk_size=chunk_size,
            dt_bias=None,
            dt_softplus=False,
            dt_limit=(0.0, float("inf")),
        )

        nchunks = math.ceil(seqlen / chunk_size)
        assert dA_cumsum.shape == (batch_size, nheads, nchunks, chunk_size)
        assert dt_out.shape == (batch_size, nheads, nchunks, chunk_size)

        print(
            f"[PASS] _chunk_cumsum_fwd basic test: batch_size={batch_size}, "
            f"seqlen={seqlen}, dA_cumsum.shape={dA_cumsum.shape}, dt_out.shape={dt_out.shape}"
        )

    @pytest.mark.parametrize("nheads", [4, 8, 16, 32])
    def test_different_nheads(self, nheads: int):
        """Test with different number of heads."""
        batch_size = 2
        seqlen = 128
        chunk_size = 64

        inputs = create_cumsum_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
        )

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt=inputs["dt"],
            A=inputs["A"],
            chunk_size=chunk_size,
            dt_bias=None,
            dt_softplus=False,
            dt_limit=(0.0, float("inf")),
        )

        nchunks = math.ceil(seqlen / chunk_size)
        assert dA_cumsum.shape == (batch_size, nheads, nchunks, chunk_size)

        print(
            f"[PASS] _chunk_cumsum_fwd nheads test: nheads={nheads}, "
            f"dA_cumsum.shape={dA_cumsum.shape}"
        )

    @pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
    def test_different_chunk_sizes(self, chunk_size: int):
        """Test with different chunk sizes."""
        batch_size = 2
        nheads = 8
        seqlen = 256

        inputs = create_cumsum_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
        )

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt=inputs["dt"],
            A=inputs["A"],
            chunk_size=chunk_size,
            dt_bias=None,
            dt_softplus=False,
            dt_limit=(0.0, float("inf")),
        )

        nchunks = math.ceil(seqlen / chunk_size)
        assert dA_cumsum.shape == (batch_size, nheads, nchunks, chunk_size)

        print(
            f"[PASS] _chunk_cumsum_fwd chunk_size test: chunk_size={chunk_size}, "
            f"nchunks={nchunks}, dA_cumsum.shape={dA_cumsum.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_with_dt_bias(self, batch_size: int):
        """Test with dt_bias parameter."""
        nheads = 8
        seqlen = 128
        chunk_size = 64

        inputs = create_cumsum_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
        )

        dt_bias = torch.randn(nheads, dtype=torch.float32, device="npu")

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt=inputs["dt"],
            A=inputs["A"],
            chunk_size=chunk_size,
            dt_bias=dt_bias,
            dt_softplus=False,
            dt_limit=(0.0, float("inf")),
        )

        nchunks = math.ceil(seqlen / chunk_size)
        assert dA_cumsum.shape == (batch_size, nheads, nchunks, chunk_size)

        print(
            f"[PASS] _chunk_cumsum_fwd with dt_bias: batch_size={batch_size}, "
            f"dA_cumsum.shape={dA_cumsum.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_with_dt_softplus(self, batch_size: int):
        """Test with dt_softplus parameter."""
        nheads = 8
        seqlen = 128
        chunk_size = 64

        inputs = create_cumsum_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
        )

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt=inputs["dt"],
            A=inputs["A"],
            chunk_size=chunk_size,
            dt_bias=None,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
        )

        nchunks = math.ceil(seqlen / chunk_size)
        assert dA_cumsum.shape == (batch_size, nheads, nchunks, chunk_size)

        print(
            f"[PASS] _chunk_cumsum_fwd with dt_softplus: batch_size={batch_size}, "
            f"dA_cumsum.shape={dA_cumsum.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_with_dt_limit(self, batch_size: int):
        """Test with dt_limit parameter to clamp dt values."""
        nheads = 8
        seqlen = 128
        chunk_size = 64

        inputs = create_cumsum_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
        )

        dt_limit = (0.001, 0.1)

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt=inputs["dt"],
            A=inputs["A"],
            chunk_size=chunk_size,
            dt_bias=None,
            dt_softplus=False,
            dt_limit=dt_limit,
        )

        nchunks = math.ceil(seqlen / chunk_size)
        assert dA_cumsum.shape == (batch_size, nheads, nchunks, chunk_size)

        print(
            f"[PASS] _chunk_cumsum_fwd with dt_limit: batch_size={batch_size}, "
            f"dt_limit={dt_limit}, dA_cumsum.shape={dA_cumsum.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_with_all_options(self, batch_size: int):
        """Test with all parameters."""
        nheads = 8
        seqlen = 128
        chunk_size = 64

        inputs = create_cumsum_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
        )

        dt_bias = torch.randn(nheads, dtype=torch.float32, device="npu")
        dt_limit = (0.001, 0.1)

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt=inputs["dt"],
            A=inputs["A"],
            chunk_size=chunk_size,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=dt_limit,
        )

        nchunks = math.ceil(seqlen / chunk_size)
        assert dA_cumsum.shape == (batch_size, nheads, nchunks, chunk_size)

        print(
            f"[PASS] _chunk_cumsum_fwd with all options: batch_size={batch_size}, "
            f"dA_cumsum.shape={dA_cumsum.shape}"
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_different_dtypes(self, dtype: torch.dtype):
        """Test with different data types."""
        batch_size = 2
        nheads = 8
        seqlen = 128
        chunk_size = 64

        if dtype == torch.bfloat16 and not torch.npu.is_bf16_supported():
            pytest.skip("BF16 not supported on this GPU")

        inputs = create_cumsum_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
            dtype=dtype,
        )

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt=inputs["dt"],
            A=inputs["A"],
            chunk_size=chunk_size,
            dt_bias=None,
            dt_softplus=False,
            dt_limit=(0.0, float("inf")),
        )

        nchunks = math.ceil(seqlen / chunk_size)
        assert dA_cumsum.shape == (batch_size, nheads, nchunks, chunk_size)

        print(
            f"[PASS] _chunk_cumsum_fwd dtype test: dtype={dtype}, "
            f"dA_cumsum.shape={dA_cumsum.shape}"
        )


class TestChunkStateFwd:
    """Test class for _chunk_state_fwd operator."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seqlen", [64, 128, 256])
    def test_basic_multi_batch(self, batch_size: int, seqlen: int):
        """Test basic functionality with different batch sizes and sequence lengths."""
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64

        inputs = create_state_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        states = _chunk_state_fwd(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            seq_idx=None,
            states=None,
            states_in_fp32=True,
        )

        assert states.shape == (batch_size, inputs["nchunks"], nheads, headdim, dstate)
        assert states.dtype == torch.float32

        print(
            f"[PASS] _chunk_state_fwd basic test: batch_size={batch_size}, "
            f"seqlen={seqlen}, states.shape={states.shape}"
        )

    @pytest.mark.parametrize("headdim", [32, 64, 128])
    def test_different_headdim(self, headdim: int):
        """Test with different head dimensions."""
        batch_size = 2
        nheads = 8
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_state_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        states = _chunk_state_fwd(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            seq_idx=None,
            states=None,
            states_in_fp32=True,
        )

        assert states.shape == (batch_size, inputs["nchunks"], nheads, headdim, dstate)

        print(
            f"[PASS] _chunk_state_fwd headdim test: headdim={headdim}, "
            f"states.shape={states.shape}"
        )

    @pytest.mark.parametrize("dstate", [8, 16, 32, 64, 128])
    def test_different_dstate(self, dstate: int):
        """Test with different state dimensions."""
        batch_size = 2
        nheads = 8
        headdim = 64
        ngroups = 2
        chunk_size = 64
        seqlen = 128

        inputs = create_state_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        states = _chunk_state_fwd(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            seq_idx=None,
            states=None,
            states_in_fp32=True,
        )

        assert states.shape == (batch_size, inputs["nchunks"], nheads, headdim, dstate)

        print(
            f"[PASS] _chunk_state_fwd dstate test: dstate={dstate}, "
            f"states.shape={states.shape}"
        )

    @pytest.mark.parametrize("ngroups", [1, 2, 4, 8])
    def test_different_ngroups(self, ngroups: int):
        """Test with different ngroups values."""
        batch_size = 2
        nheads = 8
        headdim = 64
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_state_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        states = _chunk_state_fwd(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            seq_idx=None,
            states=None,
            states_in_fp32=True,
        )

        assert states.shape == (batch_size, inputs["nchunks"], nheads, headdim, dstate)

        print(
            f"[PASS] _chunk_state_fwd ngroups test: ngroups={ngroups}, "
            f"states.shape={states.shape}"
        )

    @pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
    def test_different_chunk_sizes(self, chunk_size: int):
        """Test with different chunk sizes."""
        batch_size = 2
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        seqlen = 256

        inputs = create_state_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        states = _chunk_state_fwd(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            seq_idx=None,
            states=None,
            states_in_fp32=True,
        )

        assert states.shape == (batch_size, inputs["nchunks"], nheads, headdim, dstate)

        print(
            f"[PASS] _chunk_state_fwd chunk_size test: chunk_size={chunk_size}, "
            f"nchunks={inputs['nchunks']}, states.shape={states.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_with_seq_idx(self, batch_size: int):
        """Test with seq_idx parameter (continuous batching)."""
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_state_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        seq_idx = torch.zeros(batch_size, seqlen, dtype=torch.int32, device="npu")
        seq_idx[:, seqlen // 2 :] = 1

        states = _chunk_state_fwd(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            seq_idx=seq_idx,
            states=None,
            states_in_fp32=True,
        )

        assert states.shape == (batch_size, inputs["nchunks"], nheads, headdim, dstate)

        print(
            f"[PASS] _chunk_state_fwd with seq_idx: batch_size={batch_size}, "
            f"seq_idx unique values={torch.unique(seq_idx).tolist()}, states.shape={states.shape}"
        )

    def test_with_preallocated_states(self):
        """Test with preallocated states tensor."""
        batch_size = 2
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_state_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        preallocated_states = torch.empty(
            batch_size, inputs["nchunks"], nheads, headdim, dstate,
            dtype=torch.float32, device="npu"
        )

        states = _chunk_state_fwd(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            seq_idx=None,
            states=preallocated_states,
            states_in_fp32=True,
        )

        assert states.shape == preallocated_states.shape
        assert states.data_ptr() == preallocated_states.data_ptr()

        print(
            f"[PASS] _chunk_state_fwd with preallocated states: "
            f"states.shape={states.shape}, preallocated match=True"
        )

    def test_states_in_fp32_false(self):
        """Test with states_in_fp32=False."""
        batch_size = 2
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_state_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        states = _chunk_state_fwd(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            seq_idx=None,
            states=None,
            states_in_fp32=False,
        )

        assert states.shape == (batch_size, inputs["nchunks"], nheads, headdim, dstate)
        assert states.dtype == inputs["B"].dtype

        print(
            f"[PASS] _chunk_state_fwd states_in_fp32=False: "
            f"states.shape={states.shape}, states.dtype={states.dtype}"
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_different_dtypes(self, dtype: torch.dtype):
        """Test with different data types."""
        batch_size = 2
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        if dtype == torch.bfloat16 and not torch.npu.is_bf16_supported():
            pytest.skip("BF16 not supported on this GPU")

        inputs = create_state_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
            dtype=dtype,
        )

        states = _chunk_state_fwd(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            seq_idx=None,
            states=None,
            states_in_fp32=True,
        )

        assert states.shape == (batch_size, inputs["nchunks"], nheads, headdim, dstate)

        print(
            f"[PASS] _chunk_state_fwd dtype test: dtype={dtype}, "
            f"states.shape={states.shape}"
        )


class TestChunkStateVarlen:
    """Test class for chunk_state_varlen operator."""

    def create_varlen_inputs(
        self,
        total_seqlen: int,
        nheads: int,
        headdim: int,
        ngroups: int,
        dstate: int,
        chunk_size: int,
        num_sequences: int,
        dtype: torch.dtype = torch.float16,
        device: str = "npu",
    ):
        """Create input tensors for chunk_state_varlen."""
        nchunks = math.ceil(total_seqlen / chunk_size)

        B = torch.randn(total_seqlen, ngroups, dstate, dtype=dtype, device=device)
        x = torch.randn(total_seqlen, nheads, headdim, dtype=dtype, device=device)
        dt = torch.randn(nheads, nchunks, chunk_size, dtype=torch.float32, device=device)
        dA_cumsum = torch.randn(nheads, nchunks, chunk_size, dtype=torch.float32, device=device)
        chunk_states = torch.randn(nchunks, nheads, headdim, dstate, dtype=torch.float32, device=device)

        sequence_lengths = torch.randint(chunk_size, total_seqlen // num_sequences + chunk_size, (num_sequences,))
        sequence_lengths = sequence_lengths.tolist()
        cu_seqlens = [0]
        for length in sequence_lengths:
            cu_seqlens.append(min(cu_seqlens[-1] + length, total_seqlen))
        cu_seqlens[-1] = total_seqlen
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

        return {
            "B": B,
            "x": x,
            "dt": dt,
            "dA_cumsum": dA_cumsum,
            "chunk_states": chunk_states,
            "cu_seqlens": cu_seqlens,
            "nchunks": nchunks,
            "num_sequences": num_sequences,
        }

    def test_basic_varlen(self):
        """Test basic varlen functionality."""
        total_seqlen = 256
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        num_sequences = 4

        inputs = self.create_varlen_inputs(
            total_seqlen=total_seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
            num_sequences=num_sequences,
        )

        states = chunk_state_varlen(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            cu_seqlens=inputs["cu_seqlens"],
            chunk_states=inputs["chunk_states"],
            initial_states=None,
        )

        assert states.shape == (num_sequences, nheads, headdim, dstate)

        print(
            f"[PASS] chunk_state_varlen basic test: total_seqlen={total_seqlen}, "
            f"num_sequences={num_sequences}, states.shape={states.shape}"
        )

    @pytest.mark.parametrize("num_sequences", [1, 2, 4, 8])
    def test_different_num_sequences(self, num_sequences: int):
        """Test with different number of sequences."""
        total_seqlen = 256
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64

        inputs = self.create_varlen_inputs(
            total_seqlen=total_seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
            num_sequences=num_sequences,
        )

        states = chunk_state_varlen(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            cu_seqlens=inputs["cu_seqlens"],
            chunk_states=inputs["chunk_states"],
            initial_states=None,
        )

        assert states.shape == (inputs["num_sequences"], nheads, headdim, dstate)

        print(
            f"[PASS] chunk_state_varlen num_sequences test: num_sequences={inputs['num_sequences']}, "
            f"states.shape={states.shape}"
        )

    @pytest.mark.parametrize("headdim", [32, 64, 128])
    def test_different_headdim(self, headdim: int):
        """Test with different head dimensions."""
        total_seqlen = 256
        nheads = 8
        ngroups = 2
        dstate = 16
        chunk_size = 64
        num_sequences = 4

        inputs = self.create_varlen_inputs(
            total_seqlen=total_seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
            num_sequences=num_sequences,
        )

        states = chunk_state_varlen(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            cu_seqlens=inputs["cu_seqlens"],
            chunk_states=inputs["chunk_states"],
            initial_states=None,
        )

        assert states.shape == (num_sequences, nheads, headdim, dstate)

        print(
            f"[PASS] chunk_state_varlen headdim test: headdim={headdim}, "
            f"states.shape={states.shape}"
        )

    @pytest.mark.parametrize("dstate", [8, 16, 32, 64])
    def test_different_dstate(self, dstate: int):
        """Test with different state dimensions."""
        total_seqlen = 256
        nheads = 8
        headdim = 64
        ngroups = 2
        chunk_size = 64
        num_sequences = 4

        inputs = self.create_varlen_inputs(
            total_seqlen=total_seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
            num_sequences=num_sequences,
        )

        states = chunk_state_varlen(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            cu_seqlens=inputs["cu_seqlens"],
            chunk_states=inputs["chunk_states"],
            initial_states=None,
        )

        assert states.shape == (num_sequences, nheads, headdim, dstate)

        print(
            f"[PASS] chunk_state_varlen dstate test: dstate={dstate}, "
            f"states.shape={states.shape}"
        )

    @pytest.mark.parametrize("chunk_size", [32, 64, 128])
    def test_different_chunk_sizes(self, chunk_size: int):
        """Test with different chunk sizes."""
        total_seqlen = 256
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        num_sequences = 4

        inputs = self.create_varlen_inputs(
            total_seqlen=total_seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
            num_sequences=num_sequences,
        )

        states = chunk_state_varlen(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            cu_seqlens=inputs["cu_seqlens"],
            chunk_states=inputs["chunk_states"],
            initial_states=None,
        )

        assert states.shape == (num_sequences, nheads, headdim, dstate)

        print(
            f"[PASS] chunk_state_varlen chunk_size test: chunk_size={chunk_size}, "
            f"nchunks={inputs['nchunks']}, states.shape={states.shape}"
        )

    def test_with_initial_states(self):
        """Test with initial_states parameter."""
        total_seqlen = 256
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        num_sequences = 4

        inputs = self.create_varlen_inputs(
            total_seqlen=total_seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
            num_sequences=num_sequences,
        )

        initial_states = torch.randn(
            num_sequences, nheads, headdim, dstate,
            dtype=torch.float32, device="npu"
        )

        states = chunk_state_varlen(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            cu_seqlens=inputs["cu_seqlens"],
            chunk_states=inputs["chunk_states"],
            initial_states=initial_states,
        )

        assert states.shape == (num_sequences, nheads, headdim, dstate)

        print(
            f"[PASS] chunk_state_varlen with initial_states: "
            f"num_sequences={num_sequences}, states.shape={states.shape}"
        )

    def test_single_sequence(self):
        """Test with a single sequence."""
        total_seqlen = 128
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        num_sequences = 1

        inputs = self.create_varlen_inputs(
            total_seqlen=total_seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
            num_sequences=num_sequences,
        )

        states = chunk_state_varlen(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            cu_seqlens=inputs["cu_seqlens"],
            chunk_states=inputs["chunk_states"],
            initial_states=None,
        )

        assert states.shape == (1, nheads, headdim, dstate)

        print(
            f"[PASS] chunk_state_varlen single sequence: "
            f"total_seqlen={total_seqlen}, states.shape={states.shape}"
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_different_dtypes(self, dtype: torch.dtype):
        """Test with different data types."""
        total_seqlen = 256
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        num_sequences = 4

        if dtype == torch.bfloat16 and not torch.npu.is_bf16_supported():
            pytest.skip("BF16 not supported on this GPU")

        inputs = self.create_varlen_inputs(
            total_seqlen=total_seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
            num_sequences=num_sequences,
            dtype=dtype,
        )

        states = chunk_state_varlen(
            B=inputs["B"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            cu_seqlens=inputs["cu_seqlens"],
            chunk_states=inputs["chunk_states"],
            initial_states=None,
        )

        assert states.shape == (num_sequences, nheads, headdim, dstate)

        print(
            f"[PASS] chunk_state_varlen dtype test: dtype={dtype}, "
            f"states.shape={states.shape}"
        )


def run_all_tests():
    """Run all tests manually without pytest."""
    print("=" * 80)
    print("Running ssd_chunk_state tests")
    print("=" * 80)

    cumsum_test = TestChunkCumsumFwd()
    state_test = TestChunkStateFwd()
    varlen_test = TestChunkStateVarlen()

    print("\n" + "=" * 40)
    print("Testing _chunk_cumsum_fwd")
    print("=" * 40)

    print("\n[1] Basic multi-batch tests")
    for batch_size in [1, 2, 4, 8]:
        for seqlen in [64, 128, 256, 512]:
            cumsum_test.test_basic_multi_batch(batch_size, seqlen)

    print("\n[2] Different nheads tests")
    for nheads in [4, 8, 16, 32]:
        cumsum_test.test_different_nheads(nheads)

    print("\n[3] Different chunk_sizes tests")
    for chunk_size in [16, 32, 64, 128]:
        cumsum_test.test_different_chunk_sizes(chunk_size)

    print("\n[4] Tests with dt_bias")
    for batch_size in [1, 2, 4]:
        cumsum_test.test_with_dt_bias(batch_size)

    print("\n[5] Tests with dt_softplus")
    for batch_size in [1, 2]:
        cumsum_test.test_with_dt_softplus(batch_size)

    print("\n[6] Tests with dt_limit")
    for batch_size in [1, 2]:
        cumsum_test.test_with_dt_limit(batch_size)

    print("\n[7] Tests with all options")
    for batch_size in [1, 2]:
        cumsum_test.test_with_all_options(batch_size)

    print("\n[8] Tests with different dtypes")
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        try:
            cumsum_test.test_different_dtypes(dtype)
        except pytest.skip.Exception as e:
            print(f"[SKIP] dtype={dtype}: {e}")

    print("\n" + "=" * 40)
    print("Testing _chunk_state_fwd")
    print("=" * 40)

    print("\n[1] Basic multi-batch tests")
    for batch_size in [1, 2, 4]:
        for seqlen in [64, 128, 256]:
            state_test.test_basic_multi_batch(batch_size, seqlen)

    print("\n[2] Different headdim tests")
    for headdim in [32, 64, 128]:
        state_test.test_different_headdim(headdim)

    print("\n[3] Different dstate tests")
    for dstate in [8, 16, 32, 64, 128]:
        state_test.test_different_dstate(dstate)

    print("\n[4] Different ngroups tests")
    for ngroups in [1, 2, 4, 8]:
        state_test.test_different_ngroups(ngroups)

    print("\n[5] Different chunk_sizes tests")
    for chunk_size in [16, 32, 64, 128]:
        state_test.test_different_chunk_sizes(chunk_size)

    print("\n[6] Tests with seq_idx")
    for batch_size in [1, 2]:
        state_test.test_with_seq_idx(batch_size)

    print("\n[7] Test with preallocated states")
    state_test.test_with_preallocated_states()

    print("\n[8] Test states_in_fp32=False")
    state_test.test_states_in_fp32_false()

    print("\n[9] Tests with different dtypes")
    for dtype in [torch.float16, torch.bfloat16]:
        try:
            state_test.test_different_dtypes(dtype)
        except pytest.skip.Exception as e:
            print(f"[SKIP] dtype={dtype}: {e}")

    print("\n" + "=" * 40)
    print("Testing chunk_state_varlen")
    print("=" * 40)

    print("\n[1] Basic varlen test")
    varlen_test.test_basic_varlen()

    print("\n[2] Different num_sequences tests")
    for num_sequences in [1, 2, 4, 8]:
        varlen_test.test_different_num_sequences(num_sequences)

    print("\n[3] Different headdim tests")
    for headdim in [32, 64, 128]:
        varlen_test.test_different_headdim(headdim)

    print("\n[4] Different dstate tests")
    for dstate in [8, 16, 32, 64]:
        varlen_test.test_different_dstate(dstate)

    print("\n[5] Different chunk_sizes tests")
    for chunk_size in [32, 64, 128]:
        varlen_test.test_different_chunk_sizes(chunk_size)

    print("\n[6] Test with initial_states")
    varlen_test.test_with_initial_states()

    print("\n[7] Test single sequence")
    varlen_test.test_single_sequence()

    print("\n[8] Tests with different dtypes")
    for dtype in [torch.float16, torch.bfloat16]:
        try:
            varlen_test.test_different_dtypes(dtype)
        except pytest.skip.Exception as e:
            print(f"[SKIP] dtype={dtype}: {e}")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()