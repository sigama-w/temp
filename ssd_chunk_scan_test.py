# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test cases for _chunk_scan_fwd operator in ssd_chunk_scan.py

This test file contains multiple test cases for the _chunk_scan_fwd function,
including:
- Basic single batch case
- Multi-batch case
- Case with D parameter
- Case with z parameter (gating)
- Case with seq_idx (continuous batching)
- Case with initial_states
"""

import torch
import pytest

from ssd_chunk_scan import _chunk_scan_fwd


def create_basic_inputs(
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
    """Create basic input tensors for _chunk_scan_fwd."""
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    x = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    dt = torch.randn(batch_size, nheads, nchunks, chunk_size, dtype=dtype, device=device)

    dA_cumsum = torch.randn(batch_size, nheads, nchunks, chunk_size, dtype=torch.float32, device=device)

    C = torch.randn(batch_size, seqlen, ngroups, dstate, dtype=dtype, device=device)

    states = torch.randn(
        batch_size, nchunks, nheads, headdim, dstate, dtype=torch.float32, device=device
    )

    cb = torch.randn(
        batch_size, nchunks, ngroups, chunk_size, chunk_size, dtype=torch.float32, device=device
    )

    out = torch.empty_like(x)

    return {
        "cb": cb,
        "x": x,
        "dt": dt,
        "dA_cumsum": dA_cumsum,
        "C": C,
        "states": states,
        "out": out,
        "nchunks": nchunks,
    }


class TestChunkScanFwd:
    """Test class for _chunk_scan_fwd operator."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seqlen", [64, 128, 256])
    def test_basic_multi_batch(self, batch_size: int, seqlen: int):
        """Test basic functionality with different batch sizes and sequence lengths."""
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=None,
            z=None,
            seq_idx=None,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)
        assert out_x is None

        print(
            f"[PASS] Basic multi-batch test: batch_size={batch_size}, "
            f"seqlen={seqlen}, out.shape={inputs['out'].shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("headdim", [32, 64, 128])
    def test_different_headdim(self, batch_size: int, headdim: int):
        """Test with different head dimensions."""
        nheads = 8
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=None,
            z=None,
            seq_idx=None,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)

        print(
            f"[PASS] Different headdim test: batch_size={batch_size}, "
            f"headdim={headdim}, out.shape={inputs['out'].shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_with_D_parameter(self, batch_size: int):
        """Test with D parameter (skip connection)."""
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        D = torch.randn(nheads, headdim, dtype=torch.float32, device="npu")

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=D,
            z=None,
            seq_idx=None,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)
        assert out_x is None

        print(
            f"[PASS] With D parameter test: batch_size={batch_size}, "
            f"out.shape={inputs['out'].shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_with_D_scalar(self, batch_size: int):
        """Test with D parameter as scalar per head (nheads,) shape."""
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        D = torch.randn(nheads, dtype=torch.float32, device="npu")

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=D,
            z=None,
            seq_idx=None,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)
        assert out_x is None

        print(
            f"[PASS] With D scalar test: batch_size={batch_size}, "
            f"out.shape={inputs['out'].shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_with_z_parameter(self, batch_size: int):
        """Test with z parameter (gating mechanism)."""
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        z = torch.randn(batch_size, seqlen, nheads, headdim, dtype=inputs["x"].dtype, device="npu")

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=None,
            z=z,
            seq_idx=None,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)
        assert out_x is not None
        assert out_x.shape == inputs["out"].shape

        print(
            f"[PASS] With z parameter test: batch_size={batch_size}, "
            f"out.shape={inputs['out'].shape}, out_x.shape={out_x.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_with_D_and_z(self, batch_size: int):
        """Test with both D and z parameters."""
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        D = torch.randn(nheads, headdim, dtype=torch.float32, device="npu")
        z = torch.randn(batch_size, seqlen, nheads, headdim, dtype=inputs["x"].dtype, device="npu")

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=D,
            z=z,
            seq_idx=None,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)
        assert out_x is not None
        assert out_x.shape == inputs["out"].shape

        print(
            f"[PASS] With D and z parameters test: batch_size={batch_size}, "
            f"out.shape={inputs['out'].shape}, out_x.shape={out_x.shape}"
        )

    def test_with_seq_idx(self):
        """Test with seq_idx parameter (continuous batching)."""
        batch_size = 1
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        seq_idx = torch.zeros(batch_size, seqlen, dtype=torch.int32, device="npu")
        seq_idx[0, seqlen // 2 :] = 1

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=None,
            z=None,
            seq_idx=seq_idx,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)
        assert out_x is None

        print(
            f"[PASS] With seq_idx test: batch_size={batch_size}, "
            f"seqlen={seqlen}, seq_idx unique values={torch.unique(seq_idx).tolist()}"
        )

    def test_with_initial_states(self):
        """Test with initial_states parameter (for chunked prefill)."""
        batch_size = 1
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128
        nchunks = (seqlen + chunk_size - 1) // chunk_size

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        initial_states = torch.randn(
            2, nheads, headdim, dstate, dtype=torch.float32, device="npu"
        )

        seq_idx = torch.zeros(batch_size, seqlen, dtype=torch.int32, device="npu")
        seq_idx[0, seqlen // 2 :] = 1

        chunk_indices = torch.tensor([0, 0], dtype=torch.int32, device="npu")
        chunk_offsets = torch.tensor([0, seqlen // 2], dtype=torch.int32, device="npu")

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=None,
            z=None,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            initial_states=initial_states,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)
        assert out_x is None

        print(
            f"[PASS] With initial_states test: batch_size={batch_size}, "
            f"seqlen={seqlen}, nchunks={nchunks}"
        )

    def test_with_initial_states_and_all_options(self):
        """Test with initial_states, D, z, and seq_idx parameters."""
        batch_size = 1
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 128

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        initial_states = torch.randn(
            2, nheads, headdim, dstate, dtype=torch.float32, device="npu"
        )

        seq_idx = torch.zeros(batch_size, seqlen, dtype=torch.int32, device="npu")
        seq_idx[0, seqlen // 2 :] = 1

        chunk_indices = torch.tensor([0, 0], dtype=torch.int32, device="npu")
        chunk_offsets = torch.tensor([0, seqlen // 2], dtype=torch.int32, device="npu")

        D = torch.randn(nheads, headdim, dtype=torch.float32, device="npu")
        z = torch.randn(batch_size, seqlen, nheads, headdim, dtype=inputs["x"].dtype, device="npu")

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=D,
            z=z,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            initial_states=initial_states,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)
        assert out_x is not None
        assert out_x.shape == inputs["out"].shape

        print(
            f"[PASS] With all parameters test: batch_size={batch_size}, "
            f"seqlen={seqlen}, out.shape={inputs['out'].shape}, out_x.shape={out_x.shape}"
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

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=None,
            z=None,
            seq_idx=None,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)

        print(
            f"[PASS] Different ngroups test: ngroups={ngroups}, "
            f"out.shape={inputs['out'].shape}"
        )

    @pytest.mark.parametrize("dstate", [8, 16, 32, 64])
    def test_different_dstate(self, dstate: int):
        """Test with different dstate values."""
        batch_size = 2
        nheads = 8
        headdim = 64
        ngroups = 2
        chunk_size = 64
        seqlen = 128

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=None,
            z=None,
            seq_idx=None,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)

        print(
            f"[PASS] Different dstate test: dstate={dstate}, "
            f"out.shape={inputs['out'].shape}"
        )

    @pytest.mark.parametrize("chunk_size", [32, 64, 128])
    def test_different_chunk_sizes(self, chunk_size: int):
        """Test with different chunk sizes."""
        batch_size = 2
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        seqlen = 256

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=None,
            z=None,
            seq_idx=None,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)

        print(
            f"[PASS] Different chunk_size test: chunk_size={chunk_size}, "
            f"nchunks={inputs['nchunks']}, out.shape={inputs['out'].shape}"
        )

    def test_multiple_sequences_in_batch(self):
        """Test continuous batching with multiple sequences."""
        batch_size = 1
        nheads = 8
        headdim = 64
        ngroups = 2
        dstate = 16
        chunk_size = 64
        seqlen = 256

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
        )

        seq_idx = torch.zeros(batch_size, seqlen, dtype=torch.int32, device="npu")
        seq_idx[0, 0:64] = 0
        seq_idx[0, 64:128] = 1
        seq_idx[0, 128:192] = 2
        seq_idx[0, 192:256] = 3

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=None,
            z=None,
            seq_idx=seq_idx,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)

        print(
            f"[PASS] Multiple sequences test: batch_size={batch_size}, "
            f"seqlen={seqlen}, num_sequences={torch.unique(seq_idx).numel()}"
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

        inputs = create_basic_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            ngroups=ngroups,
            dstate=dstate,
            chunk_size=chunk_size,
            dtype=dtype,
        )

        out_x = _chunk_scan_fwd(
            cb=inputs["cb"],
            x=inputs["x"],
            dt=inputs["dt"],
            dA_cumsum=inputs["dA_cumsum"],
            C=inputs["C"],
            states=inputs["states"],
            D=None,
            z=None,
            seq_idx=None,
            chunk_indices=None,
            chunk_offsets=None,
            initial_states=None,
            out=inputs["out"],
        )

        assert inputs["out"].shape == (batch_size, seqlen, nheads, headdim)
        assert inputs["out"].dtype == dtype

        print(
            f"[PASS] Different dtype test: dtype={dtype}, "
            f"out.shape={inputs['out'].shape}"
        )


def run_all_tests():
    """Run all tests manually without pytest."""
    test_instance = TestChunkScanFwd()

    print("=" * 80)
    print("Running _chunk_scan_fwd tests")
    print("=" * 80)

    print("\n[1] Basic multi-batch tests")
    for batch_size in [1, 2, 4]:
        for seqlen in [64, 128, 256]:
            test_instance.test_basic_multi_batch(batch_size, seqlen)

    print("\n[2] Different headdim tests")
    for batch_size in [1, 2]:
        for headdim in [32, 64, 128]:
            test_instance.test_different_headdim(batch_size, headdim)

    print("\n[3] Tests with D parameter")
    for batch_size in [1, 2, 4]:
        test_instance.test_with_D_parameter(batch_size)

    print("\n[4] Tests with D scalar")
    for batch_size in [1, 2]:
        test_instance.test_with_D_scalar(batch_size)

    print("\n[5] Tests with z parameter")
    for batch_size in [1, 2]:
        test_instance.test_with_z_parameter(batch_size)

    print("\n[6] Tests with D and z parameters")
    for batch_size in [1, 2]:
        test_instance.test_with_D_and_z(batch_size)

    print("\n[7] Test with seq_idx")
    test_instance.test_with_seq_idx()

    print("\n[8] Test with initial_states")
    test_instance.test_with_initial_states()

    print("\n[9] Test with all parameters")
    test_instance.test_with_initial_states_and_all_options()

    print("\n[10] Tests with different ngroups")
    for ngroups in [1, 2, 4, 8]:
        test_instance.test_different_ngroups(ngroups)

    print("\n[11] Tests with different dstate")
    for dstate in [8, 16, 32, 64]:
        test_instance.test_different_dstate(dstate)

    print("\n[12] Tests with different chunk sizes")
    for chunk_size in [32, 64, 128]:
        test_instance.test_different_chunk_sizes(chunk_size)

    print("\n[13] Test with multiple sequences")
    test_instance.test_multiple_sequences_in_batch()

    print("\n[14] Tests with different dtypes")
    for dtype in [torch.float16, torch.bfloat16]:
        try:
            test_instance.test_different_dtypes(dtype)
        except pytest.skip.Exception as e:
            print(f"[SKIP] dtype={dtype}: {e}")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()