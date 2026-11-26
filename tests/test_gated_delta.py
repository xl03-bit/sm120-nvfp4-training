"""Synthetic numerical checks for the public gated delta rule."""
import math
import unittest

import torch

from precision_lora import gated_delta_rule


def inputs(length, key_width, value_width, dtype):
    rng = torch.Generator().manual_seed(801)
    batch, heads = 2, 2

    def vectors(width):
        # Transposing [batch, heads, time, width] gives a strided input.
        x = torch.randn(batch, heads, length, width, generator=rng)
        return x.to(device="cuda", dtype=dtype).transpose(1, 2)

    query = vectors(key_width) / math.sqrt(key_width)
    key = vectors(key_width) / math.sqrt(key_width)
    value = vectors(value_width)
    g = -torch.rand(batch, length, heads, generator=rng).cuda()
    beta = torch.rand(batch, length, heads, generator=rng).to(device="cuda", dtype=dtype)
    initial = torch.randn(batch, heads, key_width, value_width, generator=rng).cuda()
    return query, key, value, g, beta, initial


def recurrent_reference(query, key, value, g, beta, initial, normalize):
    """Direct time-step updates with a double-precision accumulator."""
    if normalize:
        # Normalization rounds in the input dtype before the state computation.
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    batch, length, heads, width = query.shape
    state = (query.new_zeros((batch, heads, width, value.shape[-1]), dtype=torch.float64)
             if initial is None else initial.double())
    output = []
    for t in range(length):
        k = key[:, t].double().unsqueeze(-1)
        state = state * g[:, t].double().exp()[..., None, None]
        prediction = (k * state).sum(-2)
        update = beta[:, t].double().unsqueeze(-1) * (value[:, t].double() - prediction)
        state = state + k * update.unsqueeze(-2)
        q = query[:, t].double().unsqueeze(-1) / math.sqrt(width)
        output.append((q * state).sum(-2))
    return torch.stack(output, dim=1).to(query.dtype), state.float()


class GatedDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("native gated delta rule requires an SM120 GPU")

    def test_recurrence_and_readonly_inputs(self):
        for length, key_width, value_width, chunk, with_state in (
            (1, 1, 3, 1, False),
            (11, 5, 7, 4, True),
            (65, 128, 128, 64, False),
        ):
            with self.subTest(length=length):
                q, k, v, g, beta, initial = inputs(length, key_width, value_width, torch.float32)
                state = initial if with_state else None
                tensors = (q, k, v, g, beta, initial)
                snapshots = [x.clone() for x in tensors]
                expected, expected_state = recurrent_reference(q, k, v, g, beta, state, False)
                for return_state in (False, True):
                    out, last = gated_delta_rule(q, k, v, g, beta, chunk_size=chunk,
                                                 initial_state=state, output_final_state=return_state)
                    torch.testing.assert_close(out, expected, rtol=3e-5, atol=2e-6)
                    if return_state:
                        torch.testing.assert_close(last, expected_state, rtol=3e-5, atol=2e-6)
                    else:
                        self.assertIsNone(last)
                for x, saved in zip(tensors, snapshots):
                    torch.testing.assert_close(x, saved, rtol=0, atol=0)

    def test_low_precision_normalization_and_state(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                q, k, v, g, beta, initial = inputs(7, 5, 3, dtype)
                # Include a zero-norm key. The epsilon must keep its update finite.
                k[:, 2].zero_()
                initial = initial.to(dtype)
                expected, expected_state = recurrent_reference(q, k, v, g, beta, initial, True)
                out, last = gated_delta_rule(q, k, v, g, beta, chunk_size=3, initial_state=initial,
                                             output_final_state=True, use_qk_l2norm_in_kernel=True)
                torch.testing.assert_close(out, expected, rtol=torch.finfo(dtype).eps, atol=2e-6)
                torch.testing.assert_close(last, expected_state, rtol=3e-5, atol=2e-6)

    def test_empty_sequence(self):
        q, k, v, g, beta, initial = inputs(0, 5, 3, torch.bfloat16)
        for state in (None, initial.bfloat16()):
            for return_state in (False, True):
                out, last = gated_delta_rule(q, k, v, g, beta, initial_state=state,
                                             output_final_state=return_state)
                torch.testing.assert_close(out, v)
                if return_state:
                    expected = torch.zeros_like(initial) if state is None else state.float()
                    torch.testing.assert_close(last, expected, rtol=0, atol=0)
                else:
                    self.assertIsNone(last)

    def test_tail_matches_explicit_zero_padding(self):
        q, k, v, gate, beta, initial = inputs(75, 128, 31, torch.bfloat16)
        data = (q, k, v, gate, beta)
        for minimum in (32, 64):
            with self.subTest(minimum=minimum):
                pad = minimum - 11
                extended = [torch.cat((x, x.new_zeros((x.shape[0], pad, *x.shape[2:]))), dim=1)
                            for x in data]
                options = dict(initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True)
                # The default uses the full GEMM dimensions on the short tail.
                short, state = gated_delta_rule(*data, min_tail_size=None if minimum == 64 else minimum,
                                                **options)
                long, extended_state = gated_delta_rule(*extended, min_tail_size=1, **options)
                self.assertTrue(torch.equal(short.view(torch.int16), long[:, :75].view(torch.int16)))
                self.assertTrue(torch.equal(state.view(torch.int32), extended_state.view(torch.int32)))

    def test_inference_guard(self):
        for index in (0, 3, 5):
            with self.subTest(grad_input=index), torch.enable_grad():
                tensors = inputs(2, 3, 5, torch.float32)
                tensors[index].requires_grad_()
                with self.assertRaisesRegex(RuntimeError, "inference|grad"):
                    gated_delta_rule(*tensors[:5], initial_state=tensors[5])
                with torch.no_grad():
                    out, _ = gated_delta_rule(*tensors[:5], initial_state=tensors[5])
                self.assertFalse(out.requires_grad)


if __name__ == "__main__":
    unittest.main()
