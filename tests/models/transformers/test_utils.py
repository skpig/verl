# Utils for testing only. No public APIs exposed.
import torch
from typing import Optional, Any, Union, Callable

__all__ = []


def _compute_entropy_loss(logits, eos_mask):
    """Reference: categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    entropy = _entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = _masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def _entropy_from_logits(logits: torch.Tensor):
    """Reference: entropy loss from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def _masked_mean(values, mask, axis=None):
    """Reference: Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def compare_tensors(expected_2d, actual_2d, verbose=False):
    expected = expected_2d.reshape(-1)
    actual = actual_2d.reshape(-1)
    diff = actual - expected
    abs_diff = diff.abs()
    expected_ref = expected.clone()
    abs_max = abs_diff.max()
    abs_max_idx = abs_diff.argmax()
    rel_diff = abs_diff / expected_ref
    rel_diff_2d = rel_diff.reshape(expected_2d.shape)
    abs_diff_2d = abs_diff.reshape(expected_2d.shape)
    if verbose:
        for i in range(expected_2d.shape[0]):
            for j in range(expected_2d.shape[1]):
                if rel_diff_2d[i, j] > 1e-2 and abs_diff_2d[i, j] > 1e-5:
                    print(f'expected[{i}, {j}] = {expected_2d[i, j]} actual[{i}, {j}] = {actual_2d[i, j]}')

    rel_max = rel_diff.max()
    rel_max_idx = rel_diff.argmax()
    rel_max_idx_cpu = rel_max_idx.detach().cpu().numpy()
    rel_max_indices = rel_max_idx_cpu // expected_2d.shape[1], rel_max_idx_cpu % expected_2d.shape[1]
    return {
        'abs_max': abs_max,
        'expected[abs_idx]': expected[abs_max_idx],
        'actual[abs_idx]': actual[abs_max_idx],
        'expected[rel_idx]': expected[rel_max_idx],
        'actual[rel_idx]': actual[rel_max_idx],
        'rel_max_idx': rel_max_indices,
        'rel_max': rel_max
    }


def prepare_data():
    from flash_attn.bert_padding import index_first_axis, rearrange
    # sample tensor available at hdfs://haruna/home/byte_data_seed/lf_lq/user/haibin.lin/rl/ppo_actor_rmpad_v3.pt
    rmpad_data = torch.load('ppo_actor_rmpad_v3.pt', map_location='cuda:0')

    response_length = rmpad_data['response_length']
    attention_mask = rmpad_data['attention_mask']
    input_ids_rmpad = rmpad_data['input_ids']
    position_ids_rmpad = rmpad_data['position_ids']
    indices = rmpad_data['indices']

    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

    # hidden_states: (nnz, hidden_size) after rmpad
    full_response_mask = attention_mask.clone()
    full_response_mask[:, :-response_length] = 0  # set the prompt part to zero
    full_response_mask_rmpad = index_first_axis(rearrange(full_response_mask.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                indices).squeeze(-1)  # (total_nnz,)

    return input_ids_rmpad, input_ids_rmpad_rolled, full_response_mask_rmpad, position_ids_rmpad


def ref_loss_fn(output, input_ids_rmpad_rolled, full_response_mask_rmpad):
    # input_ids_rmpad_rolled: (total_nnz,)
    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
    # logits_rmpad.div_(temperature)
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    full_log_probs_rmpad = -cross_entropy_loss(logits_rmpad, input_ids_rmpad_rolled,
                                               inplace_backward=False)[0]  # (total_nnz,)

    # compute entropy loss
    # logits_rmpad: (total_nnz, vocab_size)
    # full_response_mask_rmpad: (total_nnz,)
    entropy_loss = _compute_entropy_loss(logits_rmpad, full_response_mask_rmpad)
    # entropy_loss: scaler
    return entropy_loss, full_log_probs_rmpad


# Adapted from torch/testing/_comparison.py @ 2.4.1
def check_close(
    actual: Any,
    expected: Any,
    *,
    allow_subclasses: bool = True,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
    msg: Optional[Union[str, Callable[[str], str]]] = None,
    percent_threshold: float = 0,
):
    r"""Asserts that ``actual`` and ``expected`` are close.

    If ``actual`` and ``expected`` are strided, non-quantized, real-valued, and finite, they are considered close if

    .. math::

        \lvert \text{actual} - \text{expected} \rvert \le \texttt{atol} + \texttt{rtol} \cdot \lvert \text{expected} \rvert

    Non-finite values (``-inf`` and ``inf``) are only considered close if and only if they are equal. ``NaN``'s are
    only considered equal to each other if ``equal_nan`` is ``True``.

    In addition, they are only considered close if they have the same

    - :attr:`~torch.Tensor.device` (if ``check_device`` is ``True``),
    - ``dtype`` (if ``check_dtype`` is ``True``),
    - ``layout`` (if ``check_layout`` is ``True``), and
    - stride (if ``check_stride`` is ``True``).

    If either ``actual`` or ``expected`` is a meta tensor, only the attribute checks will be performed.

    If ``actual`` and ``expected`` are sparse (either having COO, CSR, CSC, BSR, or BSC layout), their strided members are
    checked individually. Indices, namely ``indices`` for COO, ``crow_indices`` and ``col_indices`` for CSR and BSR,
    or ``ccol_indices``  and ``row_indices`` for CSC and BSC layouts, respectively,
    are always checked for equality whereas the values are checked for closeness according to the definition above.

    If ``actual`` and ``expected`` are quantized, they are considered close if they have the same
    :meth:`~torch.Tensor.qscheme` and the result of :meth:`~torch.Tensor.dequantize` is close according to the
    definition above.

    ``actual`` and ``expected`` can be :class:`~torch.Tensor`'s or any tensor-or-scalar-likes from which
    :class:`torch.Tensor`'s can be constructed with :func:`torch.as_tensor`. Except for Python scalars the input types
    have to be directly related. In addition, ``actual`` and ``expected`` can be :class:`~collections.abc.Sequence`'s
    or :class:`~collections.abc.Mapping`'s in which case they are considered close if their structure matches and all
    their elements are considered close according to the above definition.

    .. note::

        Python scalars are an exception to the type relation requirement, because their :func:`type`, i.e.
        :class:`int`, :class:`float`, and :class:`complex`, is equivalent to the ``dtype`` of a tensor-like. Thus,
        Python scalars of different types can be checked, but require ``check_dtype=False``.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        allow_subclasses (bool): If ``True`` (default) and except for Python scalars, inputs of directly related types
            are allowed. Otherwise type equality is required.
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must also be specified. If omitted, default
            values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified ``rtol`` must also be specified. If omitted, default
            values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        equal_nan (Union[bool, str]): If ``True``, two ``NaN`` values will be considered equal.
        check_device (bool): If ``True`` (default), asserts that corresponding tensors are on the same
            :attr:`~torch.Tensor.device`. If this check is disabled, tensors on different
            :attr:`~torch.Tensor.device`'s are moved to the CPU before being compared.
        check_dtype (bool): If ``True`` (default), asserts that corresponding tensors have the same ``dtype``. If this
            check is disabled, tensors with different ``dtype``'s are promoted  to a common ``dtype`` (according to
            :func:`torch.promote_types`) before being compared.
        check_layout (bool): If ``True`` (default), asserts that corresponding tensors have the same ``layout``. If this
            check is disabled, tensors with different ``layout``'s are converted to strided tensors before being
            compared.
        check_stride (bool): If ``True`` and corresponding tensors are strided, asserts that they have the same stride.
        msg (Optional[Union[str, Callable[[str], str]]]): Optional error message to use in case a failure occurs during
            the comparison. Can also passed as callable in which case it will be called with the generated message and
            should return the new message.

    Raises:
        ValueError: If no :class:`torch.Tensor` can be constructed from an input.
        ValueError: If only ``rtol`` or ``atol`` is specified.
        AssertionError: If corresponding inputs are not Python scalars and are not directly related.
        AssertionError: If ``allow_subclasses`` is ``False``, but corresponding inputs are not Python scalars and have
            different types.
        AssertionError: If the inputs are :class:`~collections.abc.Sequence`'s, but their length does not match.
        AssertionError: If the inputs are :class:`~collections.abc.Mapping`'s, but their set of keys do not match.
        AssertionError: If corresponding tensors do not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If ``check_layout`` is ``True``, but corresponding tensors do not have the same
            :attr:`~torch.Tensor.layout`.
        AssertionError: If only one of corresponding tensors is quantized.
        AssertionError: If corresponding tensors are quantized, but have different :meth:`~torch.Tensor.qscheme`'s.
        AssertionError: If ``check_device`` is ``True``, but corresponding tensors are not on the same
            :attr:`~torch.Tensor.device`.
        AssertionError: If ``check_dtype`` is ``True``, but corresponding tensors do not have the same ``dtype``.
        AssertionError: If ``check_stride`` is ``True``, but corresponding strided tensors do not have the same stride.
        AssertionError: If the values of corresponding tensors are not close according to the definition above.

    The following table displays the default ``rtol`` and ``atol`` for different ``dtype``'s. In case of mismatching
    ``dtype``'s, the maximum of both tolerances is used.

    +---------------------------+------------+----------+
    | ``dtype``                 | ``rtol``   | ``atol`` |
    +===========================+============+==========+
    | :attr:`~torch.float16`    | ``1e-3``   | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.bfloat16`   | ``1.6e-2`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.float32`    | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.float64`    | ``1e-7``   | ``1e-7`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex32`  | ``1e-3``   | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex64`  | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex128` | ``1e-7``   | ``1e-7`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.quint8`     | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.quint2x4`   | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.quint4x2`   | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.qint8`      | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.qint32`     | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | other                     | ``0.0``    | ``0.0``  |
    +---------------------------+------------+----------+

    .. note::

        :func:`~torch.testing.assert_close` is highly configurable with strict default settings. Users are encouraged
        to :func:`~functools.partial` it to fit their use case. For example, if an equality check is needed, one might
        define an ``assert_equal`` that uses zero tolerances for every ``dtype`` by default:

        >>> import functools
        >>> assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
        >>> assert_equal(1e-9, 1e-10)
        Traceback (most recent call last):
        ...
        AssertionError: Scalars are not equal!
        <BLANKLINE>
        Expected 1e-10 but got 1e-09.
        Absolute difference: 9.000000000000001e-10
        Relative difference: 9.0

    Examples:
        >>> # tensor to tensor comparison
        >>> expected = torch.tensor([1e0, 1e-1, 1e-2])
        >>> actual = torch.acos(torch.cos(expected))
        >>> torch.testing.assert_close(actual, expected)

        >>> # scalar to scalar comparison
        >>> import math
        >>> expected = math.sqrt(2.0)
        >>> actual = 2.0 / math.sqrt(2.0)
        >>> torch.testing.assert_close(actual, expected)

        >>> # numpy array to numpy array comparison
        >>> import numpy as np
        >>> expected = np.array([1e0, 1e-1, 1e-2])
        >>> actual = np.arccos(np.cos(expected))
        >>> torch.testing.assert_close(actual, expected)

        >>> # sequence to sequence comparison
        >>> import numpy as np
        >>> # The types of the sequences do not have to match. They only have to have the same
        >>> # length and their elements have to match.
        >>> expected = [torch.tensor([1.0]), 2.0, np.array(3.0)]
        >>> actual = tuple(expected)
        >>> torch.testing.assert_close(actual, expected)

        >>> # mapping to mapping comparison
        >>> from collections import OrderedDict
        >>> import numpy as np
        >>> foo = torch.tensor(1.0)
        >>> bar = 2.0
        >>> baz = np.array(3.0)
        >>> # The types and a possible ordering of mappings do not have to match. They only
        >>> # have to have the same set of keys and their elements have to match.
        >>> expected = OrderedDict([("foo", foo), ("bar", bar), ("baz", baz)])
        >>> actual = {"baz": baz, "bar": bar, "foo": foo}
        >>> torch.testing.assert_close(actual, expected)

        >>> expected = torch.tensor([1.0, 2.0, 3.0])
        >>> actual = expected.clone()
        >>> # By default, directly related instances can be compared
        >>> torch.testing.assert_close(torch.nn.Parameter(actual), expected)
        >>> # This check can be made more strict with allow_subclasses=False
        >>> torch.testing.assert_close(
        ...     torch.nn.Parameter(actual), expected, allow_subclasses=False
        ... )
        Traceback (most recent call last):
        ...
        TypeError: No comparison pair was able to handle inputs of type
        <class 'torch.nn.parameter.Parameter'> and <class 'torch.Tensor'>.
        >>> # If the inputs are not directly related, they are never considered close
        >>> torch.testing.assert_close(actual.numpy(), expected)
        Traceback (most recent call last):
        ...
        TypeError: No comparison pair was able to handle inputs of type <class 'numpy.ndarray'>
        and <class 'torch.Tensor'>.
        >>> # Exceptions to these rules are Python scalars. They can be checked regardless of
        >>> # their type if check_dtype=False.
        >>> torch.testing.assert_close(1.0, 1, check_dtype=False)

        >>> # NaN != NaN by default.
        >>> expected = torch.tensor(float("Nan"))
        >>> actual = expected.clone()
        >>> torch.testing.assert_close(actual, expected)
        Traceback (most recent call last):
        ...
        AssertionError: Scalars are not close!
        <BLANKLINE>
        Expected nan but got nan.
        Absolute difference: nan (up to 1e-05 allowed)
        Relative difference: nan (up to 1.3e-06 allowed)
        >>> torch.testing.assert_close(actual, expected, equal_nan=True)

        >>> expected = torch.tensor([1.0, 2.0, 3.0])
        >>> actual = torch.tensor([1.0, 4.0, 5.0])
        >>> # The default error message can be overwritten.
        >>> torch.testing.assert_close(actual, expected, msg="Argh, the tensors are not close!")
        Traceback (most recent call last):
        ...
        AssertionError: Argh, the tensors are not close!
        >>> # If msg is a callable, it can be used to augment the generated message with
        >>> # extra information
        >>> torch.testing.assert_close(
        ...     actual, expected, msg=lambda msg: f"Header\n\n{msg}\n\nFooter"
        ... )
        Traceback (most recent call last):
        ...
        AssertionError: Header
        <BLANKLINE>
        Tensor-likes are not close!
        <BLANKLINE>
        Mismatched elements: 2 / 3 (66.7%)
        Greatest absolute difference: 2.0 at index (1,) (up to 1e-05 allowed)
        Greatest relative difference: 1.0 at index (1,) (up to 1.3e-06 allowed)
        <BLANKLINE>
        Footer
    """
    # Hide this function from `pytest`'s traceback
    __tracebackhide__ = True
    from torch.testing._comparison import not_close_error_metas
    from torch.testing._comparison import NonePair, BooleanPair, NumberPair, TensorLikePair
    error_metas = not_close_error_metas(
        actual,
        expected,
        pair_types=(
            NonePair,
            BooleanPair,
            NumberPair,
            TensorLikePair,
        ),
        allow_subclasses=allow_subclasses,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride,
        msg=msg,
    )
    if error_metas:
        if 'Mismatched elements' in error_metas[0].msg:
            percentage = float(extract_percentage(error_metas[0].msg))
            if percentage > percent_threshold * 100:
                raise error_metas[0].to_error(msg)


def extract_percentage(input_string):
    """
    Extracts the percentage value from the given string.

    Args:
        input_string (str): The input string in the template format.

    Returns:
        float: The extracted percentage as a float, or None if no percentage is found.
    """
    import re
    match = re.search(r'\((\d+(\.\d+)?)%\)', input_string)
    if match:
        return float(match.group(1))
    return None
