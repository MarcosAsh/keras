from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes import (
    iou as iou_lib,
)


@keras_export("keras.utils.bounding_boxes.non_max_suppression")
def non_max_suppression(
    boxes,
    scores,
    bounding_box_format,
    max_output_size,
    iou_threshold=0.5,
    score_threshold=float("-inf"),
):
    """Greedy non-maximum suppression over a set of bounding boxes.

    Visits boxes in descending score order and keeps each one unless it
    overlaps an already-kept box with an intersection-over-union (IoU) greater
    than `iou_threshold`. Overlap is computed with
    `keras.utils.bounding_boxes.compute_iou`, so any
    `bounding_box_format` it accepts works here.

    The result is padded to a static length so the op runs under compilation
    (`jax.jit`, `tf.function`). Slice the output with the returned `num_valid`
    count to drop the padding.

    Args:
        boxes: Tensor of shape `(num_boxes, 4)` holding the boxes in
            `bounding_box_format`.
        scores: Tensor of shape `(num_boxes,)` with a score per box.
        bounding_box_format: String, the format of the boxes, for example
            `"xyxy"` or `"yxyx"`. See
            `keras.utils.bounding_boxes.convert_format` for the supported
            formats.
        max_output_size: Python integer. The maximum number of boxes to keep
            and the length of the returned `selected_indices`.
        iou_threshold: Float in `[0, 1]`. Boxes with IoU greater than this
            value relative to an already-kept box are suppressed. Defaults to
            `0.5`.
        score_threshold: Float. Boxes with a score at or below this value are
            never kept. Defaults to `float("-inf")` (no score filtering).

    Returns:
        A tuple `(selected_indices, num_valid)`:
            `selected_indices`: int32 tensor of shape `(max_output_size,)`
                indexing into `boxes`. The first `num_valid` entries are the
                kept boxes in descending score order. The rest are padding set
                to `0`.
            `num_valid`: int32 scalar, the number of valid entries in
                `selected_indices`.

    Example:

    >>> boxes = np.array(
    ...     [[0, 0, 1, 1], [0, 0, 1, 1.1], [10, 10, 11, 11]],
    ...     dtype="float32",
    ... )
    >>> scores = np.array([0.9, 0.8, 0.7], dtype="float32")
    >>> indices, num_valid = keras.utils.bounding_boxes.non_max_suppression(
    ...     boxes, scores, "xyxy", max_output_size=3, iou_threshold=0.5
    ... )
    >>> int(num_valid)
    2
    >>> np.asarray(indices)[:int(num_valid)]
    array([0, 2], dtype=int32)
    """
    boxes = ops.convert_to_tensor(boxes)
    scores = ops.convert_to_tensor(scores)
    max_output_size = int(max_output_size)

    num_boxes = ops.shape(scores)[0]

    # Visit boxes in descending score order, so the lowest-index still-active
    # box is always the highest-scoring remaining one.
    order = ops.argsort(ops.negative(scores))
    boxes = ops.take(boxes, order, axis=0)
    scores = ops.take(scores, order, axis=0)

    # Full pairwise IoU computed once, reusing the module's `compute_iou`.
    iou = iou_lib.compute_iou(boxes, boxes, bounding_box_format)

    positions = ops.arange(num_boxes, dtype="int32")
    sentinel = ops.cast(num_boxes, "int32")
    active = scores > score_threshold
    selected = ops.zeros((max_output_size,), dtype="int32")
    num_valid = ops.convert_to_tensor(0, dtype="int32")
    k = ops.convert_to_tensor(0, dtype="int32")

    def cond(k, active, selected, num_valid):
        return ops.logical_and(k < max_output_size, ops.any(active))

    def body(k, active, selected, num_valid):
        # The first still-active position is the highest-scoring remaining box.
        best = ops.min(ops.where(active, positions, sentinel))
        chosen = ops.cast(ops.take(order, best), "int32")
        selected = ops.slice_update(
            selected, ops.expand_dims(k, 0), ops.expand_dims(chosen, 0)
        )
        # Suppress the chosen box and anything overlapping it past the
        # threshold, using the precomputed IoU row.
        iou_row = ops.take(iou, best, axis=0)
        active = ops.logical_and(active, iou_row <= iou_threshold)
        active = ops.logical_and(active, ops.not_equal(positions, best))
        return (k + 1, active, selected, num_valid + 1)

    _, _, selected, num_valid = ops.while_loop(
        cond,
        body,
        (k, active, selected, num_valid),
        maximum_iterations=max_output_size,
    )
    return selected, num_valid
