from PIL import Image, ImageDraw

GUTTER = 6
LABEL_HEIGHT = 18
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)


def sample_frame_indices(start, end, num_frames=3):
    if end <= start:
        return [start] * num_frames
    if num_frames == 1:
        return [(start + end) // 2]
    step = (end - start) / (num_frames - 1)
    return [round(start + i * step) for i in range(num_frames)]


def build_montage(frames, labels=("first", "mid", "last"), max_frame_height=None):
    """Horizontally concatenate sampled frames (left = earliest) into one
    strip image with a thin label row, so a single-image VLM prompt can be
    used to describe motion across a segment. Frames are downscaled to at most
    max_frame_height: the montage is what every caption, same-entity and
    Verifier call sees, so its pixel count sets their image-token cost.
    """
    if not frames:
        raise ValueError("build_montage requires at least one frame")

    labels = list(labels)[:len(frames)]
    while len(labels) < len(frames):
        labels.append("")

    target_height = min(frame.height for frame in frames)
    if max_frame_height:
        target_height = min(target_height, max_frame_height)
    resized = []
    for frame in frames:
        if frame.height != target_height:
            new_width = max(1, round(frame.width * target_height / frame.height))
            frame = frame.resize((new_width, target_height))
        resized.append(frame)

    total_width = sum(frame.width for frame in resized) + GUTTER * (len(resized) - 1)
    montage = Image.new("RGB", (total_width, target_height + LABEL_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(montage)

    x = 0
    for frame, label in zip(resized, labels):
        montage.paste(frame, (x, LABEL_HEIGHT))
        if label:
            draw.text((x + 4, 2), label, fill=TEXT_COLOR)
        x += frame.width + GUTTER

    return montage


# Two stacked full-resolution montages can exceed the caption servers'
# max_model_len in image tokens alone; ~2M pixels is ~2k Qwen3-VL tokens.
MAX_PAIR_PIXELS = 2_000_000


def stack_montages(top, bottom, labels=("Segment A", "Segment B"), max_pixels=MAX_PAIR_PIXELS):
    """Vertically stack two segment montages (each already a left-to-right
    frame strip) into one labelled image, so a single-image VLM prompt can
    compare entities across the two segments.
    """
    return stack_montage_rows([top, bottom], labels, max_pixels=max_pixels)


def stack_montage_rows(montages, labels, max_pixels=MAX_PAIR_PIXELS):
    """Stack montages top to bottom, each under its label, and downscale the
    result to at most max_pixels."""
    width = max(montage.width for montage in montages)
    height = sum(montage.height for montage in montages) + len(montages) * LABEL_HEIGHT + (len(montages) - 1) * GUTTER
    stacked = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(stacked)

    y = 0
    for montage, label in zip(montages, labels):
        draw.text((4, y + 2), label, fill=TEXT_COLOR)
        stacked.paste(montage, (0, y + LABEL_HEIGHT))
        y += LABEL_HEIGHT + montage.height + GUTTER

    scale = (max_pixels / (width * height)) ** 0.5
    if scale < 1:
        stacked = stacked.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    return stacked
