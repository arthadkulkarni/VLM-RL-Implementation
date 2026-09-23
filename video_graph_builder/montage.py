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


def build_montage(frames, labels=("first", "mid", "last")):
    """Horizontally concatenate sampled frames (left = earliest) into one
    strip image with a thin label row, so a single-image VLM prompt can be
    used to describe motion across a segment.
    """
    if not frames:
        raise ValueError("build_montage requires at least one frame")

    labels = list(labels)[:len(frames)]
    while len(labels) < len(frames):
        labels.append("")

    target_height = min(frame.height for frame in frames)
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
