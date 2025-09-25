import numpy as np
from PIL import Image

from alpha_seed.utils.dataset import data_decoder
from alpha_seed.utils.dataset.navit_transform import get_resized_hw_for_Navit


def denorm_box(points, height, width):
    new_points = []
    for p in points:
        new_points.append((round(p[0] * width), round(p[1] * height)))
    return new_points


def process_image_for_tiktok(frames: list[Image.Image], mask_boxes):
    mask_boxes = mask_boxes[:len(frames)]
    frames = [np.array(f) for f in frames]
    # assert len(mask_boxes) == len(frames)
    height, width = frames[0].shape[:2]

    new_frames = []
    for boxes, frame in zip(mask_boxes, frames):
        left, top, right, bottom = 0, 0, width, height
        for box in boxes:
            pts = np.array(denorm_box(box, height, width), np.int32)
            upper_bound = max([p[1] for p in pts]) + 30
            if bottom > upper_bound:
                bottom = upper_bound
            frame[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0]] = 0

        new_frames.append(Image.fromarray(frame[top:bottom, left:right]))
    return new_frames


def sample_video(
    video_reader,
    frame_indices: list | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    n_frames: int | None = None,
    time_indices: float | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    sampling_fps: int | None = None,
    min_n_frames: int | None = None,
    max_n_frames: int | None = None,
) -> list:
    if time_indices is not None:
        frame_indices = [round(float(i) * video_reader.fps) for i in time_indices]

    if start_time is not None and end_time is not None:
        start_frame = round(start_time * video_reader.fps)
        end_frame = round(end_time * video_reader.fps)

    if frame_indices is None:
        start_frame = 0 if start_frame is None else round(start_frame)
        end_frame = video_reader.max_frame_id if end_frame is None else round(end_frame)
        if end_frame > video_reader.max_frame_id:
            end_frame = video_reader.max_frame_id

        if sampling_fps is not None:
            frame_interval = max(1, round(video_reader.fps / sampling_fps))
            n_frames = len(list(range(start_frame, end_frame + 1, frame_interval)))
            if min_n_frames is not None and n_frames < min_n_frames:
                n_frames = min_n_frames
            if max_n_frames is not None and n_frames > max_n_frames:
                n_frames = max_n_frames
            n_frames = min(n_frames, video_reader.length, end_frame - start_frame + 1)

        if n_frames == 1:
            frame_indices = [start_frame]
        else:
            frame_indices = np.linspace(start_frame, end_frame, n_frames).round().astype(int).tolist()

    return frame_indices


class GeneralVideoTransform:

    def __init__(self, img_size: int, video_sampling_strategy: dict, **kwargs):
        self.img_size = img_size
        self.min_pixels = kwargs.get('min_pixels', 4 * 28 * 28)
        self.max_pixels = kwargs.get('max_pixels', 5120 * 28 * 28)
        self.mean = kwargs.get('mean', [0.48145466, 0.4578275, 0.40821073])
        self.std = kwargs.get('std', [0.26862954, 0.26130258, 0.27577711])

        _default_video_sampling_strategy = {
            'sampling_fps': 1,
            'n_frames': None,
            'min_n_frames': 16,
            'max_n_frames': None,
            'max_video_length': 24 * 1024,
            'max_pixels_choices': [640 * 28 * 28, 512 * 28 * 28, 384 * 28 * 28, 256 * 28 * 28, 160 * 28 * 28],
            'use_timestamp': True,
        }
        if video_sampling_strategy is None:
            self.video_sampling_strategy = _default_video_sampling_strategy
        else:
            self.video_sampling_strategy = video_sampling_strategy

    def __call__(self, entry, video_reader):
        n_frames = self.video_sampling_strategy.get('n_frames', None)
        sampling_fps = self.video_sampling_strategy.get('sampling_fps', None)
        assert n_frames is None or sampling_fps is None, self.video_sampling_strategy
        if n_frames is not None:
            min_n_frames = entry.get('min_n_frames', self.video_sampling_strategy.get('min_n_frames', None))
            max_n_frames = entry.get('max_n_frames', self.video_sampling_strategy.get('max_n_frames', None))
            if min_n_frames is not None:
                n_frames = max(n_frames, min_n_frames)
            if max_n_frames is not None:
                n_frames = min(n_frames, max_n_frames)
        else:
            min_sampling_fps = entry.get('min_sampling_fps', self.video_sampling_strategy.get('min_sampling_fps', None))
            max_sampling_fps = entry.get('max_sampling_fps', self.video_sampling_strategy.get('max_sampling_fps', None))
            if min_sampling_fps is not None:
                sampling_fps = max(sampling_fps, min_sampling_fps)
            if max_sampling_fps is not None:
                sampling_fps = min(sampling_fps, max_sampling_fps)

        frame_indices = sample_video(
            video_reader,
            frame_indices=entry.get('frame_indices', None),
            start_frame=entry.get('start_frame', None),
            end_frame=entry.get('end_frame', None),
            n_frames=n_frames,
            time_indices=entry.get('time_indices', None),
            start_time=entry.get('start_time', None),
            end_time=entry.get('end_time', None),
            sampling_fps=sampling_fps,
            min_n_frames=entry.get('min_n_frames', self.video_sampling_strategy.get('min_n_frames', None)),
            max_n_frames=entry.get('max_n_frames', self.video_sampling_strategy.get('max_n_frames', None)),
        )

        max_video_length = self.video_sampling_strategy["max_video_length"]
        max_pixels_choices = self.video_sampling_strategy["max_pixels_choices"]
        max_pixels = self.max_pixels
        for round_idx, max_pixels in enumerate(max_pixels_choices):
            is_last_round = round_idx == len(max_pixels_choices) - 1
            if len(frame_indices) * max_pixels / 28 / 28 > max_video_length:
                if is_last_round:
                    max_frame_num = int(max_video_length / max_pixels * 28 * 28)
                    select_ids = np.linspace(0, len(frame_indices) - 1, max_frame_num).round().astype(int).tolist()
                    frame_indices = [frame_indices[select_id] for select_id in select_ids]
                else:
                    continue
            else:
                break

        frames = video_reader.sample(frame_indices)
        if entry.get('mask_boxes', None) is not None:
            frames = process_image_for_tiktok(frames, entry['mask_boxes'])

        use_timestamp = entry.get('use_timestamp', self.video_sampling_strategy.get('use_timestamp', False))
        image_info_list = []
        for frame_idx, image in zip(frame_indices, frames):
            if isinstance(image, Image.Image):
                pass
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif hasattr(image, "asnumpy"):
                image = Image.fromarray(image.asnumpy())
            else:
                image = data_decoder.bytesdecoder(image)

            width, height = image.size
            resized_height, resized_width = get_resized_hw_for_Navit(
                height,
                width,
                img_size=self.img_size,
                min_pixels=self.min_pixels,
                max_pixels=max_pixels,
            )
            image = image.resize((resized_width, resized_height))
            image_info = {'image': image}
            if use_timestamp:
                image_info['timestamp'] = frame_idx / video_reader.fps
            image_info_list.append(image_info)

        return image_info_list


def simple_video_transform(
    video_reader,
    video_sampling_strategy: dict,
    img_size: int = 448,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 5120 * 28 * 28,
) -> list[dict]:
    n_frames = video_sampling_strategy.get('n_frames', None)
    sampling_fps = video_sampling_strategy.get('sampling_fps', None)
    assert n_frames is None or sampling_fps is None, video_sampling_strategy
    if n_frames is not None:
        min_n_frames = video_sampling_strategy.get('min_n_frames', None)
        max_n_frames = video_sampling_strategy.get('max_n_frames', None)
        if min_n_frames is not None:
            n_frames = max(n_frames, min_n_frames)
        if max_n_frames is not None:
            n_frames = min(n_frames, max_n_frames)
    else:
        min_sampling_fps = video_sampling_strategy.get('min_sampling_fps', None)
        max_sampling_fps = video_sampling_strategy.get('max_sampling_fps', None)
        if min_sampling_fps is not None:
            sampling_fps = max(sampling_fps, min_sampling_fps)
        if max_sampling_fps is not None:
            sampling_fps = min(sampling_fps, max_sampling_fps)

    frame_indices = sample_video(
        video_reader,
        frame_indices=None,
        start_frame=None,
        end_frame=None,
        n_frames=n_frames,
        time_indices=None,
        start_time=None,
        end_time=None,
        sampling_fps=sampling_fps,
        min_n_frames=video_sampling_strategy.get('min_n_frames', None),
        max_n_frames=video_sampling_strategy.get('max_n_frames', None),
    )

    max_video_length = video_sampling_strategy["max_video_length"]
    max_pixels_choices = video_sampling_strategy["max_pixels_choices"]
    for round_idx, max_pixels in enumerate(max_pixels_choices):
        is_last_round = round_idx == len(max_pixels_choices) - 1
        if len(frame_indices) * max_pixels / 28 / 28 > max_video_length:
            if is_last_round:
                max_frame_num = int(max_video_length / max_pixels * 28 * 28)
                select_ids = np.linspace(0, len(frame_indices) - 1, max_frame_num).round().astype(int).tolist()
                frame_indices = [frame_indices[select_id] for select_id in select_ids]
            else:
                continue
        else:
            break

    frames = video_reader.sample(frame_indices)

    use_timestamp = video_sampling_strategy.get('use_timestamp', False)
    image_info_list = []
    for frame_idx, image in zip(frame_indices, frames):
        if isinstance(image, Image.Image):
            pass
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif hasattr(image, "asnumpy"):
            image = Image.fromarray(image.asnumpy())
        else:
            image = data_decoder.bytesdecoder(image)

        width, height = image.size
        resized_height, resized_width = get_resized_hw_for_Navit(
            height,
            width,
            img_size=img_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = image.resize((resized_width, resized_height))
        image_info = {'image': image}
        if use_timestamp:
            image_info['timestamp'] = frame_idx / video_reader.fps
        image_info_list.append(image_info)

    return image_info_list
