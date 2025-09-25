import bisect
import io
import pickle
from typing import Any

try:
    import decord

    decord.bridge.set_bridge('native')
except ImportError:
    pass

import numpy as np
from PIL import Image, ImageSequence


def audio_decoder(item: dict[str, Any] | bytes, audio_index: int = 0):
    if isinstance(item, dict):
        audio_data = item['audio_bytes_list']
    else:
        audio_data = item

    if isinstance(audio_data, bytes):
        audio_data = pickle.loads(audio_data)
    else:
        return None

    if not (isinstance(audio_data, np.ndarray) or isinstance(audio_data, list)):
        raise ValueError(f"audio data must be list of bytes, but got data type {type(audio_data)}")

    audio_data = audio_data[audio_index]
    return audio_data


class ImgSeqReader:

    def __init__(self, frame_bytes: bytes, meta_info: dict) -> None:
        self.frame_bytes = frame_bytes
        self._fps = meta_info['fps']
        self._max_frame_id = max(meta_info['frame_indices'])
        self.idxs = sorted(meta_info['frame_indices'])
        self.idx2frame_bytes = {idx: f for idx, f in zip(self.idxs, self.frame_bytes)}

    @property
    def max_frame_id(self) -> int:
        return self._max_frame_id

    @property
    def length(self) -> int:
        return len(self.frame_bytes)

    @property
    def fps(self):
        return self._fps

    def sample(self, frame_indices: list) -> list[Image.Image]:
        frame_indices = [self.find_closest_idx(idx) for idx in frame_indices]
        frames = [Image.open(io.BytesIO(self.idx2frame_bytes[i])).convert('RGB') for i in frame_indices]
        return frames

    def find_closest_idx(self, idx: int) -> int:
        pos = bisect.bisect_left(self.idxs, idx)
        if pos == 0:
            return self.idxs[0]
        if pos == len(self.idxs):
            return self.idxs[-1]

        if abs(self.idxs[pos] - idx) < abs(self.idxs[pos - 1] - idx):
            return self.idxs[pos]
        else:
            return self.idxs[pos - 1]


class GIFReader:

    def __init__(self, video_bytes: bytes, meta_info: dict) -> None:
        gif = Image.open(io.BytesIO(video_bytes))
        self.frames = [frame.convert('RGB') for frame in ImageSequence.Iterator(gif)]

        self._max_frame_id = len(self.frames) - 1
        self._fps = meta_info.get('fps', 1)

    @property
    def max_frame_id(self):
        return self._max_frame_id

    @property
    def length(self):
        return len(self.frames)

    @property
    def fps(self):
        return self._fps

    def sample(self, frame_indices: list[int]) -> list[Image.Image] | list[float]:
        frames = [self.frames[i] for i in frame_indices]
        return frames


class VideoReader:

    def __init__(self, video_bytes: bytes) -> None:
        self.video_buffer = io.BytesIO(video_bytes)

        self.vr = decord.VideoReader(self.video_buffer, num_threads=4, ctx=decord.cpu(0), fault_tol=1)
        self.vr.seek(0)
        self._max_frame_id = len(self.vr) - 1
        self._fps = self.vr.get_avg_fps()

    @property
    def max_frame_id(self) -> int:
        return self._max_frame_id

    @property
    def length(self) -> int:
        return len(self.vr)

    @property
    def fps(self) -> float:
        return self._fps

    def sample(self, frame_indices: list[int]) -> list[Image.Image] | list[float]:
        try:
            frames = self.vr.get_batch(frame_indices).asnumpy()
        except decord._ffi.base.DECORDError:
            self.video_buffer.seek(0)
            vr = decord.VideoReader(self.video_buffer, num_threads=1, ctx=decord.cpu(0), fault_tol=1)
            frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(f).convert('RGB') for f in frames]
        return frames


def video_decoder(item: dict[str, Any], video_index: int = 0, decode_mode: str = 'video'):
    """
    Get video reader from json dictionary.
    """
    ##### currently only suppport video mode.
    assert decode_mode in ['video'], f'unsupported decode mode: {decode_mode}'
    if isinstance(item, dict):
        video_data = item['videos']
    else:
        video_data = item
    if video_data is None:
        return None

    if isinstance(video_data, bytes):
        video_data = pickle.loads(video_data)

    if isinstance(video_data, np.ndarray) or isinstance(video_data, list):
        video_data = video_data[video_index]

    video_meta_info = video_data.get('meta_info', {})
    if decode_mode == "video":
        try:
            video_reader = VideoReader(video_data['video_bytes'])
        except decord._ffi.base.DECORDError:
            video_reader = GIFReader(video_data['video_bytes'], video_meta_info)
    elif decode_mode == "image":
        video_reader = ImgSeqReader(video_data['frame_bytes'], video_meta_info)
    else:
        raise ValueError(f'Unknown decode_mode: {decode_mode}')

    return video_reader
