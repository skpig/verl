import base64
import json
import re
from io import BytesIO
from PIL import Image, ImageDraw
import torch

from transformers import AutoImageProcessor, AutoTokenizer
import asyncio

from alpha_seed.workers.agents.handlers.base_tool import BaseTool
from alpha_seed.utils.dataset.vlm_rl_dataset import decode_bytes_to_rgb_image
from alpha_seed.utils.ckpt.hdfs import download_config_and_tokenizer
import numpy as np
from collections import defaultdict


def convert_image_to_rgb(image: Image) -> Image:
    if image.mode == 'RGBA' or image.info.get('transparency', None) is not None:
        image = image.convert('RGBA')
        white = Image.new(mode='RGB', size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert('RGB')
    return image


def encode_image_as_base64(img: Image, include_media_type: bool, convert_to_rgb: bool = True) -> str:
    if convert_to_rgb:
        img = convert_image_to_rgb(img)

    output_buffer = BytesIO()
    img_type = img.format if img.format else 'PNG'
    img.save(output_buffer, format=img_type)
    base64_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

    if include_media_type:
        base64_image = f'data:image/{img_type.lower()};base64,{base64_image}'

    return base64_image


def POINT(image_bytes: bytes, points: str, draw_line: bool = False) -> dict:
    """
    name = 'POINT'
    description = 'Renders the specified points on an image and returns the result.'
    parameters = {
        'type': 'object',
        'properties': {
            'imgidx': {
                'type': 'integer',
                'description': 'Index of the image to process. The first image in the session has index 0.',
            },
            'points': {
                'type': 'string',
                'description':
                    'Points to render, each as <point>x y</point>, where x and y are integers from 0 to 999. '
                    'The origin (0,0) is the top-left corner, with the x-axis to the right and the y-axis downward. The bottom-right corner is (999, 999). '
                    'Use the 0–999 coordinate system defined here for this argument, even if another coordinate system is shown in the image. '
                    'If multiple points exist, fill in <point>x1 y1</point><point>x2 y2</point>...<point>xN yN</point>. '
                    'If draw_line is true and multiple lines must be drawn, separate points belonging to each line with a newline.'
            },
            'draw_line': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to connect the points with lines. Default is false.',
            }
        },
        'required': ['imgidx', 'points'],
    }
    """
    # Split the 'points' string by newline. Each line can represent a separate set of points for a separate line.
    lines = points.strip().split('\n')
    if not lines:
        raise ValueError('No valid points provided.')

    # Prepare to collect coordinate sets, one set for each line.
    lines_coords = []
    # Regex to verify that each line has only <point>x y</point> sequences.
    pattern_line = r'^(<point>\s*(\d{1,3})\s+(\d{1,3})\s*</point>\s*)+$'
    for line_str in lines:
        # Trim whitespace for safety.
        line_str = line_str.strip()
        if not line_str:
            # If an empty line is encountered, skip it.
            continue
        # Validate the line structure.
        if not re.match(pattern_line, line_str):
            raise ValueError('Invalid points format or invalid point coordinates.')
        # Extract the coordinates from the current line.
        coords = re.findall(r'<point>\s*(\d+)\s+(\d+)\s*</point>', line_str)
        coords_int = [(int(x), int(y)) for x, y in coords]
        if not coords_int:
            raise ValueError('No valid points found on one of the lines.')
        lines_coords.append(coords_int)

    # Open the image and draw the specified points and optional lines.
    with Image.open(BytesIO(image_bytes)) as raw_img:
        img = convert_image_to_rgb(raw_img)
        w, h = img.size
        draw = ImageDraw.Draw(img)

        min_size = min(w, h)
        radius = max(4, int(min_size * 0.02))
        radius = min(radius, 12)

        # Process each line of coordinates separately.
        for coords in lines_coords:
            # Convert (0-999) relative coordinates to absolute pixel coordinates.
            abs_coords = []
            for (x_rel, y_rel) in coords:
                x_pix = int((x_rel / 999) * w)
                y_pix = int((y_rel / 999) * h)
                abs_coords.append((x_pix, y_pix))

            # Draw a small red dot at each point.
            color = 'blue'
            for (x_pix, y_pix) in abs_coords:
                draw.ellipse((x_pix - radius // 2, y_pix - radius // 2, x_pix + radius // 2, y_pix + radius // 2),
                             fill=color,
                             outline=color)

            # If draw_line is true and there's more than one point, connect them.
            if draw_line and len(abs_coords) > 1:
                draw.line(abs_coords, fill=color, width=radius // 2)

        # Save the edited image.
        base64_image = encode_image_as_base64(img, include_media_type=False)

    return dict(text="", image_base64=base64_image)


def GROUNDING(image_bytes: bytes, bbox_str: str, crop: bool = False) -> dict:
    """
    name = 'GROUNDING'
    description = 'If crop is false, draw the bounding boxes on the image and return the edited image. If crop is true, crop the image to the bounding box.'
    parameters = {
        'type': 'object',
        'properties': {
            'imgidx': {
                'type': 'integer',
                'description': 'Index of the image to process. The first image in the session has index 0.',
            },
            'bbox_str': {
                'type': 'string',
                'description':
                    'Coordinates of bounding boxes to draw or crop. Each bounding box is written as <bbox>x1 y1 x2 y2</bbox>, where x and y are integers from 0 to 999. '
                    'The origin (0,0) is the top-left corner, with the x-axis to the right and the y-axis downward. The bottom-right corner is (999, 999). '
                    'Use the 0–999 coordinate system defined here for this argument, even if another coordinate system is shown in the image. '
                    'If multiple bboxes exist, fill in <bbox>x1 y1 x2 y2</bbox><bbox>x3 y3 x4 y4</bbox>...',
            },
            'crop': {
                'type':
                    'boolean',
                'default':
                    False,
                'description':
                    'Whether to crop the image according to bbox_str. Default is false. When true, bbox_str must contain only one bounding box.',
            }
        },
        'required': ['imgidx', 'bbox_str'],
    }
    """
    # Validate bbox_str format
    pattern = r'(<bbox>\s*(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s*</bbox>\s*)+'
    if not re.fullmatch(pattern, bbox_str):
        raise ValueError('Invalid bbox_str format. Ensure it contains valid bounding boxes like '
                         '<bbox>x1 y1 x2 y2</bbox>, where x and y are integers between 0 and 999.')

    # Parse bounding boxes
    bboxes = re.findall(r'<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</bbox>', bbox_str)
    bboxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in bboxes]

    # If cropping, only one bounding box is allowed
    if crop and len(bboxes) != 1:
        raise ValueError('When crop is true, bbox_str must contain exactly one bbox.')

    # Open the image and process
    with Image.open(BytesIO(image_bytes)) as raw_img:
        img = convert_image_to_rgb(raw_img)
        w, h = img.size
        # Convert normalized coords (0-999) to absolute pixel coords
        abs_bboxes = []
        for (x1, y1, x2, y2) in bboxes:
            abs_x1 = int((x1 / 999) * w)
            abs_y1 = int((y1 / 999) * h)
            abs_x2 = int((x2 / 999) * w)
            abs_y2 = int((y2 / 999) * h)
            # Ensure proper ordering (left < right, top < bottom)
            left, right = sorted([abs_x1, abs_x2])
            top, bottom = sorted([abs_y1, abs_y2])
            abs_bboxes.append((left, top, right, bottom))

        if crop:
            # Crop the image to the single bounding box
            (left, top, right, bottom) = abs_bboxes[0]
            img = img.crop((left, top, right, bottom))
        else:
            # Draw bounding boxes
            min_size = min(w, h)
            line_width = max(2, int(min_size * 0.02))
            line_width = min(line_width, 8)
            draw = ImageDraw.Draw(img)
            for (left, top, right, bottom) in abs_bboxes:
                top = max(top - line_width, 0)
                bottom = min(bottom + line_width, h)
                left = max(left - line_width, 0)
                right = min(right + line_width, w)
                draw.rectangle((left, top, right, bottom), outline='red', width=line_width)

        # Save the updated image
        base64_image = encode_image_as_base64(img, include_media_type=False)

    return dict(text="", image_base64=base64_image)


def ROTATE(image_bytes: bytes, degree: int) -> dict:
    """
    name = 'ROTATE'
    description = 'Rotate an image.'
    parameters = {
        'type': 'object',
        'properties': {
            'imgidx': {
                'type': 'integer',
                'description': 'Index of the image to process. The first image in the session has index 0.',
            },
            'degree': {
                'type':
                    'integer',
                'description':
                    'Integer between 1-359, representing the clockwise rotation angle. 90 means clockwise rotation of 90 degrees, 180 means rotation of 180 degrees, 270 means counterclockwise rotation of 90 degrees.',
            },
        },
        'required': ['imgidx', 'degree'],
    }
    """
    # Open and rotate (PIL's rotate() is counterclockwise, so we use -degree for clockwise)
    with Image.open(BytesIO(image_bytes)) as raw_img:
        img = convert_image_to_rgb(raw_img)
        img = img.rotate(-degree, expand=True)
        base64_image = encode_image_as_base64(img, include_media_type=False)
    return dict(text="", image_base64=base64_image)


def ZOOM(image_bytes: bytes, bbox_str: str = "", scale: float = 0.0) -> dict:
    """
    name = 'ZOOM'
    description = 'Zoom an image.'
    parameters = {
        'type': 'object',
        'properties': {
            'imgidx': {
                'type': 'integer',
                'description': 'Index of the image to process. The first image in the session has index 0.',
            },
            'bbox_str': {
                'type': 'string',
                'default': '',
                'description':
                    'The image area to be zoomed, represented as <bbox>x1 y1 x2 y2</bbox>, where x and y are integers between 0-999. '
                    'The origin (0,0) is the top-left corner, with the x-axis to the right and the y-axis downward. The bottom-right corner is (999, 999). '
                    'Use the 0–999 coordinate system defined here for this argument, even if another coordinate system is shown in the image. '
                    'If this argument is an empty string, the whole image is zoomed.',
            },
            'scale': {
                'type':
                    'number',
                'default':
                    0.0,
                'description':
                    'A floating-point number from 0.0 to 2.0 representing the zoom ratio. 1.0 means no change, 0.5 halves the size, 2.0 doubles it. If set to 0.0 with bbox_str not empty, the chosen area is automatically scaled to the full image.'
            },
        },
        'required': ['imgidx'],
    }
    """
    ## avoid too large image
    if (scale <= 0) or (scale > 2.0):
        raise ValueError(f"Scale should be between 0.0 (excluded) and 2.0 (included), but got {scale}.")
    # Validate and parse the bbox if present
    x1_rel = y1_rel = x2_rel = y2_rel = None
    if bbox_str:
        match = re.fullmatch(r'<bbox>\s*(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s*</bbox>', bbox_str)
        if not match:
            raise ValueError(
                'Invalid bbox_str format. Must be <bbox>x1 y1 x2 y2</bbox> with each coordinate between 0 and 999.')
        x1_rel, y1_rel, x2_rel, y2_rel = map(int, match.groups())

    with Image.open(BytesIO(image_bytes)) as raw_img:
        img = convert_image_to_rgb(raw_img)
        w, h = img.size
        has_bbox = all(x is not None for x in (x1_rel, y1_rel, x2_rel, y2_rel))
        # If there is a bounding box, crop and scale only that region
        if has_bbox:
            # Convert (0..999) coords to actual pixel coords
            x1_pix = int((x1_rel / 999) * w)
            y1_pix = int((y1_rel / 999) * h)
            x2_pix = int((x2_rel / 999) * w)
            y2_pix = int((y2_rel / 999) * h)
            # Ensure valid order
            left, right = sorted([x1_pix, x2_pix])
            top, bottom = sorted([y1_pix, y2_pix])
            region_occupy_ratio = (right - left) * (bottom - top) / (w * h)

            # Crop region
            crop_region = img.crop((left, top, right, bottom))
            cw, ch = crop_region.size

            # Decide how to scale
            if scale < 1e-3:
                # Scale the cropped region to fill the entire original image size
                new_size = (w, h)
            elif scale >= 1.0 and region_occupy_ratio >= 0.5:
                raise ValueError(
                    'Zooming in on a very large region is a waste of computation. Please consider zooming in on a smaller region.'
                )
            else:
                # Scale the cropped region by "scale"
                new_cw = int(cw * scale)
                new_ch = int(ch * scale)
                if new_cw < 1 or new_ch < 1:
                    raise ValueError('Invalid scale factor results in zero or negative dimension.')
                new_size = (new_cw, new_ch)
            img = crop_region.resize(new_size, Image.Resampling.LANCZOS)
        else:
            # If bbox_str is empty, zoom the entire image
            if scale < 1e-3:
                # scale=0 with empty bbox => effectively no change
                # we can just copy the original image
                img = img.copy()
            elif scale >= 1.0:
                raise ValueError(
                    'Zooming the entire image is a waste of computation. Please consider zooming in on a specific region.'
                )
            else:
                new_w = int(w * scale)
                new_h = int(h * scale)
                if new_w < 1 or new_h < 1:
                    raise ValueError('Invalid scale factor results in zero or negative dimension.')
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        out_w, out_h = img.size
        if out_w * out_h > 1920 * 1080:
            if has_bbox:
                raise ValueError(
                    'The resulting image is too large. Please consider using a smaller scale factor or a smaller bounding box.'
                )
            else:
                raise ValueError(
                    'The resulting image is too large. Please consider focusing on only a local region of interest.')

        # Save the updated (zoomed) image
        base64_image = encode_image_as_base64(img, include_media_type=False)

    return dict(text="", image_base64=base64_image)


class VisualCotEnv(BaseTool):
    """
    Only support native python function now.
    """
    VISUAL_COT_FUNC = ["POINT", "GROUNDING", "ROTATE", "ZOOM"]

    def __init__(self, **kwargs) -> None:
        tokenizer = kwargs.get('tokenizer', None)
        image_processor = kwargs.get('image_processor', None)
        config = kwargs.get('config')
        assert tokenizer is not None and image_processor is not None, f"tokenizer or image_processor is None"
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self._metrics = defaultdict(list)
        self._is_finished = False
        self.soi_token_id = self.tokenizer.encode(config.data.special_tokens.soi)
        self.eoi_token_id = self.tokenizer.encode(config.data.special_tokens.eoi)

    @property
    def metrics(self) -> dict:
        metrics = {}
        metrics.update({f'avg_{k}': np.mean(v) for k, v in self._metrics.items()})
        metrics.update({f'max_{k}': np.max(v) for k, v in self._metrics.items()})
        return {f"visual_cot_{k}": v for k, v in metrics.items()}

    def action_supported(self, action: str) -> bool:
        return True

    @property
    def finished(self) -> bool:
        return self._is_finished

    async def async_eval(self, expression, globals=None, locals=None):
        return await asyncio.to_thread(eval, expression, globals or {}, locals or {})

    async def step(self, action: str) -> tuple[bool, dict | str]:
        """
        Args:
            action (str): a function string, e.g. 'POINT(image_bytes=..., points=...)'.
        """
        match = re.match(r'^(\w+)\(', action)
        if not match:
            return False, f"Invalid function call string: {action}"
        function_name = match.group(1)
        if function_name not in self.VISUAL_COT_FUNC:
            return False, f"{function_name} is not available now."
        if ('image_bytes' not in action) and (function_name != 'PYTHON'):
            return False, "Missing image input."
        self._metrics[function_name].append(1)

        try:
            raw_visual_cot_output = await self.async_eval(action, globals=globals())
            if "image_base64" in raw_visual_cot_output:
                image_base64 = raw_visual_cot_output["image_base64"]
                image_inputs = self.image_processor(images=decode_bytes_to_rgb_image(base64.b64decode(image_base64)))
                input_ids = [
                    self.tokenizer.bos_token_id
                ] + self.tokenizer.encode("tool name=plugin\n" + raw_visual_cot_output["text"]) + self.soi_token_id + \
                [-100] * image_inputs["num_image_tokens"][0] + self.eoi_token_id + [
                        self.tokenizer.eos_token_id, self.tokenizer.bos_token_id
                    ] + self.tokenizer.encode("assistant\n")
                pixel_values = image_inputs.pop("pixel_values")
                image_grid_hw = image_inputs.pop("image_grid_hw")
            else:
                input_ids = [self.tokenizer.bos_token_id
                            ] + self.tokenizer.encode("tool name=plugin\n" + raw_visual_cot_output["text"]) + [
                                self.tokenizer.eos_token_id, self.tokenizer.bos_token_id
                            ] + self.tokenizer.encode("assistant\n")
                pixel_values = None
                image_grid_hw = None
            output = dict(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_hw=image_grid_hw,
                raw_output=raw_visual_cot_output,
            )
            return True, output
        except Exception as e:
            print(f"[env1] execution error {e}")
            return False, str(e)


def create_from_env_str(env_str: str, **kwargs):
    prefix = "visual_cot@"
    assert env_str.startswith(prefix)
    return VisualCotEnv(**kwargs)
