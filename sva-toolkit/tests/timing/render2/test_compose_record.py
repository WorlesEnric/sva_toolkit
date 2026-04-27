from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import random
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import BBox, NativeSvgRenderer, build_timing_scene, sample_native_render_spec
from sva_toolkit.timing.render2.compose import compose_record
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "td"


def test_compose_record_end_to_end_document_native() -> None:
    from PIL import Image

    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    scene = build_timing_scene(visual, semantic_document=document)
    sampled = sample_native_render_spec(random.Random(61), profile="document-native")
    spec = replace(sampled, extras={**dict(sampled.extras), "output_max_width": "1024", "output_max_height": "768"})
    result = NativeSvgRenderer().render(scene, spec)

    record = compose_record(scene, spec, result, rng=random.Random(83))
    decoded = Image.open(BytesIO(record.image_bytes))
    decoded.load()

    assert record.image_format in {"png", "jpeg", "webp"}
    assert decoded.format in {"PNG", "JPEG", "WEBP"}
    assert record.degradation_chain
    assert record.image_width >= 200
    assert record.image_height >= 100
    assert _mapped_text_region_has_ink(decoded, result.visibility.rendered_text, record.page_metadata, "req")


def _mapped_text_region_has_ink(image, rendered_text, page_metadata, text: str) -> bool:  # type: ignore[no-untyped-def]
    target = next(item for item in rendered_text if item.text == text)
    transform = page_metadata["layout_transform"]
    bbox = BBox(
        target.bbox.x * transform["scale_x"] + transform["offset_x"],
        target.bbox.y * transform["scale_y"] + transform["offset_y"],
        target.bbox.width * transform["scale_x"],
        target.bbox.height * transform["scale_y"],
    )
    padding = 8
    left = max(0, int(bbox.x - padding))
    top = max(0, int(bbox.y - padding))
    right = min(image.width, int(bbox.x + bbox.width + padding))
    bottom = min(image.height, int(bbox.y + bbox.height + padding))
    region = image.crop((left, top, right, bottom)).convert("L")
    return region.getextrema()[0] < 245
