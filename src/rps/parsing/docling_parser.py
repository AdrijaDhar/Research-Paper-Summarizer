"""Parse a PDF into a structured DoclingDocument using the Granite-Docling VLM
pipeline, accelerated via MLX on Apple Silicon and Transformers elsewhere.

Source pattern verified against docling's own example:
https://github.com/docling-project/docling/blob/main/docs/examples/minimal_vlm_pipeline.py
"""

import json
import platform
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import (
    MlxVlmEngineOptions,
    TransformersVlmEngineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def build_converter() -> DocumentConverter:
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    engine_options = MlxVlmEngineOptions() if is_apple_silicon else TransformersVlmEngineOptions()
    vlm_options = VlmConvertOptions.from_preset("granite_docling", engine_options=engine_options)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=VlmPipelineOptions(vlm_options=vlm_options),
            ),
        }
    )


def parse_pdf(pdf_path: Path, out_dir: Path, converter: DocumentConverter | None = None) -> dict:
    """Convert `pdf_path` and write markdown + full structured JSON to `out_dir`.

    Returns the structured document as a dict (page/bbox-tagged text, table,
    and figure elements — the input the KG builder in Phase 2 consumes).

    Pass a preloaded `converter` (from `build_converter()`) when processing
    many papers in one run — building it loads the Granite-Docling VLM, which
    is wasteful to repeat per paper.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    converter = converter or build_converter()
    doc = converter.convert(source=str(pdf_path)).document

    doc_dict = doc.export_to_dict()
    (out_dir / f"{pdf_path.stem}.json").write_text(json.dumps(doc_dict, indent=2))
    (out_dir / f"{pdf_path.stem}.md").write_text(doc.export_to_markdown())

    return doc_dict
