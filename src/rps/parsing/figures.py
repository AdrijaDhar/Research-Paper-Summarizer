"""Extract cropped figure/picture images from a PDF.

Uses Docling's standard (non-VLM) pipeline, not the Granite-Docling VLM
pipeline in `docling_parser.py` — `VlmPipelineOptions` doesn't expose the
`generate_page_images` / `PictureItem.get_image()` cropping path that
`PdfPipelineOptions` does, so this runs Docling a second time with a different
pipeline, purely to get figure crops. Verified against docling's own
`docs/examples/export_figures.py`.
"""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem


def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def extract_figures(pdf_path: Path, out_dir: Path, converter: DocumentConverter | None = None) -> list[dict]:
    """Pass a preloaded `converter` (from `build_converter()`) when processing
    many papers in one run, to avoid rebuilding the pipeline per paper."""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    converter = converter or build_converter()
    doc = converter.convert(str(pdf_path)).document

    figures = []
    for element, _level in doc.iterate_items():
        if not isinstance(element, PictureItem):
            continue
        index = len(figures) + 1
        image_path = figures_dir / f"figure-{index}.png"
        element.get_image(doc).save(image_path, "PNG")
        page_no = element.prov[0].page_no if element.prov else None
        figures.append(
            {"index": index, "page_no": page_no, "image_path": str(image_path), "caption": None}
        )

    return figures
