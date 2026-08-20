"""
ingest_pdfs.py — Batch PDF Ingestion into Qdrant

Scans all .pdf files in data/uploads/ and indexes any that are not
already present in the Qdrant vector store.

Deduplication: checks Qdrant for points with matching 'filename' payload.
Skips files already indexed. Run repeatedly without re-indexing duplicates.

Usage:
    python ingest_pdfs.py
    python ingest_pdfs.py --force    # re-index everything (wipe and re-upload)
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path

# ── Make sure project root is on the path ──────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(APP_DIR, ".env"))

from services.extractor import extract_text
from services.chunker import chunk_text
from services.embeddings import generate_document_embeddings
from services.vectordb import store_embeddings, client, COLLECTION_NAME

UPLOAD_FOLDER = os.path.join(APP_DIR, "data", "uploads")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_indexed_filenames() -> set:
    """Return the set of 'filename' payloads already in Qdrant."""
    indexed = set()
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=500,
            offset=offset,
            with_payload=["filename"],
            with_vectors=False,
        )
        for point in result:
            fn = point.payload.get("filename", "")
            if fn:
                indexed.add(fn)
        if next_offset is None:
            break
        offset = next_offset
    return indexed


def _canonical_name(path: str) -> str:
    """
    Strip the UUID prefix from filenames like:
      '073faf86006a4bacae38cd84adab334e_R2025 accadmeic regulation.pdf'
    → 'R2025 accadmeic regulation.pdf'
    """
    base = os.path.basename(path)
    if len(base) > 33 and base[32] == "_":
        return base[33:]
    return base


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch ingest PDFs into Qdrant")
    parser.add_argument("--force", action="store_true", help="Re-index all files (ignore existing)")
    parser.add_argument("--folder", default=UPLOAD_FOLDER, help="Path to uploads folder")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"[Ingest] ERROR: uploads folder not found at {folder}")
        sys.exit(1)

    # Collect all PDF files
    pdf_paths = sorted(Path(folder).glob("*.pdf"))
    print(f"\n[Ingest] Found {len(pdf_paths)} PDF files in {folder}")

    if not pdf_paths:
        print("[Ingest] Nothing to index. Exiting.")
        return

    # ── Deduplicate by canonical name ─────────────────────────────────────────
    # Multiple UUIDs may wrap the same underlying PDF (same canonical name).
    # We keep only ONE representative path per canonical name.
    seen_canonical: dict[str, Path] = {}
    for pdf_path in pdf_paths:
        canon = _canonical_name(str(pdf_path))
        if canon not in seen_canonical:
            seen_canonical[canon] = pdf_path
        else:
            print(f"  [Ingest] Skipping duplicate: {pdf_path.name}  (keeping {seen_canonical[canon].name})")

    unique_pdfs = list(seen_canonical.items())  # [(canonical_name, path), ...]
    print(f"[Ingest] Unique canonical PDFs: {len(unique_pdfs)}")

    # ── Check what's already in Qdrant ────────────────────────────────────────
    if args.force:
        already_indexed = set()
        print("[Ingest] --force flag set: re-indexing all files.")
    else:
        print("[Ingest] Checking Qdrant for already-indexed documents...")
        already_indexed = _get_indexed_filenames()
        print(f"[Ingest] {len(already_indexed)} filenames already in Qdrant.")

    # ── Process each unique PDF ───────────────────────────────────────────────
    total_indexed = 0
    total_skipped = 0
    failed = []

    for canonical_name, pdf_path in unique_pdfs:
        print(f"\n{'='*60}")
        print(f"[Ingest] Processing: {canonical_name}")
        print(f"  Path : {pdf_path}")

        if canonical_name in already_indexed and not args.force:
            print(f"  ✓ Already indexed — skipping.")
            total_skipped += 1
            continue

        try:
            # 1. Extract text page by page
            pages = extract_text(str(pdf_path))
            if not pages:
                print(f"  ⚠ No text extracted — skipping (possibly scanned/image PDF).")
                total_skipped += 1
                continue
            print(f"  Pages extracted : {len(pages)}")

            # 2. Chunk text
            chunks = chunk_text(pages)
            print(f"  Chunks created  : {len(chunks)}")

            if not chunks:
                print(f"  ⚠ No chunks — skipping.")
                total_skipped += 1
                continue

            # 3. Generate embeddings (Jina AI)
            print(f"  Generating embeddings via Jina AI...")
            chunk_texts = [c["text"] for c in chunks]
            embeddings = generate_document_embeddings(chunk_texts)
            print(f"  Embeddings done : {len(embeddings)}")

            # 4. Store in Qdrant
            stored = store_embeddings(
                chunks=chunks,
                embeddings=embeddings,
                filename=canonical_name,   # Use canonical name, not UUID-prefixed
            )
            print(f"  ✅ Stored {stored} chunks into Qdrant as '{canonical_name}'")
            total_indexed += 1

        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed.append((canonical_name, str(e)))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Ingest] DONE")
    print(f"  Indexed  : {total_indexed}")
    print(f"  Skipped  : {total_skipped}")
    print(f"  Failed   : {len(failed)}")
    if failed:
        print("\n  Failed files:")
        for name, err in failed:
            print(f"    - {name}: {err}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
