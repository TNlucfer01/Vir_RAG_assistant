from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

import os
from config import QDRANT_URL, QDRANT_API_KEY
from services.section_parser import detect_section

COLLECTION_NAME = "ptlee_docs" #testing

QDRANT_TIMEOUT = 120  # seconds — large PDFs need more time to upsert


def _get_qdrant_client():
    if QDRANT_URL and QDRANT_API_KEY:
        try:
            c = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                check_compatibility=False,
                timeout=QDRANT_TIMEOUT,
            )
            c.get_collections()
            return c
        except Exception as e:
            print(f"Warning: Cloud Qdrant connection failed ({e}). Falling back to local storage './data/qdrant_db'")
    os.makedirs("data/qdrant_db", exist_ok=True)
    return QdrantClient(path="data/qdrant_db")

client = _get_qdrant_client()

# --------------------------------------------------
# Create Collection
# --------------------------------------------------

try:
    collections = client.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE,
            ),
        )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="filename",
        field_schema=PayloadSchemaType.KEYWORD,
    )
except Exception as e:
    print(f"Qdrant collection setup note: {e}")


# --------------------------------------------------
# Store Embeddings
# --------------------------------------------------

def store_embeddings(chunks, embeddings, filename):

    print("Chunks received by store_embeddings:", len(chunks))
    print("Embeddings received by store_embeddings:", len(embeddings))

    points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        section = detect_section(chunk["text"])

        points.append(
            PointStruct( # what is this ?
                id=abs(hash(f"{filename}_{i}")),
                vector=list(embedding),
                payload={ # meta data 
                    "filename": filename,
                    "document": chunk["text"],
                    "page": chunk["page"],
                    "section": section,
                    "chunk_id": i,
                },
            )
        )

    print("\n========== STORING ==========\n")

    for point in points[:5]: # why only 5 we are not storing anythign anywhere at all  
        print(f"ID       : {point.id}")
        print(f"Page     : {point.payload['page']}")
        print(f"Section  : {point.payload['section']}")
        print(f"Chunk ID : {point.payload['chunk_id']}")
        print(f"Text     : {point.payload['document'][:120]}")
        print()

    print(f"Total Chunks Stored : {len(points)}")

    print("\n=============================\n")

    BATCH_SIZE = 50
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,
        )
        print(f"  Upserted batch {i//BATCH_SIZE + 1}/{(len(points)-1)//BATCH_SIZE + 1} ({len(batch)} points)")

    return len(points)



# --------------------------------------------------
# Search Embeddings
# --------------------------------------------------

def search_embeddings(query_embedding, filename=None, top_k=10):
    """
    Search the vector DB for relevant chunks.

    Args:
        query_embedding: The embedded query vector.
        filename:        If provided, restricts search to that document.
                         If None, searches across ALL stored documents.
        top_k:           Number of results to return.
    """

    # Build filter only when a specific document is requested
    query_filter = None
    if filename:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename),
                )
            ]
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=list(query_embedding),
        limit=top_k,
        query_filter=query_filter,
    )

    print("\n========== RETRIEVED ==========\n")

    for i, point in enumerate(response.points, start=1):
        print(f"Result {i}")
        print("Score   :", point.score)
        print("File    :", point.payload.get("filename"))
        print("Page    :", point.payload.get("page"))
        print("Section :", point.payload.get("section"))
        print("Chunk   :", point.payload["document"][:250])
        print("-" * 60)

    documents = [
        point.payload["document"]
        for point in response.points
    ]

    pages = [
        point.payload.get("page")
        for point in response.points
    ]

    chunk_ids = [
        point.payload.get("chunk_id")
        for point in response.points
    ]

    filenames = [
        point.payload.get("filename")
        for point in response.points
    ]

    return {
        "documents": [documents],
        "pages": pages,
        "chunk_ids": chunk_ids,
        "filenames": filenames,
    }


# --------------------------------------------------
# Get Document Preview
# --------------------------------------------------

def get_document_preview(filename, limit=5):

    response = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    points = response[0]

    points.sort(
        key=lambda point: point.payload.get("chunk_id", 0)
    )

    preview = "\n\n".join(
        point.payload.get("document", "")
        for point in points
    )

    return preview
