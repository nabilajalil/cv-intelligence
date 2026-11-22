
"""Stage 3C: Parent Document Retriever for CV Intelligence."""

from collections import defaultdict

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever

from scripts.search_public_cvs import load_public_cvs


def build_retriever():
    """Index small chunks and link them to full parent CVs."""

    load_dotenv()

    records = load_public_cvs()

    documents = [
        Document(
            page_content=record["text"],
            metadata={
                "candidate_id": record["filename"],
            },
        )
        for record in records
    ]

    # Small chunks are embedded and searched.
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    # In-memory vector store for this experiment.
    vectorstore = Chroma(
        collection_name="parent_cv_demo",
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small"
        ),
    )

    # Stores the complete CV documents.
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        id_key="parent_id",
        search_kwargs={"k": 10},
    )

    # Index the children and store their parent CVs.
    retriever.add_documents(
        documents,
        ids=[
            document.metadata["candidate_id"]
            for document in documents
        ],
    )

    return retriever, vectorstore, len(documents)


def main():
    query = "Find candidates with Python and Docker"

    retriever, vectorstore, count = build_retriever()

    print("=== STAGE 3C: PARENT DOCUMENT RETRIEVER ===")
    print("Parent CVs:", count)
    print("Indexed child chunks:", vectorstore._collection.count())
    print("Query:", query)

    # Step 1: Inspect matching child chunks.
    child_matches = vectorstore.similarity_search(
        query,
        k=5,
    )

    print("\n=== MATCHING CHILD CHUNKS ===")

    for index, child in enumerate(child_matches, 1):
        print(f"\nChild {index}")
        print("Parent ID:", child.metadata.get("parent_id"))
        print("Text:", child.page_content[:250])

    # Step 2: Retrieve the complete parent documents.
    parent_documents = retriever.invoke(query)

    print("\n=== RETRIEVED PARENT CVS ===")
    print("Unique parents:", len(parent_documents))

    for index, parent in enumerate(parent_documents, 1):
        print(f"\nParent {index}")
        print(
            "Candidate ID:",
            parent.metadata.get("candidate_id"),
        )
        print("CV length:", len(parent.page_content))
        print("CV preview:", parent.page_content[:350])


if __name__ == "__main__":
    main()
