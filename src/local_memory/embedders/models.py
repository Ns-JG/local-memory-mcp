from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """An embedding model and everything needed to use it correctly.

    ``query_prefix`` is the trap: BGE and E5 families expect an instruction on
    the query side, and E5 additionally expects a different one on the document
    side. Applying the query prefix to documents, or omitting it from queries,
    degrades retrieval silently. They live here so they cannot drift apart from
    the model.
    """

    name: str
    dim: int
    query_prefix: str = ""
    document_prefix: str = ""
    threads: int = 3
    note: str = ""

    def document_text(self, text: str) -> str:
        return f"{self.document_prefix}{text}" if self.document_prefix else text
