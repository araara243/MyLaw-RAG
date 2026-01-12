"""
LangChain RAG Pipeline for Malaysian Legal Q&A

This module assembles the complete RAG chain:
1. Query input
2. Hybrid retrieval (semantic + keyword)
3. Context formatting
4. LLM generation with legal prompts
5. Response with citations
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


# Import prompts
import sys
sys.path.insert(0, str(get_project_root() / "src"))

from generation.prompts import (
    LEGAL_SPECIALIST_SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    NO_CONTEXT_PROMPT
)
from retrieval.hybrid_retriever import HybridRetriever


class LegalRAGChain:
    """
    Complete RAG chain for Malaysian Legal Q&A.
    
    Combines hybrid retrieval with LLM generation for
    accurate, citation-grounded legal responses.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-lite",
        temperature: float = 0.1,
        n_results: int = 5,
        retrieval_method: str = "hybrid"
    ):
        """
        Initialize the Legal RAG Chain.
        
        Args:
            model_name: Google Gemini model to use (e.g., gemini-2.0-flash, gemini-1.5-pro).
            temperature: LLM temperature (low for legal accuracy).
            n_results: Number of chunks to retrieve.
            retrieval_method: "hybrid", "semantic", or "keyword".
        """
        self.model_name = model_name
        self.temperature = temperature
        self.n_results = n_results
        self.retrieval_method = retrieval_method
        
        # Initialize components
        self._retriever = None
        self._llm = None
        self._chain = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the retriever and LLM."""
        # Initialize retriever
        self._retriever = HybridRetriever()
        
        # Initialize LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_google_api_key_here":
            logger.warning(
                "GOOGLE_API_KEY not set. LLM generation will be disabled. "
                "Get a free API key at https://aistudio.google.com/ and set it in .env file."
            )
            self._llm = None
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=api_key
            )
        
        # Build the chain
        self._build_chain()
        
        logger.info(f"LegalRAGChain initialized (model: {self.model_name})")
    
    def _build_chain(self):
        """Build the LangChain RAG pipeline."""
        if self._llm is None:
            self._chain = None
            return
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", LEGAL_SPECIALIST_SYSTEM_PROMPT),
            ("human", RAG_PROMPT_TEMPLATE)
        ])
        
        # Build chain with retrieval
        def retrieve_and_format(query: str) -> str:
            results = self._retriever.retrieve(
                query,
                n_results=self.n_results,
                method=self.retrieval_method
            )
            return self._retriever.format_context(results)
        
        self._chain = (
            {
                "context": RunnableLambda(retrieve_and_format),
                "question": RunnablePassthrough()
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )
    
    def retrieve(self, question: str) -> list:
        """
        Retrieve relevant legal chunks without LLM generation.
        
        Args:
            question: The user's legal question.
        
        Returns:
            List of RetrievalResult objects.
        """
        return self._retriever.retrieve(
            question,
            n_results=self.n_results,
            method=self.retrieval_method
        )
    
    def ask(
        self,
        question: str,
        return_sources: bool = True
    ) -> dict:
        """
        Ask a legal question and get an answer with citations.
        
        Args:
            question: The user's legal question.
            return_sources: Whether to include source chunks.
        
        Returns:
            Dictionary with:
                - answer: The generated response
                - sources: List of source chunks (if return_sources=True)
        """
        # Retrieve relevant chunks
        sources = self.retrieve(question)
        
        # Check if we have relevant context
        if not sources:
            return {
                "answer": NO_CONTEXT_PROMPT.format(question=question),
                "sources": []
            }
        
        # Generate answer
        if self._chain is None:
            # LLM not available, return retrieval only
            context = self._retriever.format_context(sources)
            return {
                "answer": (
                    "⚠️ LLM generation is disabled (no API key). "
                    "Here are the relevant legal sections:\n\n" + context
                ),
                "sources": sources if return_sources else []
            }
        
        # Run the chain
        answer = self._chain.invoke(question)
        
        result = {"answer": answer}
        if return_sources:
            result["sources"] = [
                {
                    "chunk_id": s.chunk_id,
                    "act_name": s.act_name,
                    "section_number": s.section_number,
                    "section_title": s.section_title,
                    "score": s.score
                }
                for s in sources
            ]
        
        return result
    
    def ask_stream(self, question: str):
        """
        Ask a question with streaming response.
        
        Yields chunks of the response as they are generated.
        """
        if self._chain is None:
            yield "⚠️ LLM generation is disabled (no API key)."
            return
        
        # Build streaming version of chain
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        streaming_llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            streaming=True,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", LEGAL_SPECIALIST_SYSTEM_PROMPT),
            ("human", RAG_PROMPT_TEMPLATE)
        ])
        
        # Retrieve context
        results = self._retriever.retrieve(
            question,
            n_results=self.n_results,
            method=self.retrieval_method
        )
        context = self._retriever.format_context(results)
        
        # Stream response
        chain = prompt | streaming_llm | StrOutputParser()
        
        for chunk in chain.stream({"context": context, "question": question}):
            yield chunk


def test_rag_chain():
    """Test the RAG chain with sample questions."""
    chain = LegalRAGChain()
    
    test_questions = [
        "What is the definition of 'consideration' under Malaysian contract law?",
        "When can a court grant specific performance of a contract?",
        "What are the licensing requirements for housing developers in Malaysia?",
    ]
    
    print("=" * 70)
    print("Legal RAG Chain Test")
    print("=" * 70)
    
    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"QUESTION: {question}")
        print("=" * 70)
        
        result = chain.ask(question)
        
        print("\nANSWER:")
        print("-" * 50)
        print(result["answer"])
        
        if result.get("sources"):
            print("\nSOURCES:")
            print("-" * 50)
            for source in result["sources"]:
                if isinstance(source, dict):
                    print(
                        f"  - {source['act_name']}, Section {source['section_number']}"
                    )
                else:
                    print(
                        f"  - {source.act_name}, Section {source.section_number}"
                    )


if __name__ == "__main__":
    test_rag_chain()
