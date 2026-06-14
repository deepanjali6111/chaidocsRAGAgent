import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from chaidocs_rag import ChaiDocsRAG

load_dotenv()

# ─────────────────────────────────────────────────────────────
# FIX: Subclass ChatGoogleGenerativeAI and strip bad kwargs
# that RAGAS injects at call-time which gemini-2.5-flash rejects
# ─────────────────────────────────────────────────────────────
class SafeGemini(ChatGoogleGenerativeAI):
    """
    Strips kwargs that RAGAS passes at call-time (temperature, top_p etc.)
    which gemini-2.5-flash rejects via GenerativeServiceClient.
    These must be set at construction time via GenerationConfig, not at call-time.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        kwargs.pop("top_k", None)
        kwargs.pop("max_tokens", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        kwargs.pop("top_k", None)
        kwargs.pop("max_tokens", None)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


# ─────────────────────────────────────────────────────────────
# Step 1 — Initialize RAG
# ─────────────────────────────────────────────────────────────
print("Initializing RAG system...")
rag = ChaiDocsRAG()
rag.process_docs()
rag.setup_chain()

# ─────────────────────────────────────────────────────────────
# Step 2 — Test questions
# ─────────────────────────────────────────────────────────────
test_questions = [
    "What is Django?",
    "How to integrate Tailwind in a Django project?",
    "What is Nginx?",
    "How to configure Nginx on a VPS?",
    "What is Git and GitHub?"
]

# ─────────────────────────────────────────────────────────────
# IMPORTANT: Ground truths MUST semantically match questions
# Mismatched ground truths are the #1 cause of NaN scores
# ─────────────────────────────────────────────────────────────
ground_truths = [
    "Django is a high-level Python web framework for rapid development.",
    "Tailwind can be integrated into Django using npm and configuring static files.",
    "Nginx is a high-performance web server and reverse proxy.",
    "Configure Nginx on a VPS by editing /etc/nginx/sites-available and reloading.",
    "Git is a version control system; GitHub is a platform for hosting Git repositories."
]

# ─────────────────────────────────────────────────────────────
# Step 3 — Collect answers and contexts from RAG
# ─────────────────────────────────────────────────────────────
answers = []
contexts = []

for question in test_questions:
    print(f"\nQuerying: {question}")
    answer = rag.query(question)
    answers.append(answer)

    retrieved_docs = rag.retriever.invoke(question)
    context_texts = [doc.page_content for doc in retrieved_docs]
    contexts.append(context_texts)

    # Debug prints — tells you exactly why NaN happens
    print(f"  Answer length   : {len(answer)} chars")
    print(f"  Chunks retrieved: {len(context_texts)}")
    if not context_texts:
        print("  WARNING: 0 chunks retrieved → context_precision WILL be NaN")
    if not answer or len(answer) < 10:
        print("  WARNING: Empty/very short answer → faithfulness WILL be NaN")

# ─────────────────────────────────────────────────────────────
# Step 4 — Build RAGAS dataset
# ─────────────────────────────────────────────────────────────
dataset = Dataset.from_dict({
    "question":     test_questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})

# ─────────────────────────────────────────────────────────────
# Step 5 — Setup SafeGemini as evaluator
# temperature set HERE at construction, not injected at call-time
# ─────────────────────────────────────────────────────────────
eval_llm = LangchainLLMWrapper(
    SafeGemini(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,                        # set once here, never at call-time
        convert_system_message_to_human=True,   # required for Gemini + LangChain
    )
)

eval_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# ─────────────────────────────────────────────────────────────
# Step 6 — Run RAGAS evaluation
# ─────────────────────────────────────────────────────────────
print("\nRunning RAGAS evaluation...")
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=eval_llm,
    embeddings=eval_embeddings,
    raise_exceptions=False,   # don't crash on single metric failure
)

# ─────────────────────────────────────────────────────────────
# Step 7 — Print results
# ─────────────────────────────────────────────────────────────
def fmt(val):
    try:
        return f"{float(val):.3f}"
    except Exception:
        return "NaN — check warnings above"

print("\n=== RAGAS Results ===")
print(f"Faithfulness:      {fmt(results['faithfulness'])}")
print(f"Answer Relevancy:  {fmt(results['answer_relevancy'])}")
print(f"Context Precision: {fmt(results['context_precision'])}")