from processing.vector_store import VectorStore
from generation.generator import build_prompt, generate_answer
from visualization.visualization_manager import VisualizationManager
import traceback
import hashlib


class RAGPipeline:
    def __init__(self, index_path="my_faiss_index"):
        self.vector_store = VectorStore()
        self.vector_store.load(index_path)
        self.viz_manager = VisualizationManager()
        self.viz_cache = {}

    def ask(self, query, k=3, mode="beginner", use_visualization=True):
        results = self.vector_store.search(query, k=k)
        chunks  = [r["chunk"] for r in results]

        prompt = build_prompt(query, chunks, mode)
        answer = generate_answer(prompt)

        viz_html = viz_code = None

        if use_visualization:
            try:
                cache_key = hashlib.md5(f"{query}_{mode}".encode()).hexdigest()

                if cache_key in self.viz_cache:
                    viz_html, viz_code = self.viz_cache[cache_key]
                    print(f"[RAGPipeline] Using cached visualization for: {query[:50]}")
                else:
                    print(f"[RAGPipeline] Generating new visualization for: {query[:50]}")
                    viz_result = self.viz_manager.generate_visualization(
                        query=query, context=answer
                    )
                    if viz_result["success"]:
                        viz_html = viz_result["html"]
                        viz_code = viz_result["code"]
                        self.viz_cache[cache_key] = (viz_html, viz_code)
                        print("[RAGPipeline] Visualization generated successfully")
                    else:
                        print(f"[RAGPipeline] Visualization failed: {viz_result.get('error')}")

            except Exception as e:
                print(f"[RAGPipeline] Visualization error: {e}")
                traceback.print_exc()

        return {
            "question":     query,
            "answer":       answer,
            "sources":      chunks,
            "visualization": viz_html,
            "code":         viz_code,
            "success":      True,
        }

    def should_visualize(self, query: str) -> bool:
        """
        Always visualize — the LLM handles any topic.
        Short greetings and purely meta questions are excluded.
        """
        skip = {"hi", "hello", "thanks", "thank you", "bye", "ok", "okay", "yes", "no"}
        if query.strip().lower() in skip:
            return False
        # Exclude very short one-word inputs that aren't concept names
        if len(query.split()) == 1 and len(query) < 4:
            return False
        return True

    def clear_cache(self):
        self.viz_cache.clear()