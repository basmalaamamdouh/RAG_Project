from rag.rag_pipeline import RAGPipeline


def main():
    print(" Starting RAG System...")

    rag = RAGPipeline(index_path="my_faiss_index")

    while True:
        query = input("\n Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        result = rag.ask(query)

        print("\n Answer:")
        print(result["answer"])

        print("\n📚 Sources:")
        for i, src in enumerate(result["sources"], 1):
            print(f"{i}. {src[:150]}...")


if __name__ == "__main__":
    main()