import asyncio
from typing import AsyncGenerator

class StreamingRAGPipeline(RAGPipeline):
    async def ask_streaming(self, query, k=3, mode="beginner") -> AsyncGenerator:
        """Stream responses token by token"""
        
        # Stream retrieval status
        yield {"type": "status", "content": "🔍 Searching knowledge base..."}
        
        results = self.vector_store.search(query, k=k)
        chunks = [r["chunk"] for r in results]
        
        yield {"type": "status", "content": "💡 Generating explanation..."}
        
        # Stream generation (if your generator supports streaming)
        prompt = build_prompt(query, chunks, mode)
        
        # This would need your generator to support streaming
        # For now, just get full answer
        answer = generate_answer(prompt)
        
        yield {"type": "text", "content": answer}
        
        # Generate visualization
        if self.should_visualize(query):
            yield {"type": "status", "content": "📊 Creating visualization..."}
            
            viz_result = self.viz_manager.generate_visualization(query, answer)
            if viz_result['success']:
                yield {"type": "visualization", "content": viz_result['html']}
                yield {"type": "code", "content": viz_result['code']}
        
        yield {"type": "done"}