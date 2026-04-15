#!/usr/bin/env python3
"""
NeuralLens — AI/ML Visual Tutor
Upgraded Flask app with streaming, quiz mode, and concept cards
"""

import os
import json
import hashlib
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from rag.rag_pipeline import RAGPipeline

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
CORS(app)

rag_pipeline = RAGPipeline(index_path="my_faiss_index")

# ─── Concept cards data ───────────────────────────────────────────────────────
CONCEPTS = [
    {"id": "knn",        "title": "K-Nearest Neighbors", "icon": "🔵", "category": "supervised",   "query": "Explain and visualize how KNN works"},
    {"id": "svm",        "title": "Support Vector Machine","icon": "🔷","category": "supervised",   "query": "Explain and visualize how SVM works"},
    {"id": "dtree",      "title": "Decision Tree",        "icon": "🌳", "category": "supervised",   "query": "Explain and visualize how decision trees work"},
    {"id": "rforest",    "title": "Random Forest",        "icon": "🌲", "category": "supervised",   "query": "Explain and visualize how random forests work"},
    {"id": "linreg",     "title": "Linear Regression",   "icon": "📈", "category": "supervised",   "query": "Explain and visualize linear regression"},
    {"id": "logreg",     "title": "Logistic Regression", "icon": "📊", "category": "supervised",   "query": "Explain and visualize logistic regression"},
    {"id": "kmeans",     "title": "K-Means Clustering",  "icon": "⭕", "category": "unsupervised", "query": "Explain and visualize K-Means clustering"},
    {"id": "pca",        "title": "PCA",                 "icon": "🔻", "category": "unsupervised", "query": "Explain and visualize Principal Component Analysis"},
    {"id": "tsne",       "title": "t-SNE",               "icon": "🌀", "category": "unsupervised", "query": "Explain and visualize t-SNE dimensionality reduction"},
    {"id": "dbscan",     "title": "DBSCAN",              "icon": "🔗", "category": "unsupervised", "query": "Explain and visualize DBSCAN clustering"},
    {"id": "nn",         "title": "Neural Networks",     "icon": "🧠", "category": "deep",         "query": "Explain and visualize how neural networks work"},
    {"id": "cnn",        "title": "CNN",                 "icon": "🖼️", "category": "deep",         "query": "Explain and visualize convolutional neural networks"},
    {"id": "rnn",        "title": "RNN & LSTM",          "icon": "🔄", "category": "deep",         "query": "Explain and visualize RNN and LSTM networks"},
    {"id": "attention",  "title": "Attention Mechanism", "icon": "👁️", "category": "deep",         "query": "Explain and visualize the attention mechanism in transformers"},
    {"id": "gd",         "title": "Gradient Descent",   "icon": "⛰️", "category": "optimization", "query": "Explain and visualize gradient descent optimization"},
    {"id": "backprop",   "title": "Backpropagation",     "icon": "↩️", "category": "optimization", "query": "Explain and visualize backpropagation"},
    {"id": "overfitting","title": "Overfitting & Bias",  "icon": "⚖️", "category": "optimization", "query": "Explain and visualize overfitting, underfitting, bias variance tradeoff"},
    {"id": "regularize", "title": "Regularization",      "icon": "🛡️", "category": "optimization", "query": "Explain L1 L2 regularization and visualize their effect"},
]

CATEGORIES = {
    "supervised":   "Supervised Learning",
    "unsupervised": "Unsupervised Learning",
    "deep":         "Deep Learning",
    "optimization": "Optimization & Theory",
}

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/api/concepts', methods=['GET'])
def get_concepts():
    return jsonify({"concepts": CONCEPTS, "categories": CATEGORIES})


@app.route('/api/chat', methods=['POST'])
def chat():
    """Standard (non-streaming) chat endpoint"""
    try:
        data = request.json
        query = data.get('query', '')
        mode  = data.get('mode', 'beginner')

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        result = rag_pipeline.ask(query=query, k=3, mode=mode, use_visualization=True)

        if 'history' not in session:
            session['history'] = []
        session['history'].append({
            'query':   query,
            'response': result['answer'],
            'mode':    mode,
            'has_viz': result['visualization'] is not None
        })

        return jsonify({
            'text':          result['answer'],
            'visualization': result['visualization'],
            'code':          result.get('code'),
            'sources':       result.get('sources', []),
            'success':       result['success']
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Server-Sent Events streaming endpoint with visualization support"""
    data  = request.json
    query = data.get('query', '')
    mode  = data.get('mode', 'beginner')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    def generate():
        import traceback
        from groq import Groq
        from generation.generator import build_prompt

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        try:
            # Step 1 — retrieval status
            yield f"data: {json.dumps({'type': 'status', 'content': '🔍 Searching knowledge base...'})}\n\n"

            results = rag_pipeline.vector_store.search(query, k=3)
            chunks  = [r['chunk'] for r in results]
            prompt  = build_prompt(query, chunks, mode)

            # Step 2 — streaming tokens
            yield f"data: {json.dumps({'type': 'status', 'content': '✍️ Generating explanation...'})}\n\n"

            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=True
            )

            full_answer = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content or ''
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Step 3 — visualization (ALWAYS generate for ML concepts)
            yield f"data: {json.dumps({'type': 'status', 'content': '📊 Generating visualization...'})}\n\n"

            cache_key = hashlib.md5(f"{query}_{mode}".encode()).hexdigest()
            
            if cache_key in rag_pipeline.viz_cache:
                viz_html, viz_code = rag_pipeline.viz_cache[cache_key]
                print(f"[Stream] Using cached visualization for: {query[:50]}")
            else:
                print(f"[Stream] Generating new visualization for: {query[:50]}")
                viz_result = rag_pipeline.viz_manager.generate_visualization(
                    query=query, context=full_answer
                )
                viz_html = viz_result.get('html')
                viz_code = viz_result.get('code')
                if viz_result['success'] and viz_html:
                    rag_pipeline.viz_cache[cache_key] = (viz_html, viz_code)
                    print(f"[Stream] Visualization generated successfully")
                else:
                    print(f"[Stream] Visualization failed: {viz_result.get('error')}")

            # Send visualization HTML
            if viz_html:
                yield f"data: {json.dumps({'type': 'visualization', 'content': viz_html})}\n\n"
            
            # Send code for the "View code" button
            if viz_code:
                yield f"data: {json.dumps({'type': 'code', 'content': viz_code})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Content-Type': 'text/event-stream'
        }
    )


@app.route('/api/quiz', methods=['POST'])
def quiz():
    """Generate MCQ quiz questions from a topic"""
    try:
        data  = request.json
        topic = data.get('topic', '')
        mode  = data.get('mode', 'beginner')

        if not topic:
            return jsonify({'error': 'No topic provided'}), 400

        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        results = rag_pipeline.vector_store.search(topic, k=5)
        chunks  = [r['chunk'] for r in results]
        context = "\n\n".join(chunks)

        difficulty = {
            'beginner':     'simple conceptual questions',
            'intermediate': 'applied and analytical questions',
            'expert':       'deep mathematical and implementation questions'
        }.get(mode, 'moderate questions')

        prompt = f"""You are an ML educator creating a quiz.

Topic: {topic}
Context: {context}

Generate exactly 4 multiple choice questions ({difficulty}).
Return ONLY valid JSON, no other text:

{{
  "questions": [
    {{
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct": 0,
      "explanation": "Brief explanation of the correct answer"
    }}
  ]
}}

The correct field is the 0-based index of the correct option."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        import re
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)

        quiz_data = json.loads(raw)
        return jsonify({'success': True, 'quiz': quiz_data, 'topic': topic})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/modes', methods=['GET'])
def get_modes():
    return jsonify({
        'modes': ['beginner', 'intermediate', 'expert'],
        'descriptions': {
            'beginner':     'Simple explanations with analogies',
            'intermediate': 'Technical but accessible explanations',
            'expert':       'Deep mathematical and algorithmic details'
        }
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify({'history': session.get('history', [])})


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    session['history'] = []
    return jsonify({'success': True})


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    rag_pipeline.clear_cache()
    return jsonify({'success': True, 'message': 'Cache cleared'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)