from flask import Flask, request, jsonify
from flasgger import Swagger
import asyncio

from app.agent_infrastrure.agents.document_retriver import document_retriever
from app.evaluation.deep_eval_runner import run_deep_eval


app = Flask(__name__)

swagger = Swagger(app)


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "ArXiv RAG API",
        "version": "1.0.0",
        "swagger_url": "/apidocs/",
        "endpoints": {
            "retrieve_documents": "/retrieve_documents/"
        }
    }), 200


@app.route("/retrieve_documents/", methods=["POST"])
def retrieve_documents_endpoint():
    """
    Retrieve relevant ArXiv documents based on user query
    ---
    tags:
      - Document Retrieval
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            user_query:
              type: string
              example: "brownian motion"
    responses:
      200:
        description: Documents + agent response
    """
    data = request.json
    user_query = data.get("user_query")
    if not user_query:
        return jsonify({"error": "Missing 'user_query'"}), 400
    
    documents = asyncio.run(document_retriever.ainvoke({"user_query": user_query}))

    results = [
        {"title": doc.metadata.get("title"),
         "category": doc.metadata.get("category"),
         "abstract": doc.page_content}
        for doc in documents
    ]

    # return jsonify(results)

    agent_output = " ".join([d["abstract"] for d in results]) if results else "No docs found."

    run_deep_eval(user_query, agent_output, results, tools_used=["document_retriever"])

    return jsonify({"documents": results, "agent_output": agent_output})


