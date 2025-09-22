from flask import Flask, request, jsonify
from flasgger import Swagger
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from app.agent_infrastrure.agents.main_agent import main_agent
from app.core.security import verify_token

load_dotenv()

app = Flask(__name__)

swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "ArXiv RAG API",
        "version": "1.0.0"
    },
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "🔑 Use format: Bearer <your_token>"
        }
    },
    "security": [
        {"Bearer": []}
    ]
})


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "ArXiv RAG API",
        "version": "1.0.0",
        "swagger_url": "/apidocs/",
        "endpoints": {
            "chat": "/chat/"
        }
    }), 200


@app.route("/chat/", methods=["POST"])
@verify_token
def chat():
    """
    Chat with the arxiv agent
    ---
    tags:
      - Chat
    security:
      - Bearer: []
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            user_query:
              type: string
    responses:
      200:
        description: Agent response
    """
    data = request.json
    user_query = data.get("user_query")
    if not user_query:
        return jsonify({"error": "Missing 'user_query'"}), 400
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    response = asyncio.run(main_agent.ainvoke({
        "messages": [{"role": "user", "content": user_query}],
        "date": current_date
    }))
    
    agent_response = response["messages"][-1].content
    return jsonify({"response": agent_response})
