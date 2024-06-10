from flask import Flask, request, jsonify
from llms.recommendation_llm import RecommendationModel 

app = Flask(__name__)
llm = RecommendationModel()

# POST endpoint 
@app.route('/chat', methods=['POST'])
def chat():
    # Get the user's message from the request
    user_message = request.json['prompt']

    # Process the user's message and generate a response
    response = llm.querySimpleChromaDB(user_message, 4)

    # Return the response as a JSON object
    return jsonify({'response': response})

if __name__ == '__main__':
    # recommendationLLM = RecommendationModel()
    app.run(port=5000, debug=False)