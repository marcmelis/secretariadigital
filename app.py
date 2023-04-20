from flask import Flask, request, jsonify

# Import your existing functions and variables
from assistant import (
    answer_query_with_context, get_embeddings, get_df,
    CONFIG_FILE, EMBEDDINGS_FILE, DEFAULT_LANGUAGE, API_KEY, COMPLETIONS_MODEL,
    COMPLETIONS_API_PARAMS, MAX_SECTION_LEN, SEPARATOR, tokenizer, separator_len,
    get_query_embedding, order_document_sections_by_query_similarity,
    construct_prompt, generate_chat_completion_message_from_prompt
)

app = Flask(__name__)

document_embeddings = get_embeddings()
df = get_df()

@app.route('/answer_query', methods=['POST'])
def answer_query():
    data = request.get_json()
    user_message = data['user_message']
    if user_message:
        answer = answer_query_with_context(user_message, df, document_embeddings)
        return jsonify({'answer': answer})
    else:
        return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(debug=True)
