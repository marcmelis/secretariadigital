# Secretaria Digital API

This is a Flask API that uses OpenAI's GPT model with a context fine tune to answer queries based on a set of documents. Users can send questions to the `/answer_query` endpoint, and the API will return an answer based on the precomputed embeddings of the documents.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/marcmelis/secretariadigital.git
```

2. Change into the project directory:

```bash
cd secretariadigital
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your API key as an environment variable:

```bash
export API_KEY="your_openai_api_key"
```

Replace `your_openai_api_key` with your actual OpenAI API key.

## Usage

1. Run the Flask server:

```bash
python app.py
```

2. Make a POST request to the `/answer_query` endpoint:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"user_message":"Your question here"}' http://localhost:5000/answer_query
```

Replace `"Your question here"` with your actual question.

## Deployment to Heroku

1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli).

2. Log in to your Heroku account:

```bash
heroku login
```

3. Create a new Heroku app:

```bash
heroku create your-app-name
```

Replace `your-app-name` with a unique name for your app.

4. Add your OpenAI API key to the Heroku app's environment variables:

```bash
heroku config:set API_KEY="your_openai_api_key"
```

Replace `your_openai_api_key` with your actual OpenAI API key.

5. Deploy your code to Heroku:

```bash
git push heroku main
```

6. Your API is now available at `https://your-app-name.herokuapp.com`. You can make a POST request to the `/answer_query` endpoint using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"user_message":"Your question here"}' https://your-app-name.herokuapp.com/answer_query
```

Replace `"Your question here"` with your actual question.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
