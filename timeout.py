from openai import OpenAI
import os
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()

# Set API credentials and endpoint
openai_api_key = os.getenv("LAMBDA_API_KEY")
openai_api_base = "https://api.lambda.ai/v1"

# Initialize the OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Choose the model
model = "llama-4-scout-17b-16e-instruct"
for i in tqdm(range(100)):
    # Create a multi-turn chat completion request
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": "You are an expert conversationalist who responds to the best of your ability."
        }, {
            "role": "user",
            "content": "Who won the world series in 2020?"
        }, {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020."
        }, {
            "role": "user",
            "content": "Where was it played?"
        }],
        model=model,
    )

    # Print the full chat completion response
    