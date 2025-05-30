from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()
# Set API credentials and endpoint
openai_api_key = os.getenv("LAMBDA_API_KEY")
openai_api_base = "https://api.lambda.ai/v1"

# Initialize the OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# List available models from the Lambda Inference API and print the result
models = client.models.list()
for model in models.data:
    print(model.id)
