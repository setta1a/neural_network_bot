from llama_index import GPTSimpleVectorIndex, Document
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load and index the story
with open('data/text.txt', 'r') as t:
    data= t.read()

Sdoc = Document(data)
index = GPTSimpleVectorIndex([Sdoc])

# Querying the index
query= "summarize the text"
ssf= index.query(query).response
print("Relevant text:", ssf)

# Chat function using GPT-4
def chat(user_input):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are x"},
            # Add other system messages
            {"role": "system", "content": ssf},
            {"role": "user", "content": user_input},
        ],
    )

    response = completion.choices[0].message.content
    return response.strip()

# Example usage
user_input = "test"
response = chat(user_input)
print(response)