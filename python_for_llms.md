# Python for Large Language Models: Complete Guide

## Table of Contents
1. [Python Fundamentals for LLMs](#python-fundamentals-for-llms)
2. [Essential Libraries](#essential-libraries)
3. [Environment Setup](#environment-setup)
4. [Working with LLM APIs](#working-with-llm-apis)
5. [Key Frameworks](#key-frameworks)
6. [Data Processing for LLMs](#data-processing-for-llms)
7. [Building LLM Applications](#building-llm-applications)
8. [Advanced Techniques](#advanced-techniques)
9. [Practical Projects](#practical-projects)
10. [Best Practices](#best-practices)

---

## Python Fundamentals for LLMs

### Core Concepts You Need

#### 1. String Manipulation
```python
# Essential for prompt engineering and text processing
text = "Hello, world!"

# String formatting (crucial for dynamic prompts)
name = "Alice"
prompt = f"Hello {name}, please explain quantum physics."

# Multi-line strings for complex prompts
complex_prompt = """
You are an expert teacher. Please explain the following concept:
Topic: {topic}
Audience: {audience}
Style: {style}
"""

formatted_prompt = complex_prompt.format(
    topic="Machine Learning",
    audience="beginners",
    style="conversational"
)
```

#### 2. Lists and Dictionaries
```python
# Managing conversation history
conversation = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI stands for Artificial Intelligence..."},
    {"role": "user", "content": "How does it work?"}
]

# Model parameters
model_config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0
}
```

#### 3. Functions and Classes
```python
# Reusable prompt functions
def create_coding_prompt(language, task, difficulty="beginner"):
    return f"""
    Write a {language} program to {task}.
    Difficulty level: {difficulty}
    Please include comments explaining the code.
    """

# LLM wrapper class
class LLMClient:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.conversation_history = []
    
    def add_message(self, role, content):
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_response(self, message):
        # Implementation would go here
        pass
```

#### 4. Error Handling
```python
import time
import random

def robust_api_call(client, prompt, max_retries=3):
    """Make API calls with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise e
```

#### 5. File I/O
```python
import json

# Reading prompts from files
def load_prompt_template(filename):
    with open(filename, 'r') as file:
        return file.read()

# Saving conversation logs
def save_conversation(conversation, filename):
    with open(filename, 'w') as file:
        json.dump(conversation, file, indent=2)

# Loading training data
def load_training_data(filename):
    with open(filename, 'r') as file:
        return [json.loads(line) for line in file]
```

---

## Essential Libraries

### 1. HTTP Requests
```python
# For API calls
import requests
import json

def simple_openai_call(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    return response.json()
```

### 2. Environment Variables
```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Safely access API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
```

### 3. Data Processing
```python
import pandas as pd
import numpy as np

# Processing datasets for LLM training/evaluation
def prepare_conversation_data(csv_file):
    df = pd.read_csv(csv_file)
    
    conversations = []
    for _, row in df.iterrows():
        conversation = {
            "messages": [
                {"role": "user", "content": row['user_message']},
                {"role": "assistant", "content": row['assistant_response']}
            ]
        }
        conversations.append(conversation)
    
    return conversations
```

### 4. Text Processing
```python
import re
from typing import List

def clean_text(text: str) -> str:
    """Clean text for LLM processing"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters if needed
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    """Split text into chunks for processing"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

---

## Environment Setup

### 1. Virtual Environment
```bash
# Create virtual environment
python -m venv llm_env

# Activate (Linux/Mac)
source llm_env/bin/activate

# Activate (Windows)
llm_env\Scripts\activate

# Install packages
pip install openai anthropic transformers torch numpy pandas python-dotenv
```

### 2. Requirements File
```python
# requirements.txt
openai==1.3.0
anthropic==0.7.0
transformers==4.35.0
torch==2.1.0
numpy==1.24.3
pandas==2.0.3
python-dotenv==1.0.0
requests==2.31.0
streamlit==1.28.0
langchain==0.0.335
```

### 3. Environment Variables (.env file)
```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
```

---

## Working with LLM APIs

### 1. OpenAI API
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_gpt(messages, model="gpt-3.5-turbo", temperature=0.7):
    """Chat with GPT models"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
messages = [
    {"role": "system", "content": "You are a helpful Python tutor."},
    {"role": "user", "content": "Explain list comprehensions in Python."}
]

response = chat_with_gpt(messages)
print(response)
```

### 2. Anthropic Claude API
```python
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def chat_with_claude(prompt, model="claude-3-sonnet-20240229"):
    """Chat with Claude models"""
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
response = chat_with_claude("Explain the difference between lists and tuples in Python.")
print(response)
```

### 3. Streaming Responses
```python
def stream_gpt_response(messages):
    """Stream GPT responses for real-time display"""
    try:
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="")
                full_response += content
        
        return full_response
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### 4. Function Calling
```python
def get_weather(location):
    """Mock weather function"""
    return f"The weather in {location} is sunny with 72°F"

def chat_with_functions():
    """Example of function calling with GPT"""
    functions = [
        {
            "name": "get_weather",
            "description": "Get current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "What's the weather like in New York?"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=functions,
        function_call="auto"
    )
    
    if response.choices[0].message.function_call:
        function_name = response.choices[0].message.function_call.name
        function_args = json.loads(response.choices[0].message.function_call.arguments)
        
        if function_name == "get_weather":
            result = get_weather(function_args["location"])
            return result
    
    return response.choices[0].message.content
```

---

## Key Frameworks

### 1. LangChain
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Basic LangChain setup
llm = OpenAI(temperature=0.7)

# Prompt templates
prompt_template = PromptTemplate(
    input_variables=["topic", "audience"],
    template="Explain {topic} to {audience} in simple terms."
)

# Chains
chain = LLMChain(llm=llm, prompt=prompt_template)
result = chain.run(topic="machine learning", audience="children")

# Memory for conversations
memory = ConversationBufferMemory()
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory
)
```

### 2. Transformers (Hugging Face)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load a pre-trained model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(input_text, max_length=100):
    """Generate response using local model"""
    # Encode input
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(input_text):].strip()

# Example usage
response = generate_response("Hello, how are you?")
print(response)
```

### 3. Streamlit for Web Apps
```python
import streamlit as st
from openai import OpenAI

st.title("LLM Chat Interface")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages
        )
        assistant_response = response.choices[0].message.content
        st.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
```

---

## Data Processing for LLMs

### 1. Text Preprocessing
```python
import re
import string
from collections import Counter

def preprocess_text(text):
    """Comprehensive text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation (optional)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text.strip()

def create_training_dataset(conversations):
    """Format conversations for training"""
    formatted_data = []
    
    for conv in conversations:
        formatted_conv = {
            "instruction": conv["user_message"],
            "input": "",
            "output": conv["assistant_response"]
        }
        formatted_data.append(formatted_conv)
    
    return formatted_data
```

### 2. Token Counting
```python
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text for a specific model"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def truncate_text(text, max_tokens, model="gpt-3.5-turbo"):
    """Truncate text to fit within token limit"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

# Example usage
text = "Your long text here..."
token_count = count_tokens(text)
print(f"Token count: {token_count}")

if token_count > 4000:
    truncated_text = truncate_text(text, 4000)
    print("Text truncated to fit token limit")
```

### 3. Batch Processing
```python
import asyncio
import aiohttp
from typing import List, Dict

async def process_batch_async(prompts: List[str], api_key: str) -> List[str]:
    """Process multiple prompts asynchronously"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async def process_single_prompt(session, prompt):
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        ) as response:
            result = await response.json()
            return result["choices"][0]["message"]["content"]
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_single_prompt(session, prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results

# Usage
prompts = ["Explain AI", "What is Python?", "How do computers work?"]
# results = asyncio.run(process_batch_async(prompts, api_key))
```

---

## Building LLM Applications

### 1. Chatbot Class
```python
class ChatBot:
    def __init__(self, system_prompt="You are a helpful assistant.", model="gpt-3.5-turbo"):
        self.system_prompt = system_prompt
        self.model = model
        self.conversation_history = [{"role": "system", "content": system_prompt}]
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def chat(self, user_message):
        """Send a message and get a response"""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        try:
            # Get response from LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        
        except Exception as e:
            return f"Error: {e}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
    
    def save_conversation(self, filename):
        """Save conversation to file"""
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

# Usage
bot = ChatBot("You are a Python programming tutor.")
response = bot.chat("How do I create a list in Python?")
print(response)
```

### 2. Document Q&A System
```python
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentQA:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def find_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Find most relevant documents for a query"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
    
    def answer_question(self, question: str) -> str:
        """Answer a question based on the documents"""
        relevant_docs = self.find_relevant_docs(question)
        context = "\n\n".join(relevant_docs)
        
        prompt = f"""
        Based on the following context, answer the question:
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
documents = [
    "Python is a programming language created by Guido van Rossum.",
    "Lists in Python are ordered collections that can hold different data types.",
    "Functions in Python are defined using the 'def' keyword."
]

qa_system = DocumentQA(documents)
answer = qa_system.answer_question("Who created Python?")
print(answer)
```

### 3. Code Generation Assistant
```python
class CodeAssistant:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_code(self, description: str, language: str = "python") -> str:
        """Generate code based on description"""
        prompt = f"""
        Write a {language} function that {description}.
        
        Requirements:
        - Include proper error handling
        - Add docstrings and comments
        - Follow best practices
        - Include example usage
        
        Code:
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def explain_code(self, code: str) -> str:
        """Explain what a piece of code does"""
        prompt = f"""
        Explain this code in detail:
        
        ```
        {code}
        ```
        
        Please explain:
        1. What the code does
        2. How it works step by step
        3. Any important concepts used
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def debug_code(self, code: str, error: str) -> str:
        """Help debug code with error message"""
        prompt = f"""
        Help me debug this code:
        
        Code:
        ```
        {code}
        ```
        
        Error:
        {error}
        
        Please:
        1. Identify the problem
        2. Explain why it's happening
        3. Provide the corrected code
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
assistant = CodeAssistant()
code = assistant.generate_code("sorts a list of numbers in ascending order")
print(code)
```

---

## Advanced Techniques

### 1. Prompt Engineering Functions
```python
def create_few_shot_prompt(examples: List[Dict], task_description: str, new_input: str) -> str:
    """Create a few-shot learning prompt"""
    prompt = f"{task_description}\n\n"
    
    # Add examples
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"
    
    # Add new input
    prompt += f"Now, please handle this input:\n"
    prompt += f"Input: {new_input}\n"
    prompt += f"Output:"
    
    return prompt

def create_chain_of_thought_prompt(problem: str) -> str:
    """Create a chain-of-thought reasoning prompt"""
    return f"""
    Let's solve this step by step.
    
    Problem: {problem}
    
    Please think through this carefully:
    1. First, let me understand what's being asked
    2. Then, I'll identify the key information
    3. Next, I'll work through the solution step by step
    4. Finally, I'll provide the answer
    
    Let's begin:
    """

# Usage examples
examples = [
    {"input": "The cat sat on the mat", "output": "Positive"},
    {"input": "I hate this weather", "output": "Negative"}
]

prompt = create_few_shot_prompt(
    examples, 
    "Classify the sentiment of the following text as Positive or Negative:",
    "I love sunny days"
)
```

### 2. Response Validation
```python
import json
from typing import Optional, Dict, Any

def validate_json_response(response: str) -> Optional[Dict[Any, Any]]:
    """Validate and parse JSON response from LLM"""
    try:
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            json_str = response.strip()
        
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def ensure_json_response(client, prompt: str, max_retries: int = 3) -> Optional[Dict]:
    """Ensure LLM returns valid JSON"""
    json_prompt = f"""
    {prompt}
    
    Please respond with valid JSON only. No explanation needed.
    Format your response as:
    ```json
    {{
        "your": "response"
    }}
    ```
    """
    
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": json_prompt}],
            temperature=0.1
        )
        
        result = validate_json_response(response.choices[0].message.content)
        if result:
            return result
        
        print(f"Attempt {attempt + 1} failed to return valid JSON")
    
    return None
```

### 3. Cost Tracking
```python
class CostTracker:
    def __init__(self):
        # Pricing per 1K tokens (as of 2024)
        self.pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12}
        }
        self.total_cost = 0.0
        self.usage_log = []
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request"""
        if model not in self.pricing:
            return 0.0
        
        input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
        total = input_cost + output_cost
        
        self.total_cost += total
        self.usage_log.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total
        })
        
        return total
    
    def get_usage_summary(self) -> Dict:
        """Get usage summary"""
        return {
            "total_cost": self.total_cost,
            "total_requests": len(self.usage_log),
            "average_cost_per_request": self.total_cost / len(self.usage_log) if self.usage_log else 0
        }

# Usage
tracker = CostTracker()
cost = tracker.calculate_cost("gpt-4", 100, 50)
print(f"Request cost: ${cost:.4f}")
```

---

## Practical Projects

### 1. Simple AI Writing Assistant
```python
class WritingAssistant:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def improve_text(self, text: str, style: str = "professional") -> str:
        """Improve writing style and clarity"""
        prompt = f"""
        Please improve the following text to be more {style}.
        Focus on clarity, grammar, and flow while maintaining the original meaning.
        
        Original text:
        {text}
        
        Improved text:
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def generate_outline(self, topic: str, target_audience: str) -> str:
        """Generate content outline"""
        prompt = f"""
        Create a detailed outline for content about "{topic}" 
        targeted at {target_audience}.
        
        Include:
        - Main sections and subsections
        - Key points for each section
        - Estimated word counts
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content

# Usage
assistant = WritingAssistant()
improved = assistant.improve_text("This thing is really good and works well.", "academic")
print(improved)
```

### 2. Data Analysis Helper
```python
import pandas as pd

class DataAnalysisHelper:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def analyze_dataframe(self, df: pd.DataFrame, question: str) -> str:
        """Analyze DataFrame and answer questions"""
        # Generate data summary
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head().to_dict()
        }
        
        prompt = f"""
        I have a dataset with the following characteristics:
        
        Shape: {summary['shape']}
        Columns: {summary['columns']}
        Data types: {summary['dtypes']}
        Missing values: {summary['missing_values']}
        
        Sample data:
        {summary['sample_data']}
        
        Question: {question}
        
        Please provide insights and suggest analysis approaches.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def suggest_visualizations(self, df: pd.DataFrame) -> str:
        """Suggest appropriate visualizations"""
        # Analyze column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        prompt = f"""
        Based on this dataset structure, suggest appropriate visualizations:
        
        Numeric columns: {numeric_cols}
        Categorical columns: {categorical_cols}
        Dataset shape: {df.shape}
        
        Please suggest:
        1. Most important visualizations to create
        2. Python code using matplotlib/seaborn
        3. Insights each visualization would reveal
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
# df = pd.read_csv("your_data.csv")
# helper = DataAnalysisHelper()
# insights = helper.analyze_dataframe(df, "What are the main trends in this data?")
```

---

## Best Practices

### 1. Security
```python
import os
import hashlib
from functools import wraps

def secure_api_key():
    """Securely manage API keys"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables")
    
    # Never log or print API keys
    # Use environment variables or secure vaults
    return api_key

def rate_limit(max_calls_per_minute=60):
    """Simple rate limiting decorator"""
    import time
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls older than 1 minute
            calls[:] = [call_time for call_time in calls if now - call_time < 60]
            
            if len(calls) >= max_calls_per_minute:
                sleep_time = 60 - (now - calls[0])
                time.sleep(sleep_time)
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls_per_minute=30)
def safe_api_call(prompt):
    # Your API call here
    pass
```

### 2. Logging and Monitoring
```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_app.log'),
        logging.StreamHandler()
    ]
)

class LLMLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def log_request(self, prompt: str, model: str, tokens_used: int):
        """Log API request details"""
        self.logger.info(f"Request - Model: {model}, Tokens: {tokens_used}, Prompt length: {len(prompt)}")
    
    def log_error(self, error: Exception, context: str):
        """Log errors with context"""
        self.logger.error(f"Error in {context}: {str(error)}")
    
    def log_performance(self, operation: str, duration: float):
        """Log performance metrics"""
        self.logger.info(f"Performance - {operation}: {duration:.2f}s")

# Usage
logger = LLMLogger()
logger.log_request("Example prompt", "gpt-3.5-turbo", 150)
```

### 3. Testing
```python
import unittest
from unittest.mock import Mock, patch

class TestLLMFunctions(unittest.TestCase):
    
    def setUp(self):
        self.mock_client = Mock()
    
    @patch('openai.OpenAI')
    def test_chat_function(self, mock_openai):
        """Test chat function with mocked API"""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        # Test your function
        result = chat_with_gpt([{"role": "user", "content": "test"}])
        
        # Assertions
        self.assertEqual(result, "Test response")
        mock_openai.return_value.chat.completions.create.assert_called_once()
    
    def test_prompt_template(self):
        """Test prompt template generation"""
        template = create_coding_prompt("Python", "sort a list")
        
        self.assertIn("Python", template)
        self.assertIn("sort a list", template)
        self.assertIn("comments", template)

if __name__ == "__main__":
    unittest.main()
```

### 4. Configuration Management
```python
from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class LLMConfig:
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    max_retries: int = 3
    api_key: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        with open(config_path, 'w') as file:
            yaml.dump(self.__dict__, file, default_flow_style=False)

# config.yaml example:
"""
model: "gpt-4"
temperature: 0.5
max_tokens: 2000
timeout: 45
max_retries: 5
"""

# Usage
config = LLMConfig.from_yaml("config.yaml")
```

---

## Getting Started Checklist

### Step 1: Environment Setup
- [ ] Install Python 3.8+
- [ ] Create virtual environment
- [ ] Install required packages
- [ ] Set up environment variables
- [ ] Test basic imports

### Step 2: API Access
- [ ] Get OpenAI API key
- [ ] Test basic API call
- [ ] Understand rate limits and pricing
- [ ] Set up error handling

### Step 3: Practice Projects
- [ ] Build a simple chatbot
- [ ] Create a text summarizer
- [ ] Make a code explanation tool
- [ ] Build a question-answering system

### Step 4: Advanced Features
- [ ] Implement streaming responses
- [ ] Add conversation memory
- [ ] Create prompt templates
- [ ] Build a web interface

### Step 5: Production Ready
- [ ] Add comprehensive error handling
- [ ] Implement logging and monitoring
- [ ] Add security measures
- [ ] Write tests
- [ ] Create documentation

---

This guide provides everything you need to start working with LLMs in Python. Begin with the fundamentals, practice with the examples, and gradually work your way up to building complete applications!