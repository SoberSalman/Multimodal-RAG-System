# src/core/llm_client.py

import requests
import json
import logging
from typing import List, Dict, Generator

logger = logging.getLogger(__name__)

class LocalLLM:
    """Client for interacting with local LLM (LM Studio)"""
    
    def __init__(
        self,
        endpoint_url: str = "http://0.0.0.0:1234/v1/chat/completions",
        model: str = "gemma-2-2b-it:2",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        self.endpoint_url = endpoint_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to local LLM server"""
        try:
            response = requests.get(self.endpoint_url.replace("/chat/completions", "/models"))
            if response.status_code == 200:
                logger.info("Successfully connected to local LLM server")
            else:
                logger.warning(f"Could not connect to local LLM server: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to local LLM server: {e}")
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate response from the LLM"""
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream
        }
        
        try:
            if stream:
                return self._generate_stream(payload)
            else:
                response = requests.post(
                    self.endpoint_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.error(f"Error from LLM: {response.status_code} - {response.text}")
                    return f"Error: {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def _generate_stream(self, payload: Dict) -> Generator[str, None, None]:
        """Generate response from the LLM with streaming"""
        try:
            with requests.post(
                self.endpoint_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                
                if response.status_code != 200:
                    yield f"Error: {response.status_code} - {response.text}"
                    return
                
                for line in response.iter_lines():
                    if not line:
                        continue
                        
                    decoded_line = line.decode('utf-8')
                    
                    if decoded_line.startswith("data: "):
                        decoded_line = decoded_line.replace("data: ", "")
                    
                    if decoded_line == "[DONE]":
                        break
                    
                    try:
                        data_json = json.loads(decoded_line)
                        if "choices" in data_json and len(data_json["choices"]) > 0:
                            delta = data_json["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_prompt_template(self, prompt_type: str, context: str, question: str) -> str:
        """Get prompt template based on type"""
        templates = {
            "zero_shot": f"""Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:""",
            
            "cot": f"""Answer the question based on the following context. Think step by step.

Context: {context}

Question: {question}

Let's approach this step by step:
1. First, I'll identify the key information in the context
2. Then, I'll analyze how it relates to the question
3. Finally, I'll formulate a clear and complete answer

Answer:""",
            
            "few_shot": f"""Answer the question based on the following context. Here are some examples:

Example 1:
Context: The revenue for Q1 2023 was $1.2 million.
Question: What was the Q1 revenue?
Answer: The Q1 revenue was $1.2 million.

Example 2:
Context: The chart shows increasing profit margins from 15% to 25% over 3 years.
Question: How did profit margins change?
Answer: Profit margins increased from 15% to 25% over a 3-year period.

Now answer this question:
Context: {context}
Question: {question}
Answer:"""
        }
        
        return templates.get(prompt_type, templates["zero_shot"])
