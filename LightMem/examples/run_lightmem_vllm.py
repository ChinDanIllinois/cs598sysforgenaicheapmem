import json
import os
import time
from tqdm import tqdm
from typing import Dict, List, Optional
from openai import OpenAI

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dotenv
dotenv.load_dotenv()

from lightmem.memory.lightmem import LightMemory


# =========== vLLM Configuration ============
# Default to localhost if not specified
your_vllm_model_name = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
your_vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
your_vllm_api_key = os.getenv("VLLM_API_KEY", "EMPTY")

your_vllm_options_stable = {
    "seed": 42,
    "top_k": 1,
    "top_p": 1.0,
    "temperature": 0.0,
} 

# ============ Small Model Paths ============
LLMLINGUA_MODEL_PATH=os.getenv("LLMLINGUA_MODEL_PATH", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
EMBEDDING_MODEL_PATH=os.getenv("EMBEDDING_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")

# ============ Data Configuration ============
DATA_PATH=os.getenv("DATA_PATH", os.path.join(os.path.dirname(__file__), "longmemeval_s_cleaned.json"))
QDRANT_DATA_DIR=os.getenv("QDRANT_DATA_DIR", "./qdrant_data")


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response) 
    return prompt

def true_or_false(response):
    if response is None:
        return False
    normalized = str(response).strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = first_line.replace('.', '').replace('!', '').replace(':', '').replace(';', '').split()
    if not tokens:
        return False
    head = tokens[0]
    if head in ("yes", "y"):
        return True
    if head in ("no", "n"):
        return False
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return False

def load_lightmem(collection_name):
    config = {
        "pre_compress": True,
        "pre_compress_streaming": os.getenv("STREAMING_PRECOMPRESS", "0") == "1",
        "autonomous_sleep": os.getenv("AUTONOMOUS_SLEEP", "0") == "1",
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,
                    "device_map": "cuda",
                    "use_llmlingua2": True,
                },
                "compress_config": {
                    "rate": 0.6
                }
            }
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "vllm",
            "configs": {
                "model": your_vllm_model_name,
                "vllm_base_url": your_vllm_base_url,
                "api_key": your_vllm_api_key,
                "max_tokens": 4096,
                "llm_batch_size": int(os.getenv("BATCH_SIZE", "16")),
                "llm_batch_timeout": 10,
            }
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": EMBEDDING_MODEL_PATH,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cuda"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": QDRANT_DATA_DIR,
            }
        },
        "update": "offline",
        "logging": {
            "level": "ERROR",
            "file_enabled": True,
            "log_dir": "logs",
            "log_filename_prefix": "run_vllm",
            "console_enabled": True,
            "file_level": "DEBUG",
        }
    }
    lightmem = LightMemory.from_config(config)
    return lightmem


class VllmModel:
    """
    An example vLLM model class for generating responses during the evaluation.
    """
    def __init__(
            self,
            model_name: str,
            base_url: str,
            api_key: str = "EMPTY",
            options: Optional[Dict] = None,
        ):
        self.name = model_name
        self.options = options
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def call(self, messages: List[Dict[str, str]], **kwargs):
        max_retries = kwargs.get("max_retries", 3)
        
        params = {
            "model": self.name,
            "messages": messages,
        }
        if self.options:
            if "temperature" in self.options: params["temperature"] = self.options["temperature"]
            if "top_p" in self.options: params["top_p"] = self.options["top_p"]
            if "seed" in self.options: params["seed"] = self.options["seed"]
            if "max_tokens" in self.options: params["max_tokens"] = self.options["max_tokens"]

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(**params)
                response = completion.choices[0].message.content
                if not response:
                    response = ""
                print(response)
                return response

            except Exception as e:
                print(f"[Retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise


def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(proj_root, "results_vllm")
    os.makedirs(out_dir, exist_ok=True)

    llm = VllmModel(
        model_name=your_vllm_model_name,
        base_url=your_vllm_base_url,
        api_key=your_vllm_api_key,
        options=your_vllm_options_stable,
    )
    
    # Using the same model as judge for simplicity, could be different
    llm_judge = llm

    data = json.load(open(DATA_PATH, "r")) if DATA_PATH else []

    INIT_RESULT = {
        "add_input_prompt": [],
        "add_output_prompt": [],
        "api_call_nums": 0
    }    
    total_correct = 0
    total_samples = 0

    # Initialize once to stay on GPU
    lightmem = load_lightmem(collection_name="eval_session")

    all_results = []
    for item in tqdm(data, desc="Evaluating", unit="question"):
        question_id = item["question_id"]
        result_filename = os.path.join(out_dir, f"result_{question_id}.json")
        
        # Checkpoint: Skip if already processed
        if os.path.exists(result_filename):
            try:
                with open(result_filename, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                    all_results.append(checkpoint_data)
                    total_correct += checkpoint_data.get("correct", 0)
                    total_samples += 1
                    continue
            except Exception as e:
                print(f"Error loading checkpoint for {question_id}: {e}")

        # Reset state for new question
        lightmem.clear_memory()
        
        # Dynamically update the retriever's collection for this question
        # This ensures we have a clean collection for each question in the shared DB
        lightmem.embedding_retriever.collection_name = question_id
        lightmem.embedding_retriever.create_col(384, on_disk=True)
        
        sessions = item.get("haystack_sessions", [])
        timestamps = item.get("haystack_dates", [])

        results_list = []
        futures = []

        time_start = time.time()
        for session, timestamp in zip(sessions, timestamps):
            while session and session[0]["role"] != "user":
                session.pop(0)
            num_turns = len(session) // 2  
            for turn_idx in range(num_turns):
                turn_messages = session[turn_idx*2 : turn_idx*2 + 2]
                if len(turn_messages) < 2 or turn_messages[0]["role"] != "user" or turn_messages[1]["role"] != "assistant":
                    continue
                for msg in turn_messages:
                    msg["time_stamp"] = timestamp
                is_last_turn = (
                    session is sessions[-1] and turn_idx == num_turns - 1
                )
                result = lightmem.add_memory(
                    messages=turn_messages,
                    user_id=question_id,
                    force_segment=is_last_turn,
                    force_extract=is_last_turn,
                )
                if result != INIT_RESULT:
                    if "extraction_future" in result:
                        futures.append(result["extraction_future"])
                    else:
                        results_list.append(result)

        # Wait for all async extractions to complete before retrieving
        for fut in futures:
            try:
                results_list.append(fut.result())
            except Exception as e:
                print(f"Error extracting memory: {e}")

        time_end = time.time()
        construction_time = time_end - time_start

        related_memories = lightmem.retrieve(item["question"], user_id=question_id, limit=20)
        messages = []
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({
            "role": "user",
            "content": f"Question time:{item.get('question_date', '')} and question:{item['question']}\nPlease answer the question based on the following memories: {str(related_memories)}"
        })
        generated_answer = llm.call(messages)

        if 'abs' in question_id:
            prompt = get_anscheck_prompt(
                item["question_type"], item["question"], item.get("answer", ""), generated_answer, abstention=True
            )
        else:
            prompt = get_anscheck_prompt(
                item["question_type"], item["question"], item.get("answer", ""), generated_answer
            )
        messages = [{"role": "user", "content": prompt}]
        response = llm_judge.call(messages)

        correct = 1 if true_or_false(response) else 0
        total_correct += correct
        total_samples += 1

        save_data = {
            "question_id": question_id,
            "results": results_list,
            "construction_time": construction_time,
            "generated_answer": generated_answer,
            "ground_truth": item.get("answer", ""),
            "correct": correct,
        }
        
        # Save individual result for checkpointing
        with open(result_filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
            
        all_results.append(save_data)

    # Save all results to a single file at the end
    final_report_path = os.path.join(out_dir, "final_evaluation_report.json")
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    # Shutdown background threads at the very end
    lightmem.stop()

    if total_samples > 0:
        accuracy = (total_correct / total_samples) * 100
        print("\n" + "="*40)
        print("FINAL ACCURACY RESULTS")
        print("="*40)
        print(f"Total Questions Processed: {total_samples}")
        print(f"Correct Answers:          {total_correct}")
        print(f"Overall Accuracy:         {accuracy:.2f}%")
        print(f"Detailed logs saved in:   {out_dir}")
        print("="*40)


if __name__ == "__main__":
    main()
