# from typing import Dict, Optional, List, Union, Any
# from transformers import PreTrainedTokenizerBase

# from lightmem.configs.pre_compressor.llmlingua_2 import LlmLingua2Config


# class LlmLingua2Compressor:
#     def __init__(self, config: Optional[LlmLingua2Config] = None):
#         self.config = config

#         try:
#             import importlib
#             importlib.import_module('llmlingua')
#         except ImportError:
#             raise ImportError(
#                 "Required package 'llmlingua' not found. "
#                 "Please install it with: pip install llmlingua\n"
#                 "Or for the latest version: pip install git+https://github.com/microsoft/LLMLingua.git"
#             )

#         try:
#             from llmlingua import PromptCompressor
#             if config.llmlingua_config['use_llmlingua2'] is True:
#                 self._compressor = PromptCompressor(
#                     model_name=config.llmlingua_config['model_name'],
#                     device_map=config.llmlingua_config['device_map'],
#                     use_llmlingua2=config.llmlingua_config['use_llmlingua2'],
#                     llmlingua2_config=config.llmlingua2_config
#                 )
#             else:
#                 self._compressor = PromptCompressor(
#                     model_name=config.llmlingua_config['model_name'],
#                     device_map=config.llmlingua_config['device_map']
#                 )
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize LlmLingua2Compressor: {str(e)}")

#     def compress(
#         self,
#         messages: List[Dict[str, str]],
#         tokenizer: Union[PreTrainedTokenizerBase, Any, None],
#     ) -> List[Dict[str, str]]:
#         # TODO: Consider adding an extra field in the message, compressed_content, and put the compressed content in this field while keeping content unchanged.
#         """
#         Compress the content of each message.

#         Args:
#             messages: List of message dicts containing 'role' and 'content'.
#             tokenizer: Tokenizer to check token length after compression.

#         Returns:
#             List of messages with compressed content.
#         """
#         for mes in messages:
#             content = mes.get('content', '')
#             if not content or not content.strip():
#                 # If content is empty, it doesn't need compression
#                 continue

#             compress_config = {
#                 'context': [content],
#                 **self.config.compress_config
#             }

#             try:
#                 comp_content = self._compressor.compress_prompt(**compress_config)['compressed_prompt']
#             except Exception as e:
#                 print(f"compress error, skip this message: {e}")
#                 comp_content = content  # Keep the original content if compression fails

#             # Check if the compressed content is still too long
#             if tokenizer is not None:
#                 try:
#                     while len(tokenizer.encode(comp_content)) >= 512 and comp_content.strip():
#                         new_compress_config = {
#                             'context': comp_content,
#                             **self.config.compress_config
#                         }
#                         comp_content = self._compressor.compress_prompt(**new_compress_config)['compressed_prompt']
#                 except Exception as e:
#                     print(f"secondary compress error: {e}")
#                     # If an error occurs, exit the loop and keep the current compression result
#                     break

#             # Update message
#             if comp_content.strip():
#                 mes['content'] = comp_content.strip()

#         return messages

#     @property
#     def inner_compressor(self):
#         """
#         Access the underlying PromptCompressor instance directly.
#         """
#         return self._compressor

from typing import Dict, Optional, List, Union, Any
from transformers import PreTrainedTokenizerBase
import copy

from lightmem.configs.pre_compressor.llmlingua_2 import LlmLingua2Config


class LlmLingua2Compressor:
    def __init__(self, config: Optional[LlmLingua2Config] = None):
        self.config = config

        try:
            import importlib
            importlib.import_module("llmlingua")
        except ImportError:
            raise ImportError(
                "Required package 'llmlingua' not found. "
                "Please install it with: pip install llmlingua\n"
                "Or for the latest version: pip install git+https://github.com/microsoft/LLMLingua.git"
            )

        self._lock = threading.Lock()
        try:
            from llmlingua import PromptCompressor
            if config.llmlingua_config["use_llmlingua2"] is True:
                self._compressor = PromptCompressor(
                    model_name=config.llmlingua_config["model_name"],
                    device_map=config.llmlingua_config["device_map"],
                    use_llmlingua2=config.llmlingua_config["use_llmlingua2"],
                    llmlingua2_config=config.llmlingua2_config,
                )
            else:
                self._compressor = PromptCompressor(
                    model_name=config.llmlingua_config["model_name"],
                    device_map=config.llmlingua_config["device_map"],
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LlmLingua2Compressor: {str(e)}")

    def compress_batch(
        self,
        batch_of_messages: List[List[Dict[str, str]]],
        tokenizer: Union[PreTrainedTokenizerBase, Any, None] = None,
    ) -> List[List[Dict[str, str]]]:
        """
        Compress a batch of message lists (cross-tenant batching).
        """
        # Flatten all messages to compress them in one GPU call if possible
        all_contents = []
        mapping = [] # (batch_idx, msg_idx)
        
        for b_idx, messages in enumerate(batch_of_messages):
            for m_idx, mes in enumerate(messages):
                content = mes.get('content', '').strip()
                if content:
                    # Truncate to avoid model hangs on long inputs
                    # LLMLingua-2 usually has a 512 context limit
                    if len(content) > 2000: # Heuristic for ~500 tokens
                        content = content[:2000]
                    all_contents.append(content)
                    mapping.append((b_idx, m_idx))

        if not all_contents:
            return batch_of_messages

        # Run compression in one big batch
        # Safety: Ensure total tokens doesn't cause a hang if the model is picky
        compress_config = {
            'context': all_contents,
            **self.config.compress_config
        }
        
        try:
            # LLMLingua-2 PromptCompressor.compress_prompt handles lists in 'context'
            print("DEBUG: Entering LLMLingua-2 model call...")
            with self._lock:
                results = self._compressor.compress_prompt(**compress_config)
            print("DEBUG: Exited LLMLingua-2 model call successfully.")
            compressed_prompts = results['compressed_prompt']
            
            # If it's a single string (only 1 message total), wrap it in a list
            if isinstance(compressed_prompts, str):
                compressed_prompts = [compressed_prompts]

            # Map results back to original messages
            for i, (b_idx, m_idx) in enumerate(mapping):
                if i < len(compressed_prompts):
                    batch_of_messages[b_idx][m_idx]['content'] = compressed_prompts[i].strip()

        except Exception as e:
            print(f"Batch compression error: {e}")
            # Fallback is to return original (partially updated or not)

        return batch_of_messages

    def compress(
        self,
        messages: List[Dict[str, str]],
        tokenizer: Union[PreTrainedTokenizerBase, Any, None],
    ) -> List[Dict[str, str]]:
        """
        Compress the content of each message.

        Returns a deep-copied compressed list and does not mutate the input.
        """
        compressed_messages = copy.deepcopy(messages)

        for mes in compressed_messages:
            content = mes.get("content", "")
            if not content or not content.strip():
                continue

            compress_config = {
                "context": [content],
                **self.config.compress_config,
            }

            try:
                comp_content = self._compressor.compress_prompt(**compress_config)["compressed_prompt"]
            except Exception as e:
                print(f"compress error, skip this message: {e}")
                comp_content = content

            if tokenizer is not None:
                try:
                    while len(tokenizer.encode(comp_content)) >= 512 and comp_content.strip():
                        new_compress_config = {
                            "context": [comp_content],
                            **self.config.compress_config,
                        }
                        comp_content = self._compressor.compress_prompt(**new_compress_config)[
                            "compressed_prompt"
                        ]
                except Exception as e:
                    print(f"secondary compress error: {e}")
                    pass

            if comp_content.strip():
                mes["content"] = comp_content.strip()

        return compressed_messages

    def compress_with_stats(
        self,
        messages: List[Dict[str, str]],
        tokenizer: Union[PreTrainedTokenizerBase, Any, None],
    ):
        raw_chars = sum(len(m.get("content", "")) for m in messages)
        compressed_messages = self.compress(messages, tokenizer)
        compressed_chars = sum(len(m.get("content", "")) for m in compressed_messages)

        stats = {
            "num_messages": len(messages),
            "raw_chars": raw_chars,
            "compressed_chars": compressed_chars,
            "compression_ratio": (compressed_chars / raw_chars if raw_chars > 0 else 1.0),
        }
        return compressed_messages, stats

    @property
    def inner_compressor(self):
        return self._compressor