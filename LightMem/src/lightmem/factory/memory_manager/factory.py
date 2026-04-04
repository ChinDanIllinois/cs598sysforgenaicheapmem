import logging
from typing import Dict, Optional
from importlib import import_module
from lightmem.configs.memory_manager.base import MemoryManagerConfig
from .batch_processor import GeminiBatchProcessor, VllmBatchProcessor


class MemoryManagerFactory:
    _MODEL_MAPPING: Dict[str, str] = {
        "deepseek": "lightmem.factory.memory_manager.deepseek.DeepseekManager",
        "openai": "lightmem.factory.memory_manager.openai.OpenaiManager",
        "transformers": "lightmem.factory.memory_manager.transformers.TransformersManager",
        "ollama": "lightmem.factory.memory_manager.ollama.OllamaManager",
        "vllm": "lightmem.factory.memory_manager.vllm.VllmManager",
        "vllm_offline": "lightmem.factory.memory_manager.vllm_offline.VllmOfflineManager",
        "gemini": "lightmem.factory.memory_manager.gemini.GeminiManager",
        "mock": "lightmem.factory.memory_manager.mock.MockManager",
    }

    @classmethod
    def from_config(cls, config: MemoryManagerConfig):
        """
        Instantiate a compressor by dynamically importing the class based on config.
        
        Args:
            config: PreCompressorConfig containing model name and specific configs
            
        Returns:
            An instance of the requested compressor model
            
        Raises:
            ValueError: If model name is not supported or instantiation fails
            ImportError: If the module or class cannot be imported
        """
        model_name = config.model_name
        
        if model_name not in cls._MODEL_MAPPING:
            raise ValueError(
                f"Unsupported manager model: {model_name}. "
                f"Supported models are: {list(cls._MODEL_MAPPING.keys())}"
            )

        class_path = cls._MODEL_MAPPING[model_name]
        
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = import_module(module_path)
            manager_class = getattr(module, class_name)
            
            manager_kwargs = {}
            if config.configs is not None:
                manager_kwargs["config"] = config.configs
            
            # Instantiate the manager
            manager = manager_class(**manager_kwargs)
            
            # Setup batching if enabled
            if config.configs and (config.configs.llm_batch_size > 1 or getattr(config.configs, "vllm_adaptive_shaping", False)):
                logger = logging.getLogger("LightMem")
                batch_manager = None
                
                if model_name == "vllm":
                    batch_manager = VllmBatchProcessor(
                        base_url=config.configs.vllm_base_url or "http://localhost:8000/v1",
                        model=config.configs.model,
                        batch_size=config.configs.llm_batch_size,
                        timeout=config.configs.llm_batch_timeout,
                        api_key=config.configs.api_key or "EMPTY",
                        logger=logger,
                        adaptive_shaping=getattr(config.configs, "vllm_adaptive_shaping", False),
                        metrics_url=getattr(config.configs, "vllm_metrics_url", None)
                    )
                elif model_name == "gemini":
                    # Gemini needs the client from the manager
                    if hasattr(manager, "client"):
                        batch_manager = GeminiBatchProcessor(
                            client=manager.client,
                            model=config.configs.model,
                            batch_size=config.configs.llm_batch_size,
                            timeout=config.configs.llm_batch_timeout,
                            poll_interval=5, # Default poll interval
                            api_key=config.configs.api_key,
                            logger=logger
                        )
                
                if batch_manager:
                    manager.batch_manager = batch_manager
                    logger.info(f"Enabled batching for {model_name} with batch_size={config.configs.llm_batch_size}")

            return manager
            
        except ImportError as e:
            raise ImportError(
                f"Could not import manager'{class_path}': {str(e)}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Maybe class '{class_name}' not found in module '{module_path}': {str(e)}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate {model_name} manager: {str(e)}"
            ) from e
