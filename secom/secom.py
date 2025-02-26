# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import os
import traceback
from typing import List

import tiktoken
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from llmlingua import PromptCompressor
from omegaconf import OmegaConf

from .utils import OpenAILLM, extract_result, extract_yes_no


class SeCom:
    """
    SeCom, a system that constructs memory bank at segment level by introducing a conversation SEgmentation model, while applying COMpression based denoising on memory units to enhance memory retrieval.

    This class initializes with the segmentor, compressor and retriever according to its configuration in config.yaml,
        preparing it for conversation memory management.
    The SeCom class is versatile and can be adapted for various models and specific requirements in memory management.
    Users can specify different model names and configurations as needed for their particular use case.
    The architecture is based on the paper
        "SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents".

    Args:
        granularity (str, optional): The granularity to construct memory bank and perform retrieval. Default is "segment".
        config_path (str, optional): The path to load config. Default config is "configs/mpnet.yaml"
    Example:
        >>> memory_manager = SeCom(granularity="segment", config_path="configs/mpnet.yaml")
        >>> conversation_history = [["First session of a very looooooong conversation history", "The second user-bot turn of the first session"], ["Second Session ..."]]
        >>> requests = ["A question regarding the conversation history", "Another question"]
        >>> result = memory_manager.get_memory(request, conversation_history, compress_rate=0.9, retrieve_topk=3)
        >>> print(result["retrieved_texts"])
        # This will print a list of retrieved text that is relevant to the requests, with each string corresponds to a request.

    """

    def __init__(
        self,
        granularity: str = "segment",
        config_path: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "configs/mpnet.yaml"
        ),
    ):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.granularity = granularity
        self.config_path = config_path
        self.config = OmegaConf.load(self.config_path)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        self.segments = []
        self.memory_bank = []

        self.segmentor = None
        if "segmentor" in self.config:
            self.init_segmentor(**self.config.segmentor)
        self.compressor = None
        if "compressor" in self.config:
            self.init_compressor(**self.config.compressor)

    def clear(
        self,
    ):
        self.memory_bank.clear()

    def get_memory(
        self,
        requests: List[str],
        conversation_history: List[List[str]] = [],
        compress_rate: float = 0.9,
        retrieve_topk: int = 3,
    ):
        """
        Get requests relevant memory from conversation history.

        Args:
            requests (List[str]): List of request strings that may need to reference the conversation history to generate responses.
            conversation_history (List[List[str]], optional): List of sessions that consists of multiple user-bot interaction turns, use to construct memory bank.
                Default is empty, using previous memory bank.
            new_turn（str): New user-bot interaction turn to be added to the memroy bank.
            compress_rate (float, optional): The maximum compression rate when performing denoising. Default is 0.9.
            retrieve_topk (float, optional): The maximum memory unit to be retrieved. Default is 3.

        Returns:
            dict: A dictionary containing:
                - "retrieved_texts" (List[str]): List of the retrieved memory units.
                - "n_retrieved_turn" (float): The average number of retrieved turns.
                - "n_retrieved_token" (float): The average number of retrieved tokens (computed by gpt-4 tokenizer).
        """
        if isinstance(requests, str):
            requests = [requests]
        if conversation_history:
            self.build_memory(conversation_history, compress_rate=compress_rate)
            self.init_retriever(retrieve_topk, **self.config.retriever)
        else:
            assert len(self.memory_bank) > 0, "pass in conversation_history first"
            self.update_retriever(retrieve_topk)
        retrieved_texts, n_turn, n_token = self.retrieve(requests)

        return_dict = {
            "retrieved_texts": retrieved_texts,
            "n_retrieved_turn": n_turn,
            "n_retrieved_token": n_token,
        }

        return return_dict

    def update_memory(
        self,
        new_turn: str,
    ):
        """
        Add the newly evolved user-bot interaction turn to the memory bank.

        Args:
            new_turn（str): New user-bot interaction turn to be added to the memroy bank.

        Returns:
            update the built-in memory bank, return nothing.
        """
        assert len(self.memory_bank) > 0, "pass in conversation_history first"
        replace_last = False
        if self.granularity == "segment":
            replace_last = self.update_segment(new_turn)
        elif self.granularity == "session":
            assert len(self.memory_bank) > 0
            self.memory_bank[-1].page_content += f"\n{new_turn}"
            self.memory_bank[-1].metadata["content"].append(new_turn)
            replace_last = True
        elif self.granularity == "turn":
            self.memory_bank.append(
                Document(page_content=new_turn, metadata={"content": new_turn}),
                ids=[len(self.memory_bank) - 1],
            )
        self.update_database(replace_last)

    def build_memory(
        self,
        conversation_history: List[List[str]],
        compress_rate: float = 0.9,
    ):
        """
        Build the memory bank from long-term conversation history.

        Args:
            conversation_history (List[List[str]], optional): List of sessions that consists of multiple user-bot interaction turns, use to construct memory bank.
                Default is empty, using previous memory bank.
            compress_rate (float, optional): The maximum compression rate when performing denoising. Default is 0.9.

        Returns:
            build the built-in memory bank, return nothing.
        """
        if self.granularity == "segment":
            segments = self.segment(conversation_history)
            units = segments
        elif self.granularity == "session":
            units = conversation_history
        elif self.granularity == "turn":
            units = []
            for session in conversation_history:
                for turn in session:
                    units.append([turn])
        if compress_rate < 1:
            comp_units = self.compress(units, compress_rate)
            self.memory_bank = [
                Document(
                    page_content="\n".join(comp_unit)
                    if isinstance(comp_unit, list)
                    else comp_unit,
                    metadata={"content": unit, "idx": idx},
                )
                for idx, (unit, comp_unit) in enumerate(zip(units, comp_units))
            ]
        else:
            self.memory_bank = [
                Document(
                    page_content="\n".join(unit) if isinstance(unit, list) else unit,
                    metadata={"content": unit, "idx": idx},
                )
                for idx, unit in enumerate(units)
            ]

    def init_segmentor(self, segment_model, prompt_path, incremental_prompt_path):
        self.segment_model = segment_model
        self.segmentor = OpenAILLM(segment_model)
        with open(os.path.join(self.root_dir, prompt_path), "r", encoding="utf-8") as f:
            self.segment_prompt = f.read()
        with open(
            os.path.join(self.root_dir, incremental_prompt_path), "r", encoding="utf-8"
        ) as f:
            self.incremental_segment_prompt = f.read()

    def init_compressor(self, compress_model):
        self.compressor = PromptCompressor(compress_model, use_llmlingua2=True)

    def init_retriever(self, topk, storage, embedding_model="", device_map="cuda"):
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.vectorstores import FAISS, Chroma

        self.embedding_model = embedding_model
        if embedding_model:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model, model_kwargs={"device": device_map}
            )
            self.vector_store = (locals()[storage]).from_documents(
                self.memory_bank,
                self.embeddings,
                ids=[i for i in range(len(self.memory_bank))],
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": topk})
        else:
            self.retriever = (locals()[storage]).from_documents(
                self.memory_bank, k=topk
            )

    def update_database(self, replace_last=False):
        if self.embedding_model:
            if replace_last:
                self.vector_store.delete(ids=[len(self.memory_bank) - 1])
            self.vector_store.add_documents(
                [self.memory_bank[-1]], ids=[len(self.memory_bank) - 1]
            )

    def update_retriever(self, topk):
        if self.embedding_model:
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": topk})
        else:
            self.retriever = self.retriever.from_documents(self.memory_bank, k=topk)

    def prefix_exchanges_with_idx(self, exchanges):
        exchanges_str_with_idx = ""
        for i, exchange in enumerate(exchanges):
            exchanges_str_with_idx += f"[Exchange {i}]: {exchange}\n\n"
        return exchanges_str_with_idx

    def segment(
        self,
        sessions,
    ):
        segments = []
        for session_idx, exchanges in enumerate(sessions):
            exchanges_str_with_idx = self.prefix_exchanges_with_idx(exchanges)

            prompt = self.segment_prompt.format(
                text_to_be_segmented=exchanges_str_with_idx
            )
            response = self.segmentor(prompt, max_tokens=1024)

            seg_jsonl, extract_success = extract_result(response, "segmentation")
            if not extract_success:
                success = False
                print(f"bad response: {response}")
            else:
                success = True
                lines = seg_jsonl.strip().split("\n")
                num_exchanges = []
                segmentations = []
                prev_idx = 0
                for line in lines:
                    try:
                        line_dict = json.loads(line.strip().strip(","))
                        n_ex = int(line_dict["num_exchanges"])
                        num_exchanges.append(n_ex)
                        segmentations.append(exchanges[prev_idx : prev_idx + n_ex])
                        prev_idx = prev_idx + n_ex
                    except Exception:
                        print(traceback.format_exc())
                        success = False
                        break
            if success:
                print(
                    f"{session_idx}-th session is segmented to {len(segmentations)} segments"
                )
                segments.extend(segmentations)
            else:
                print(f"{session_idx}-th session not segmented")
                for i in range(0, len(exchanges), 3):
                    segments.append(exchanges[i : i + 3])

        return segments

    def update_segment(
        self,
        new_turn: str,
    ):
        assert len(self.memory_bank) > 0, "empty memory bank"
        last_segment_text = self.memory_bank[-1].page_content
        prompt = self.incremental_segment_prompt.format(
            new_turn=new_turn, prev_session=last_segment_text
        )
        response = self.segmentor(prompt, max_tokens=1024)
        include = extract_yes_no(response)
        if include:
            self.memory_bank[-1].page_content += f"\n{new_turn}"
            self.memory_bank[-1].metadata["content"].append(new_turn)
            replace_last = True
        else:
            self.memory_bank.append(
                Document(
                    page_content=new_turn,
                    metadata={"content": [new_turn], "idx": len(self.memory_bank)},
                )
            )
            replace_last = False
        return replace_last

    def compress(
        self,
        segments,
        compress_rate=0.9,
    ):
        if compress_rate < 1.0 and not self.compressor:
            print("Compressor not initialized, reload compress_rate to 1.0")
            return segments
        comp_segments = []
        for segment in segments:
            comp_segment = self.compressor.compress_prompt(
                segment,
                rate=compress_rate,
                use_context_level_filter=False,
                force_tokens=["\n", ".", "[human]", "[bot]"],
            )["compressed_prompt_list"]
            comp_segments.append(comp_segment)

        return comp_segments

    def retrieve(
        self,
        requests,
    ):
        retrieved_texts = []
        retrieved_n_exs = []
        retrieved_n_tokens = []
        n_exchange = 0
        n_token = 0
        for q in requests:
            text_list = []
            r_docs = self.retriever.invoke(q)
            for doc in r_docs:
                if isinstance(doc.metadata["content"], list):
                    n_exchange += len(doc.metadata["content"])
                text = (
                    "\n".join(doc.metadata["content"])
                    if isinstance(doc.metadata["content"], list)
                    else doc.metadata["content"]
                )
                text_list.append(text)
                n_token += len(self.tokenizer.encode(text))

            retrieved_texts.append("\n\n".join(text_list))
            retrieved_n_exs.append(n_exchange)
            retrieved_n_tokens.append(n_token)

        n_exchange /= len(requests)
        n_token /= len(requests)
        return retrieved_texts, n_exchange, n_token

    def init_retriever_external_memory(
        self, memory_bank, topk, storage, embedding_model="", device_map="cuda"
    ):
        """
        Only for step-by-step experiment.

        """
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.vectorstores import FAISS, Chroma

        self.embedding_model = embedding_model
        if embedding_model:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model, model_kwargs={"device": device_map}
            )
            self.vector_store = (locals()[storage]).from_documents(
                memory_bank,
                self.embeddings,
                ids=[i for i in range(len(memory_bank))],
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": topk})
        else:
            self.retriever = (locals()[storage]).from_documents(memory_bank, k=topk)

    def retrieve_external_memory(
        self,
        requests,
        memory_units,
        comp_memory_units=None,
        retrieve_topk=3,
    ):
        """
        Only for step-by-step experiment.

        """
        if comp_memory_units:
            memory_bank = [
                Document(
                    page_content="\n".join(comp_unit)
                    if isinstance(comp_unit, list)
                    else comp_unit,
                    metadata={"content": unit, "idx": idx},
                )
                for idx, (unit, comp_unit) in enumerate(
                    zip(memory_units, comp_memory_units)
                )
            ]
        else:
            memory_bank = [
                Document(
                    page_content="\n".join(unit) if isinstance(unit, list) else unit,
                    metadata={"content": unit, "idx": idx},
                )
                for idx, unit in enumerate(memory_units)
            ]
        self.init_retriever_external_memory(
            memory_bank, topk=retrieve_topk, **self.config.retriever
        )

        retrieved_texts = []
        retrieved_n_exs = []
        retrieved_n_tokens = []
        n_exchange = 0
        n_token = 0
        for q in requests:
            text_list = []
            r_docs = self.retriever.invoke(q)
            for doc in r_docs:
                if isinstance(doc.metadata["content"], list):
                    n_exchange += len(doc.metadata["content"])
                text = (
                    "\n".join(doc.metadata["content"])
                    if isinstance(doc.metadata["content"], list)
                    else doc.metadata["content"]
                )
                text_list.append(text)
                n_token += len(self.tokenizer.encode(text))

            retrieved_texts.append("\n\n".join(text_list))
            retrieved_n_exs.append(n_exchange)
            retrieved_n_tokens.append(n_token)

        n_exchange /= len(requests)
        n_token /= len(requests)
        return retrieved_texts, n_exchange, n_token
