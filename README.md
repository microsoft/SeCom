<div align="center">
<h1>🧠 SeCom: On Memory Construction and Retrieval for <br>Personalized Conversational Agents</h1>
<h4>
<a href="https://www.arxiv.org/abs/2502.05589">📄 Paper (ICLR 2025)</a> &nbsp; 
<a href="https://llmlingua.com/secom.html">🌐 Project Page</a> &nbsp; 
</h4>

</div>

## Key Takeaways
💡 **Memory granularity matters**: Turn-level, session-level & summarization-based memory struggle with retrieval accuracy and the semantic integrity or relevance of the context.

💡 **Prompt compression methods** (e.g., [LLMLingua-2](https://llmlingua.com/llmlingua2.html)) **can denoise memory retrieval**, boosting both **retrieval accuracy** and **response quality.**

✅ **SeCom** – an approach that **segments conversations topically** for memory construction and performs memory retrieval based on compressed memory units.

📊 **Result** – superior performance on long-term conversation benchmarks such as LOCOMO and Long-MT-Bench+!

## Install

```bash
pip install llmlingua
pip install -e .
```

SeCom uses [dot_env](https://github.com/theskumar/python-dotenv) to manage the API_KEY.

```bash
pip install python-dotenv
```

Specifiy your OPENAI_API_KEY and OPENAI_API_BASE in `~/dot_env/openai.env`

```
OPENAI_API_KEY=""
OPENAI_API_BASE=""
```

## Usage

```python

from secom import SeCom

memory_manager = SeCom(granularity="segment")

conversation_history = [
    [
        "First session of a very looooooong conversation history",
        "The second user-bot turn of the first session",
    ],
    ["Second Session ..."],
]
requests = ["A question regarding the conversation history", "Another question"]
result = memory_manager.get_memory(
    requests, conversation_history, compress_rate=0.9, retrieve_topk=1
)
print(result["retrieved_texts"])
# >>>
```

For more examples, see "example/" and "experiment/".

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
