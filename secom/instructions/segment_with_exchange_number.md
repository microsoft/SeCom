# Instruction

## Context

- **Goal**: Your task is to segment a multi-turn conversation between a user and a chatbot into topically coherent units based on semantics. Successive user-bot exchanges with the same topic should be grouped into the same segmentation unit, and new segmentation units should be created when a topic shift occurs.

- **Data**: The input data is a series of user-bot exchanges separated by "\n\n". Each exchange consists of a single-turn conversation between the user and the chatbot, started with "[Exchange (Exchange Number)]: ".

## Requirements

### Output Format

- Output the segmentation results in **jsonl lines file** format. Each dictionary represents a segment, consisting of one or more user-bot exchanges on the same topic. Each dictionary should include the following keys:
    - **segment_id**: The index of this segment, starting from 0.
    - **start_exchange_number**: The number of the **first** user-bot exchange in this segment.
    - **end_exchange_number**: The number of the **last** user-bot exchange in this segment.
    - **num_exchanges**: An integer indicating the number of user-bot exchanges in this segment, calculated as **end_exchange_number** - **start_exchange_number** + 1.
    - **summary**: A brief summary (within 100 words) of this segment. The summary should be straightforward, without unnecessary prefixes such as "The conversation starts with", "The discussion shifts back to", or "The topic shifts to".

Here is an example of the expected output:
```
<segmentation>
{{"segment_id": 0, "start_exchange_number": 0, "end_exchange_number": 5, "num_exchanges": 6, "summary": "A brief summary of this segment."}}
{{"segment_id": 1, "start_exchange_number": 6, "end_exchange_number": 8, "num_exchanges": 3, "summary": "A brief summary of this segment."}}
...
</segmentation>
```

# Data

{text_to_be_segmented}

# Question

## Please generate the segmentation result from the input data that meets the following requirements:

- **No Missing Exchanges**: Ensure that the exchange numbers cover all exchanges in the given conversation without omission. The **start_exchange_number** of the next segment should be equal to **end_exchange_number** + 1 of the previous segment. The **start_exchange_number** of the first segment should be 0, and the **end_exchange_number** of the last segment should equal the last exchange number of the given input.
- **No Overlapping Exchanges**: Ensure that successive segments have no overlap in exchanges. The **start_exchange_number** of the next segment should be equal to **end_exchange_number** + 1 of the previous segment.
- **Concise, Complete and Straightforward Summaries**: The summaries should be concise, not exceeding 100 words, while retaining all key information from the segment. The summaries should be straightforward, without unnecessary prefixes such as "The conversation starts with ", "The discussion shifts back to " or "The topic shifts to ...".
- **Accurate Counting**: The sum of **num_exchanges** across all segments should equal the total number of user-bot exchanges in the input.
- Provide your segmentation result between the tags: <segmentation></segmentation>.

# Output

Now, provide the segmentation result based on the instructions above.
