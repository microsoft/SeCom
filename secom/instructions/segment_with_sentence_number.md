# Instruction

## Context

- **Goal**: Your task is to segment a multi-turn conversation between a user and an assistant into topically coherent units based on semantics. Group successive sentences on the same topic into one segment and create new segments when there is a topic shift.

- **Data**: The input data is a series of sentences separated by "\n\n". Each sentence begins with "[Sentence (Sentence Number)]: " followed by the utterance from either the user or the assistant.

## Requirements

### Output Format

- Output the segmentation results in **JSONL (JSON Lines)** format. Each dictionary represents a segment, consisting of one or more sentences on the same topic. Each dictionary should include the following keys:
    - **segment_id**: The index of this segment, starting from 0.
    - **start_sentence_number**: The number of the **first** sentence in this segment.
    - **end_sentence_number**: The number of the **last** sentence in this segment.
    - **num_sentences**: An integer indicating the number of sentences in this segment, calculated as **end_sentence_number** - **start_sentence_number** + 1.
    - **summary**: A brief summary (within 100 words) of this segment. The summary should be straightforward, without unnecessary prefixes such as "The conversation starts with", "The discussion shifts back to", or "The topic shifts to".

Here is an example of the expected output:
```
<segmentation>
{{"segment_id": 0, "start_sentence_number": 0, "end_sentence_number": 5, "num_sentences": 6, "summary": "A brief summary of this segment."}}
{{"segment_id": 1, "start_sentence_number": 6, "end_sentence_number": 8, "num_sentences": 3, "summary": "A brief summary of this segment."}}
...
</segmentation>
```

# Data

{text_to_be_segmented}

# Question

## Please generate the segmentation result from the input data that meets the following requirements:

- **No Missing Sentences**: Ensure that the sentence numbers cover all sentences in the given conversation without omission. The **start_sentence_number** of the next segment should be equal to **end_sentence_number** + 1 of the previous segment. The **start_sentence_number** of the first segment should be 0, and the **end_sentence_number** of the last segment should equal the last sentence number of the given input.
- **No Overlapping Sentences**: Ensure that successive segments have no overlap in sentences. The **start_sentence_number** of the next segment should be equal to **end_sentence_number** + 1 of the previous segment.
- **Concise, Complete and Straightforward Summaries**: The summaries should be concise, not exceeding 100 words, while retaining all key information from the segment. The summaries should be straightforward, without unnecessary prefixes such as "The conversation starts with ", "The discussion shifts back to " or "The topic shifts to ...".
- **Accurate Counting**: The sum of **num_sentences** across all segments should equal the total number of user-bot sentences in the input.
- Provide your segmentation result between the tags: <segmentation></segmentation>.

# Output

Now, provide the segmentation result based on the instructions above.
