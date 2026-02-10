import langextract as lx
import textwrap
from collections import Counter, defaultdict

# Define comprehensive prompt and examples for complex literary text
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships from the given text.

    Provide meaningful attributes for every entity to add context and depth.

    Important: Use exact text from the input for extraction_text. Do not paraphrase.
    Extract entities in order of appearance with no overlapping text spans.

    Note: In play scripts, speaker names appear in ALL-CAPS followed by a period.""")

examples = [
    lx.data.ExampleData(
        text=textwrap.dedent("""\
            ROMEO. But soft! What light through yonder window breaks?
            It is the east, and Juliet is the sun.
            JULIET. O Romeo, Romeo! Wherefore art thou Romeo?"""),
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe", "character": "Romeo"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor", "character_1": "Romeo", "character_2": "Juliet"}
            ),
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="JULIET",
                attributes={"emotional_state": "yearning"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="Wherefore art thou Romeo?",
                attributes={"feeling": "longing question", "character": "Juliet"}
            ),
        ]
    )
]

# Process Romeo & Juliet directly from Project Gutenberg
print("Downloading and processing Romeo and Juliet from Project Gutenberg...")

input_text = "By the time the fog lifted from the river, Elias had already decided to leave. The town revealed itself reluctantly, brick by brick, like a memory that didn’t want to be recalled. Windows glowed with a tired yellow light, and somewhere a radio played a song that felt older than the language it was sung in. He stood on the bridge and watched the water drag reflections downstream, convinced that if he stared long enough, something of himself would follow. His father used to say that places remember you longer than people do. Elias had laughed at that once. Now the streets whispered his name with every step, and the doors he passed seemed to lean inward, listening. The bakery still smelled of burnt sugar. The clock tower still ran five minutes slow. Nothing had changed, which was the most unsettling thing of all. He reached into his coat pocket and felt the folded letter, soft at the edges from too much handling. Leaving was supposed to feel decisive, heroic even. Instead, it felt like misplacing something important and only realizing it once it was already gone."

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemma3:1b",  # Automatically selects Ollama provider
    model_url="http://localhost:11434",
    fence_output=False,
    use_schema_constraints=False,
    extraction_passes=3,      # Multiple passes for improved recall
    max_workers=20,           # Parallel processing for speed
    max_char_buffer=1000 
)   

print(f"Extracted {len(result.extractions)} entities from {len(result.text):,} characters")

# Save and visualize the results
lx.io.save_annotated_documents([result], output_name="romeo_juliet_extractions.jsonl", output_dir=".")

# Generate the interactive visualization
html_content = lx.visualize("romeo_juliet_extractions.jsonl")
with open("romeo_juliet_visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)

print("Interactive visualization saved to romeo_juliet_visualization.html")