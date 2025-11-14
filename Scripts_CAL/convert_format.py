import json
import re
from pathlib import Path
from tqdm import tqdm

# ====================================================================
# Configuration
# ====================================================================

INPUT_DIRS = [
    "./DA",
    "./DB"
]

OUTPUT_SUFFIX = "_converted"  # Will create DA_converted, DB_converted

# ====================================================================
# Conversion Functions
# ====================================================================

def convert_response_format(old_response: str) -> str:
    """
    Convert from old format to new format:

    Old format:
        Reasoning: ...
        Reflection: YES/NO
        Answer: ...

    New format:
        <think>
        [Reasoning content only, NO Reflection]
        </think>

        \\boxed{[Answer]}
    """
    # Extract components using regex
    reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=Reflection:|Answer:|$)', old_response, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'Answer:\s*(.*?)$', old_response, re.DOTALL | re.IGNORECASE)

    # Extract text
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    # Build new format (without Reflection)
    new_response = f"""<think>
{reasoning}
</think>

\\boxed{{{answer}}}"""

    return new_response

def convert_file(input_path: str, output_path: str):
    """Convert a JSONL file from old format to new format."""
    print(f"Converting: {input_path} -> {output_path}")

    converted_count = 0
    total_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Processing"):
            if not line.strip():
                continue

            total_count += 1

            try:
                # Parse JSON
                data = json.loads(line)

                # Convert response
                old_response = data.get('response', '')
                new_response = convert_response_format(old_response)

                # Update data
                data['response'] = new_response

                # Write to output
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                converted_count += 1

            except Exception as e:
                print(f"Error processing line {total_count}: {e}")
                # Write original line if conversion fails
                f_out.write(line)

    print(f"Converted {converted_count}/{total_count} samples")
    return converted_count, total_count

def main():
    """Main conversion function."""
    print("="*80)
    print("Format Conversion Tool")
    print("Converting from: Reasoning/Reflection/Answer")
    print("Converting to: <think>Reasoning only</think> + \\boxed{Answer}")
    print("Note: Reflection is DISCARDED")
    print("="*80)

    total_files = 0
    total_samples = 0

    for input_dir in INPUT_DIRS:
        input_path = Path(input_dir)

        if not input_path.exists():
            print(f"Warning: Directory not found: {input_dir}")
            continue

        # Create output directory
        output_dir = Path(f"{input_dir}{OUTPUT_SUFFIX}")
        output_dir.mkdir(exist_ok=True)

        # Process all JSONL files in the directory
        for input_file in input_path.glob("*.jsonl"):
            output_file = output_dir / input_file.name

            converted, total = convert_file(str(input_file), str(output_file))
            total_files += 1
            total_samples += converted

    print("\n" + "="*80)
    print("Conversion Complete!")
    print(f"Total files converted: {total_files}")
    print(f"Total samples converted: {total_samples}")
    print("="*80)

if __name__ == "__main__":
    main()
