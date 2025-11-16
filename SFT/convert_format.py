import json
import re
from pathlib import Path
from tqdm import tqdm

# ====================================================================
# Configuration
# ====================================================================

INPUT_DIRS = [
    "./data/A",
    "./data/B"
]

OUTPUT_SUFFIX = "_converted"  # Will create DA_converted, DB_converted

# ====================================================================
# Conversion Functions
# ====================================================================

def is_already_new_format(response: str) -> bool:
    """Check if response is already in new format."""
    # Check for presence of <think> and <bias> tags
    has_think = '<think>' in response and '</think>' in response
    has_bias = '<bias>' in response and '</bias>' in response
    return has_think and has_bias

def is_target_boxed_format(response: str) -> bool:
    """Check if response is already in target format with \\boxed{}."""
    has_think = '<think>' in response and '</think>' in response
    has_boxed = '\\boxed{' in response
    has_no_bias = '<bias>' not in response
    return has_think and has_boxed and has_no_bias

def convert_response_format(old_response: str) -> str:
    """
    Convert to target format:

    Source format 1 (old):
        Reasoning: ...
        Reflection: YES/NO
        Answer: ...

    Source format 2 (intermediate):
        <think>...</think>
        <bias>Yes/No</bias>
        answer

    Target format:
        <think>
        [Reasoning content]
        </think>

        \\boxed{answer}
    """
    # Check if already in target format
    if is_target_boxed_format(old_response):
        return old_response  # Return as-is if already converted

    # Case 1: Convert from intermediate format (<think> + <bias> + answer)
    if is_already_new_format(old_response):
        # Extract reasoning from <think>
        think_match = re.search(r'<think>(.*?)</think>', old_response, re.DOTALL | re.IGNORECASE)
        reasoning = think_match.group(1).strip() if think_match else ""

        # Extract answer after <bias> tag
        bias_match = re.search(r'<bias>.*?</bias>\s*(.*)', old_response, re.DOTALL | re.IGNORECASE)
        answer = bias_match.group(1).strip() if bias_match else ""

        # Build target format
        new_response = f"""<think>
{reasoning}
</think>

\\boxed{{{answer}}}"""
        return new_response

    # Case 2: Convert from old format (Reasoning/Reflection/Answer)
    reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=Reflection:|Answer:|$)', old_response, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'Answer:\s*(.*?)$', old_response, re.DOTALL | re.IGNORECASE)

    # Extract text (discard Reflection)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    # Build target format
    new_response = f"""<think>
{reasoning}
</think>

\\boxed{{{answer}}}"""

    return new_response

def convert_file(input_path: str, output_path: str):
    """Convert a JSONL file from old format to new format."""
    print(f"Converting: {input_path} -> {output_path}")

    converted_count = 0
    skipped_count = 0
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

                # Check if already in target format
                if is_target_boxed_format(old_response):
                    skipped_count += 1
                    new_response = old_response
                else:
                    new_response = convert_response_format(old_response)
                    converted_count += 1

                # Update data
                data['response'] = new_response

                # Write to output
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"Error processing line {total_count}: {e}")
                # Write original line if conversion fails
                f_out.write(line)

    print(f"Total: {total_count} | Converted: {converted_count} | Already new format: {skipped_count}")
    return converted_count, total_count

def main():
    """Main conversion function."""
    print("="*80)
    print("Format Conversion Tool")
    print("Source formats:")
    print("  - Old: Reasoning/Reflection/Answer")
    print("  - Intermediate: <think>...</think> + <bias>...</bias> + Answer")
    print("Target format: <think>...</think> + \\boxed{answer}")
    print("Note: <bias> tags are REMOVED in target format")
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
