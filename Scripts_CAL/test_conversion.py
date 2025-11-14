"""
Test script to verify the format conversion
"""

old_format = """Reasoning: The premise states that "She works in a warehouse now," implying her current employment status. The hypothesis, "She just started recently at the warehouse," suggests a recent start. The premise does not specify when she started working there, only that she currently works there. This means it's possible she could have started recently, but the timing is not confirmed by the premise. Thus, the hypothesis is neither directly supported nor contradicted by the premise, resulting in a neutral relationship.

Reflection: NO

Answer: neutral"""

expected_new_format = """<think>
The premise states that "She works in a warehouse now," implying her current employment status. The hypothesis, "She just started recently at the warehouse," suggests a recent start. The premise does not specify when she started working there, only that she currently works there. This means it's possible she could have started recently, but the timing is not confirmed by the premise. Thus, the hypothesis is neither directly supported nor contradicted by the premise, resulting in a neutral relationship.
</think>

\\boxed{neutral}"""

print("OLD FORMAT:")
print("-" * 80)
print(old_format)
print("\n" + "=" * 80 + "\n")
print("EXPECTED NEW FORMAT:")
print("-" * 80)
print(expected_new_format)
print("\n" + "=" * 80 + "\n")

# Test conversion
from convert_format import convert_response_format

converted = convert_response_format(old_format)
print("CONVERTED FORMAT:")
print("-" * 80)
print(converted)
print("\n" + "=" * 80 + "\n")

# Check if conversion matches expected
if converted.strip() == expected_new_format.strip():
    print("✓ Conversion successful!")
else:
    print("✗ Conversion mismatch")
    print("\nExpected:")
    print(repr(expected_new_format))
    print("\nGot:")
    print(repr(converted))
