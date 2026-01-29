#!/usr/bin/env python
import sys
import json
sys.path.insert(0, 'generate_sets/knowledge_sets_static')

from generate_knowledge_set import main

# Load config
with open('generate_sets/knowledge_sets_static/config_wag_drizzle_style.json') as f:
    config = json.load(f)

# Run generation
result = main(config_json=config, output_base='generate_sets/knowledge_sets_static/outputs')
print("\n" + "="*50)
print("Knowledge set generated successfully!")
print("="*50)
print(f"\nOutput paths:")
for category, paths in result['outputs_paths'].items():
    print(f"\n{category}:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")
