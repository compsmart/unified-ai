#!/usr/bin/env python3
"""
Interactive CLI for Unified AI

Commands:
  store <entity> <role> <value>  - Store a fact
  query <entity> <role>          - Query a fact
  learn <concept> [category]     - Register a new concept
  predict <concept>              - Predict next in sequence
  sequence <c1> <c2> <c3> ...    - Train a sequence
  list                           - Show all concepts
  facts                          - Show all stored facts
  save <path>                    - Save model
  load <path>                    - Load model
  help                           - Show this help
  quit                           - Exit
"""

import sys
import os
import readline  # Enable arrow keys and history

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_ai import UnifiedAI, UnifiedAIConfig


def print_help():
    print("""
Unified AI Interactive Shell
============================

Commands:
  store <entity> <role> <value>  - Store a fact
      Example: store Alice job Engineer

  query <entity> <role>          - Query a fact
      Example: query Alice job

  learn <concept> [category]     - Register a new concept
      Example: learn Active value
      Example: learn Status role

  predict <concept>              - Predict next in sequence
      Example: predict Start

  sequence <c1> <c2> <c3> ...    - Train a sequence
      Example: sequence Wake Eat Work Sleep

  list [category]                - Show all concepts (optional: filter by category)
      Example: list
      Example: list value

  facts                          - Show stored fact contexts

  stats                          - Show system statistics

  save <path>                    - Save model to directory
      Example: save /tmp/mymodel

  load <path>                    - Load model from directory
      Example: load /tmp/mymodel

  help                           - Show this help

  quit / exit                    - Exit the shell
""")


def main():
    print("=" * 50)
    print("UNIFIED AI - Interactive Shell")
    print("=" * 50)
    print("Type 'help' for commands, 'quit' to exit")
    print()

    # Initialize system
    config = UnifiedAIConfig(
        lattice_size=256,
        hv_dimensions=10000,
        similarity_threshold=0.10
    )
    ai = UnifiedAI(config)

    # Pre-register some common concepts
    for role in ['status', 'type', 'location', 'job', 'color', 'size']:
        ai.register_concept(role, category='role')

    print(f"Initialized with {len(ai.hippocampus.item_memory)} base concepts")
    print()

    while True:
        try:
            line = input("unified> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            elif cmd == 'help':
                print_help()

            elif cmd == 'store':
                if len(args) < 3:
                    print("Usage: store <entity> <role> <value>")
                    continue
                entity, role, value = args[0], args[1], ' '.join(args[2:])

                # Auto-register concepts if needed
                if role not in ai.hippocampus.item_memory:
                    ai.register_concept(role, category='role')
                if value not in ai.hippocampus.item_memory:
                    ai.register_concept(value, category='value')

                result = ai.store_fact(entity, role, value)
                if result['success']:
                    print(f"Stored: {entity}.{role} = {value}")
                else:
                    print(f"Failed: {result.get('error', 'unknown error')}")

            elif cmd == 'query':
                if len(args) < 2:
                    print("Usage: query <entity> <role>")
                    continue
                entity, role = args[0], args[1]
                result = ai.query_fact(entity, role)
                if result['value']:
                    print(f"{entity}.{role} = {result['value']} (confidence: {result['confidence']:.2f})")
                else:
                    print(f"No value found for {entity}.{role}")

            elif cmd == 'learn':
                if len(args) < 1:
                    print("Usage: learn <concept> [category]")
                    continue
                concept = args[0]
                category = args[1] if len(args) > 1 else 'general'
                ai.register_concept(concept, category=category)
                print(f"Learned concept: {concept} (category: {category})")

            elif cmd == 'sequence':
                if len(args) < 2:
                    print("Usage: sequence <c1> <c2> <c3> ...")
                    continue
                # Register concepts with neuron_indices for lattice
                neurons_per_concept = 20
                for i, c in enumerate(args):
                    if c not in ai.hippocampus.item_memory:
                        indices = list(range(i * neurons_per_concept, (i + 1) * neurons_per_concept))
                        ai.register_concept(c, category='state', neuron_indices=indices)
                ai.train_sequence(args, repetitions=20)
                print(f"Trained sequence: {' -> '.join(args)}")

            elif cmd == 'predict':
                if len(args) < 1:
                    print("Usage: predict <concept>")
                    continue
                # Check if concept has neurons
                neurons = ai.semantic_bridge.concepts_to_neurons(args)
                if not neurons:
                    print(f"Concept '{args[0]}' not linked to lattice. Train a sequence first.")
                    continue
                result = ai.predict_next(current_concepts=args, steps=5)
                predictions = result.get('predicted_concepts', [])
                if predictions:
                    print("Predictions:")
                    for concept, conf in predictions[:5]:
                        print(f"  {concept}: {conf:.2f}")
                else:
                    print("No predictions available")

            elif cmd == 'list':
                category_filter = args[0] if args else None
                concepts = ai.hippocampus.item_memory
                cat_index = ai.hippocampus.category_index

                if category_filter:
                    if category_filter in cat_index:
                        items = cat_index[category_filter]
                        print(f"Concepts in '{category_filter}':")
                        for c in sorted(items)[:30]:
                            print(f"  {c}")
                        if len(items) > 30:
                            print(f"  ... and {len(items) - 30} more")
                        print(f"Total: {len(items)}")
                    else:
                        print(f"No category '{category_filter}' found")
                        print(f"Available: {list(cat_index.keys())}")
                else:
                    for cat, items in sorted(cat_index.items()):
                        print(f"\n{cat}:")
                        for c in sorted(items)[:20]:
                            print(f"  {c}")
                        if len(items) > 20:
                            print(f"  ... and {len(items) - 20} more")
                    print(f"\nTotal concepts: {len(concepts)}")

            elif cmd == 'facts':
                contexts = list(ai.fact_memory.state_vectors.keys())
                if contexts:
                    print(f"Stored fact contexts ({len(contexts)}):")
                    for ctx in contexts[:20]:
                        print(f"  {ctx}")
                    if len(contexts) > 20:
                        print(f"  ... and {len(contexts) - 20} more")
                else:
                    print("No facts stored yet")

            elif cmd == 'stats':
                print(f"Concepts: {len(ai.hippocampus.item_memory)}")
                print(f"Facts: {len(ai.fact_memory.state_vectors)}")
                print(f"Lattice size: {ai.config.lattice_size}")
                print(f"HV dimensions: {ai.config.hv_dimensions}")

            elif cmd == 'save':
                if len(args) < 1:
                    print("Usage: save <path>")
                    continue
                path = args[0]
                ai.save(path)
                print(f"Model saved to: {path}")

            elif cmd == 'load':
                if len(args) < 1:
                    print("Usage: load <path>")
                    continue
                path = args[0]
                if os.path.exists(path):
                    ai.load(path)
                    print(f"Model loaded from: {path}")
                else:
                    print(f"Path not found: {path}")

            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
