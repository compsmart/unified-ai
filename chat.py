#!/usr/bin/env python3
"""
Interactive Chat Interface for Unified AI

A conversational agent that learns, reasons, and responds honestly.
"""

import sys
import os
import readline  # Enable arrow keys and history

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from conversation.agent import ConversationalAgent


def main():
    print("=" * 50)
    print("UNIFIED AI - Conversational Agent")
    print("=" * 50)
    print("Type 'help' for commands, 'quit' to exit")
    print()

    agent = ConversationalAgent()

    # Check for saved state
    save_path = os.path.join(os.path.dirname(__file__), 'models', 'chat_agent')
    if os.path.exists(save_path):
        try:
            agent.load(save_path)
            print("(Loaded previous conversation state)")
        except:
            pass

    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Special commands
        if user_input.lower() == 'save':
            os.makedirs(save_path, exist_ok=True)
            agent.save(save_path)
            print(f"AI: Saved to {save_path}")
            continue

        if user_input.lower() == 'load':
            if os.path.exists(save_path):
                agent.load(save_path)
                print("AI: Loaded previous state.")
            else:
                print("AI: No saved state found.")
            continue

        # Get response
        response = agent.chat(user_input)

        # Display response
        print(f"AI: {response.text}")

        if response.learned:
            print("    (learned)")

        if response.action == "quit":
            # Auto-save on quit
            os.makedirs(save_path, exist_ok=True)
            agent.save(save_path)
            break

    print("\nConversation ended.")


if __name__ == "__main__":
    main()
