import argparse
from pprint import pprint

from .client import NRPClient
from .tui import run_tui
from .agent_stub import UserResponseAgent


def main() -> None:
    parser = argparse.ArgumentParser(prog="nrp-tui")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("tui", help="Launch interactive TUI (default)")
    sub.add_parser("list-models", help="Print models to stdout")
    chat_parser = sub.add_parser("chat", help="Chat with a selected model")
    chat_parser.add_argument(
        "--model",
        default="gemma3",
        help="Model id to use (default: gemma3)",
    )

    args = parser.parse_args()

    if args.cmd == "list-models":
        client = NRPClient()
        models = client.list_models()
        pprint(models)
    elif args.cmd == "chat":
        run_chat_cli(model=args.model)
    else:
        run_tui()


def run_chat_cli(model: str) -> None:
    agent = UserResponseAgent(model=model)
    print(f"Starting chat with model '{model}' (Ctrl+C or 'exit' to quit)")
    print("System prompt: User Response agent focused on concise end-user replies.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        try:
            print(f"{model} (thinking...)")
            reply = agent.send(user_input)
        except Exception as exc:
            print(f"[error] Chat request failed: {exc}")
            continue

        print(f"{model}: {reply}\n")


if __name__ == "__main__":
    main()
