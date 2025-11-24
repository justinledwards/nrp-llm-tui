import argparse
from pprint import pprint

from .client import NRPClient
from .tui import run_tui
from .agent_stub import UserResponseAgent
from .sessions import SessionStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="nrp-tui")
    sub = parser.add_subparsers(dest="cmd")

    tui_parser = sub.add_parser("tui", help="Launch interactive TUI (default)")
    tui_parser.add_argument(
        "--session",
        default="tui",
        help="Session name to load or create (default: tui)",
    )
    tui_parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start a new session even if one already exists with the same name.",
    )
    sub.add_parser("list-models", help="Print models to stdout")
    chat_parser = sub.add_parser("chat", help="Chat with a selected model")
    chat_parser.add_argument(
        "--model",
        default="gemma3",
        help="Model id to use (default: gemma3)",
    )
    chat_parser.add_argument(
        "--session",
        default="cli",
        help="Session label to use for conversation logs (default: cli)",
    )
    chat_parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start a fresh session instead of resuming an existing one.",
    )

    args = parser.parse_args()

    if args.cmd == "list-models":
        client = NRPClient()
        models = client.list_models()
        pprint(models)
    elif args.cmd == "chat":
        run_chat_cli(
            model=args.model,
            session_name=args.session,
            resume=not args.new_session,
        )
    elif args.cmd == "tui":
        run_tui(
            session_label=args.session,
            resume=not args.new_session,
        )
    else:
        run_tui()


def run_chat_cli(model: str, session_name: str | None = None, resume: bool = True) -> None:
    session_label = session_name or "cli"
    store = SessionStore()
    session = store.get_or_create(session_label, resume=resume)
    agent = UserResponseAgent(model=model, session=session, load_history=resume)
    print(f"Starting chat with model '{model}' (Ctrl+C or 'exit' to quit)")
    print("System prompt: User Response agent focused on concise end-user replies.")
    print(f"Session: {session.display_name} ({session.id})")
    print(f"Conversation log: {agent.log_path}")
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
