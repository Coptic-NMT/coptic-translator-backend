from pathlib import Path



API_KEYS = {
    k: v for k, v in (
        line.split("=") for line in Path('server_python/.env.local').read_text().splitlines() if line
    )
}