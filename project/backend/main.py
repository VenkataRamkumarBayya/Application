from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import io
import contextlib

# ✅ Shared context for exec
global_context = {}

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
def init_db():
    conn = sqlite3.connect("ml_lab.db")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS code_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code_text TEXT NOT NULL,
        output_text TEXT,
        error_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()

# Serve frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("../frontend/demo.js")

# Execute code
@app.post("/run-code")
async def run_code(request: Request):
    data = await request.json()
    code = data.get("code", "")

    conn = sqlite3.connect("ml_lab.db")
    conn.row_factory = sqlite3.Row

    try:
        stdout = io.StringIO()
        stderr = io.StringIO()

        global_context["conn"] = conn

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, global_context)

        output = stdout.getvalue()
        error = stderr.getvalue()

    except Exception as e:
        output = ""
        error = str(e)

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO code_runs (code_text, output_text, error_text)
        VALUES (?, ?, ?)
        """,
        (code, output, error)
    )
    conn.commit()
    run_id = cur.lastrowid
    conn.close()

    return {
        "output": output,
        "error": error,
        "run_id": run_id
    }
