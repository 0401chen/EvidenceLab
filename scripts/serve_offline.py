"""Serve an isolated rehearsal library with model transport disabled; no data seeding."""
import argparse, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--port", type=int, default=8782)
args=parser.parse_args(); data=args.data_dir.resolve()
if not data.is_relative_to((ROOT/"artifacts").resolve()) or not 1024<=args.port<=65535:
    parser.error("Use an isolated folder under project artifacts/ and a port from 1024 to 65535")
os.environ.update(EVIDENCELAB_DATA_DIR=str(data),EVIDENCE_PARSER="pymupdf",LLM_API_KEY="",OPENAI_API_KEY="",
    LLM_MODEL="",LLM_BASE_URL="http://127.0.0.1:9/v1")
sys.path.insert(0,str(ROOT))
from backend.app.main import app
from backend.app import providers
from backend.app.errors import ProviderError
from fastapi.responses import HTMLResponse
import httpx,uvicorn
async def denied(*args,**kwargs):
    raise ProviderError("offline_review","隔离排练库已关闭模型调用。")
async def no_transport(*args,**kwargs):
    raise RuntimeError("Offline rehearsal forbids model networking")
providers._chat_json=denied;httpx.AsyncClient.send=no_transport
@app.get("/")
def labelled():
    html=(ROOT/"frontend/dist/index.html").read_text(encoding="utf8")
    return HTMLResponse(html.replace("</body>",'<div style="position:fixed;bottom:0;left:0;right:0;z-index:99999;background:#fff3cd;padding:5px;text-align:center;font-size:12px">隔离排练库 · 模型调用关闭 · 人工结果不代表模型提取成功</div></body>'))
app.router.routes.insert(0,app.router.routes.pop())
uvicorn.run(app,host="127.0.0.1",port=args.port,access_log=False,log_level="warning")
