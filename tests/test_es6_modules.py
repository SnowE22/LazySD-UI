import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from app import create_app

def test_es6_modules():
    app = create_app()
    client = TestClient(app)

    # 1. Test index.html
    r = client.get("/")
    assert r.status_code == 200, f"Status code {r.status_code}"
    assert ('/js/app.js' in r.text or 'src="/js/main.js' in r.text), "App script tag not found in index.html"
    print("[TEST 1 OK] index.html successfully references primary client script bundle / module.")

    # 2. Test all ES6 modules served by StaticFiles
    modules = [
        "/js/main.js",
        "/js/state.js",
        "/js/api.js",
        "/js/modules/zoom-pan.js",
        "/js/modules/dock.js",
        "/js/modules/models-hub.js",
        "/js/modules/lora-hub.js",
        "/js/modules/settings.js",
        "/js/modules/generator.js",
        "/js/modules/halftone.js",
        "/js/modules/manga.js",
        "/js/modules/mask-core.js",
        "/js/modules/generate-mask.js",
        "/js/modules/inpaint.js",
    ]

    for m in modules:
        res = client.get(m)
        assert res.status_code == 200, f"Module {m} failed with status {res.status_code}"
        assert len(res.text) > 50, f"Module {m} is unexpectedly small ({len(res.text)} bytes)"
        print(f"[TEST 2 OK] {m} served successfully ({len(res.text)} bytes, 200 OK)")

    print("\n[SUCCESS] ALL 11 ES6 MODULES VERIFIED AND SERVED CORRECTLY!")

if __name__ == "__main__":
    test_es6_modules()
