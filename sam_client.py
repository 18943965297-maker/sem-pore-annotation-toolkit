import json
import mimetypes
import os
import uuid
from urllib import error, request


class SamApiError(RuntimeError):
    pass


class SamClient:
    def __init__(self, base_url=None, timeout=180):
        if base_url is None:
            base_url = os.environ.get("SAM_API_URL", "http://127.0.0.1:8000/sam")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _json(self, req):
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read()
                return None if not raw else json.loads(raw.decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except json.JSONDecodeError:
                pass
            raise SamApiError(f"SAM API {exc.code}: {detail}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise SamApiError(f"无法连接 SAM API：{exc}") from exc

    def capabilities(self):
        return self._json(request.Request(f"{self.base_url}/capabilities/"))

    def start_session(self, image_bytes, model_type="sam2_s", initial_mask_bytes=None):
        boundary = f"----sam-editor-{uuid.uuid4().hex}"
        parts = []

        def field(name, value):
            parts.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(), b"\r\n",
            ])

        def file_field(name, filename, data):
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            parts.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {mime}\r\n\r\n".encode(), data, b"\r\n",
            ])

        field("type", model_type)
        file_field("image", "image.png", image_bytes)
        if initial_mask_bytes is not None:
            file_field("initial_mask", "mask.png", initial_mask_bytes)
        parts.append(f"--{boundary}--\r\n".encode())
        req = request.Request(
            f"{self.base_url}/session/start",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        return self._json(req)

    def predict(self, session_id, request_id, points, labels, multimask=True):
        payload = json.dumps({
            "session_id": session_id,
            "request_id": request_id,
            "points": points,
            "labels": labels,
            "multimask": multimask,
        }).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/session/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._json(req)

    def close_session(self, session_id):
        if not session_id:
            return
        req = request.Request(
            f"{self.base_url}/session/{session_id}", method="DELETE"
        )
        try:
            self._json(req)
        except SamApiError as exc:
            if "404" not in str(exc):
                raise
