import sys, os, time, tempfile, subprocess, re
import threading, queue
from pathlib import Path
import cv2

# ---------- 경로/환경 ----------
# 레포 루트 자동 추론(기본) + 환경변수 FASTVLM_DIR로 오버라이드
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]  # .../ml-fastvlm
FASTVLM_DIR = Path(os.getenv("FASTVLM_DIR", str(_DEFAULT_ROOT)))
PREDICT_PY = FASTVLM_DIR / "predict.py"

MODEL_PATH = os.getenv("FASTVLM_MODEL", "checkpoints/fastvlm_0.5b_stage3")
# 기본 프롬프트 (환경변수 > 기본값). 두 번째 CLI 인자로 덮어쓸 수 있음.
PROMPT_DEFAULT = os.getenv("FASTVLM_PROMPT", "Describe the scene briefly.")
# 실행 중 변경 가능한 현재 프롬프트 (mutable)
CURRENT_PROMPT = PROMPT_DEFAULT
USE_VLM = os.getenv("USE_VLM", "1") not in ("0", "false", "False")
PROC_TIMEOUT = int(os.getenv("FASTVLM_TIMEOUT", "180"))  # 초
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")  # <-- 추가

# ---------- 유틸 ----------
def parse_predict_output(out: str) -> str:
    """
    predict.py 출력에서 최종 답변만 안정적으로 추출.
    우선순위:
      1) 'ASSISTANT:' 혹은 'Answer:' 마커 뒤 텍스트
      2) 마지막 비어있지 않은 줄
    """
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    joined = "\n".join(lines)

    # 마커 패턴
    for pat in (r"ASSISTANT:\s*(.*)", r"Answer:\s*(.*)"):
        m = re.search(pat, joined, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()

    return lines[-1] if lines else ""

def wrap_text_cv(text, font, font_scale, thickness, max_width):
    """
    OpenCV 텍스트 폭 기준으로 줄바꿈.
    """
    words = text.split()
    if not words:
        return []

    lines, cur = [], words[0]
    for w in words[1:]:
        size, _ = cv2.getTextSize(cur + " " + w, font, font_scale, thickness)
        if size[0] <= max_width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def draw_overlay(frame, text, max_lines=3):
    if not text:
        return frame

    H, W = frame.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    padding = 10
    max_text_width = int(W * 0.95)

    lines = wrap_text_cv(text, font, scale, thick, max_text_width)
    lines = lines[:max_lines]

    # 계산된 높이
    line_h = cv2.getTextSize("Ag", font, scale, thick)[0][1] + 6
    box_h = padding * 2 + line_h * len(lines)

    # 상단 반투명 박스(불투명 사각형)
    cv2.rectangle(frame, (0, 0), (W, box_h), (0, 0, 0), -1)

    y = padding + line_h
    for ln in lines:
        cv2.putText(frame, ln, (padding, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h

    return frame

def resolve_model_path(s: str) -> str:
    """
    - 절대/상대 로컬 디렉터리가 존재하면 그 경로를 반환
    - 없으면 HF repo_id로 간주하여 원문을 반환
    """
    p = Path(s)
    if p.is_dir():
        return str(p.resolve())
    p2 = (FASTVLM_DIR / s)
    if p2.is_dir():
        return str(p2.resolve())
    return s

# ---------- VLM 워커 스레드 ----------
class VLMWorker(threading.Thread):
    def __init__(self, work_q: queue.Queue, result_q: queue.Queue, model_path: str, hf_token: str | None):
        super().__init__(daemon=True)
        self.work_q = work_q
        self.result_q = result_q
        self.model_path = model_path
        self.hf_token = hf_token

    def run(self):
        while True:
            item = self.work_q.get()
            try:
                if item is None:
                    # 종료 신호
                    return
                img_path, prompt = item

                if not PREDICT_PY.exists():
                    self.result_q.put(("err", f"predict.py not found at: {PREDICT_PY}"))
                    continue

                cmd = [
                    sys.executable, str(PREDICT_PY),
                    "--model-path", self.model_path,           # <-- 수정: resolve된 경로 사용
                    "--image-file", img_path,
                    "--prompt", prompt
                ]
                env = os.environ.copy()
                if self.hf_token and "HUGGINGFACE_HUB_TOKEN" not in env:
                    env["HUGGINGFACE_HUB_TOKEN"] = self.hf_token  # <-- 토큰 전달

                try:
                    out = subprocess.check_output(
                        cmd, stderr=subprocess.STDOUT, text=True, timeout=PROC_TIMEOUT, env=env
                    )
                    answer = parse_predict_output(out)
                    if not answer:
                        answer = "(empty answer)"
                    self.result_q.put(("ok", answer))
                except subprocess.TimeoutExpired:
                    self.result_q.put(("err", f"predict timeout > {PROC_TIMEOUT}s"))
                except subprocess.CalledProcessError as e:
                    # 에러 로그의 마지막 20줄만 요약 + 힌트
                    lines = e.output.splitlines()
                    tail = "\n".join(lines[-20:]) if lines else str(e)
                    low = tail.lower()
                    hints = []
                    if ("repo_id" in low) or ("repo type" in low):
                        hints.append("[hint] FASTVLM_MODEL이 올바른 로컬 경로(체크포인트 디렉터리) 또는 유효한 HF repo_id인지 확인하세요.")
                    if ("private" in low or "gated" in low) and not self.hf_token:
                        hints.append("[hint] gated/private 모델은 'huggingface-cli login' 또는 HUGGINGFACE_HUB_TOKEN 설정이 필요합니다.")
                    if hints:
                        tail += "\n" + "\n".join(hints)
                    self.result_q.put(("err", f"predict failed:\n{tail}"))
            finally:
                # 임시 파일 정리
                if item and isinstance(item, tuple):
                    img_path = item[0]
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass
                # 작업 완료 표시
                self.work_q.task_done()

# ---------- 메인 루프 ----------
def main():
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;3000000|buffer_size;102400|max_delay;500000|reorder_queue_size;0",
    )

    global CURRENT_PROMPT
    default_url = "rtsp://USER:PASS@192.168.0.96:554/axis-media/media.amp?streamprofile=profile_1_h264"
    # 인자: 1) RTSP_URL 2) 초기 프롬프트
    rtsp_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("RTSP_URL", default_url)
    if len(sys.argv) > 2:
        CURRENT_PROMPT = sys.argv[2]

    model_resolved = resolve_model_path(MODEL_PATH)  # <-- 추가

    print(f"[infer] python: {sys.executable}")
    print(f"[infer] repo:   {FASTVLM_DIR}")
    print(f"[infer] model:  {MODEL_PATH}")
    print(f"[infer] model(resolved): {model_resolved}")  # <-- 추가
    print(f"[infer] prompt: {CURRENT_PROMPT}")
    print(f"[infer] rtsp:   {rtsp_url}")
    print(f"[infer] predict.py: {PREDICT_PY}")
    print(f"[infer] HF token: {'set' if HF_TOKEN else 'not set'}")  # <-- 추가

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[infer] RTSP open failed, retry with longer timeout...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|stimeout;10000000|buffer_size;102400|max_delay;500000|reorder_queue_size;0"
        )
        cap.release()
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[infer] RTSP still failed to open. Check URL/credentials/network/firewall.")
        sys.exit(1)

    print("[infer] RTSP opened")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    work_q, result_q = queue.Queue(maxsize=1), queue.Queue()
    vlm = VLMWorker(work_q, result_q, model_resolved, HF_TOKEN) if USE_VLM else None  # <-- 수정
    if vlm:
        vlm.start()
        print("[infer] VLM worker started")
    else:
        print("[infer] VLM disabled (USE_VLM=0)")

    last_vlm_ts = 0.0
    overlay_text = ""
    prompt_changed_ts = 0.0
    prev_ts, frame_count = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame_count += 1
        now = time.time()
        if now - prev_ts >= 1.0:
            fps = frame_count / (now - prev_ts)
            cv2.setWindowTitle("CCTV + FastVLM", f"CCTV + FastVLM (FPS {fps:.1f})")
            prev_ts, frame_count = now, 0

        # 1) 주기적으로 최신 프레임을 VLM에 전달(큐는 1개만 유지)
        if USE_VLM and (now - last_vlm_ts > 1.0) and work_q.empty():
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            # 크기 줄여 저장(속도/대역폭)
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280.0 / w
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                frame_small = frame
            cv2.imwrite(tmp_path, frame_small, [cv2.IMWRITE_JPEG_QUALITY, 90])

            try:
                work_q.put_nowait((tmp_path, CURRENT_PROMPT))
                last_vlm_ts = now
            except queue.Full:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        # 2) 결과 수신 시 오버레이 갱신
        try:
            status, msg = result_q.get_nowait()
            overlay_text = msg if status == "ok" else f"[VLM error] {msg}"
        except queue.Empty:
            pass

        # 3) 오버레이 표시
        frame_disp = draw_overlay(frame, overlay_text, max_lines=3)  # 세줄까지만 문구 표시

        cv2.imshow("CCTV + FastVLM", frame_disp)
        k = cv2.waitKey(1)
        if k == 27:  # ESC
            break
        if k in (ord('p'), ord('P')):
            # 콘솔에서 새 프롬프트 입력 (GUI 창 포커스 -> 터미널로 전환 필요)
            try:
                new_prompt = input("New prompt > ").strip()
                if new_prompt:
                    CURRENT_PROMPT = new_prompt
                    overlay_text = f"[prompt updated] {CURRENT_PROMPT}"
                    prompt_changed_ts = time.time()
            except EOFError:
                pass

    if vlm:
        # 종료 신호 전송 및 정리
        try:
            work_q.put_nowait(None)
        except queue.Full:
            # 큐가 꽉 차 있으면 비우고 종료 신호
            try:
                work_q.get_nowait()
                work_q.task_done()
            except queue.Empty:
                pass
            work_q.put_nowait(None)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
