import sys
import os
import time

import cv2


# 영상 스트리밍 불러오는 코드
def main():
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;3000000|buffer_size;102400|max_delay;500000|reorder_queue_size;0",
    )

    print(f"[streaming] file: {__file__}")
    print(f"[streaming] python: {sys.executable}")

    # RTSP URL: from CLI arg > env > default
    default_url = "rtsp://pinksooin:mypassword123@192.168.0.96:554/axis-media/media.amp?streamprofile=profile_1_h264"
    rtsp_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.getenv("RTSP_URL", default_url)
    )

    # Try to open RTSP
    print(f"[streaming] opening RTSP (FFMPEG+TCP, short timeout): {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[streaming] RTSP open failed. Retrying with longer timeout...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|stimeout;10000000|buffer_size;102400|max_delay;500000|reorder_queue_size;0"
        )
        cap.release()
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[streaming] RTSP still failed to open. Check URL/credentials/network/firewall.")
        sys.exit(1)

    print("[streaming] RTSP opened")

    # Reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    no_frame_warn_ts = 0
    prev_ts = time.time()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            now = time.time()
            if now - no_frame_warn_ts > 2:
                print("[streaming] no frame yet... retrying")
                no_frame_warn_ts = now
            time.sleep(0.05)
            continue

        frame_count += 1
        now = time.time()
        if now - prev_ts >= 1.0:
            fps = frame_count / (now - prev_ts)
            print(f"[streaming] FPS: {fps:.1f}")
            prev_ts = now
            frame_count = 0

        cv2.imshow("CCTV", frame)
        # ESC to exit -> 마우스 커서로는 종료 안됌
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
