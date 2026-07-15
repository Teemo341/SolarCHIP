#!/usr/bin/env python3
"""
进程监控脚本：持续监控指定 PID，进程结束后自动发送邮件通知。
用法: python scripts/monitor_process.py <PID>
"""

import sys
import time
import os
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# ==================== 邮件配置（QQ邮箱） ====================
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465  # SSL
SENDER_EMAIL = "838839085@qq.com"
AUTH_CODE = "mmbbhivqekvobdac"  # QQ邮箱SMTP授权码
RECEIVER_EMAIL = "838839085@qq.com"

# ==================== 监控配置 ====================
CHECK_INTERVAL = 60  # 检查间隔（秒）
# ==========================================================


def get_process_info(pid: int) -> dict:
    """获取进程信息"""
    try:
        # 读取 /proc/<pid>/cmdline 获取命令行
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
            cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()

        # 读取 /proc/<pid>/stat 获取状态
        with open(f"/proc/{pid}/stat", "r") as f:
            stat = f.read()
            # 进程名在括号内，格式: pid (comm) state ...
            comm_end = stat.rfind(")")
            comm = stat[stat.find("(") + 1:comm_end]
            state = stat[comm_end + 2] if comm_end + 2 < len(stat) else "?"

        return {
            "pid": pid,
            "comm": comm,
            "cmdline": cmdline,
            "state": state,
        }
    except FileNotFoundError:
        return None
    except Exception as e:
        return {"pid": pid, "error": str(e)}


def read_exit_status(pid: int) -> str:
    """尝试读取进程退出信息"""
    try:
        # 检查 /proc/<pid>/status 中是否有退出信息
        with open(f"/proc/{pid}/status", "r") as f:
            status = f.read()
            for line in status.splitlines():
                if line.startswith("State:"):
                    return line.strip()
    except FileNotFoundError:
        pass
    return "进程已退出（无法获取退出状态）"


def send_email(process_info: dict, start_time: datetime, exit_time: datetime, exit_code: str):
    """发送邮件通知"""
    hostname = socket.gethostname()
    duration = exit_time - start_time

    subject = f"[训练完成] PID {process_info.get('pid')} 已结束 - {hostname}"

    body = f"""
<h3>进程监控通知</h3>
<p>您监控的进程已结束，详情如下：</p>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
  <tr><td><b>PID</b></td><td>{process_info.get('pid')}</td></tr>
  <tr><td><b>进程名</b></td><td>{process_info.get('comm')}</td></tr>
  <tr><td><b>命令行</b></td><td>{process_info.get('cmdline')}</td></tr>
  <tr><td><b>主机</b></td><td>{hostname}</td></tr>
  <tr><td><b>开始监控时间</b></td><td>{start_time.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
  <tr><td><b>结束时间</b></td><td>{exit_time.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
  <tr><td><b>运行时长</b></td><td>{str(duration).split('.')[0]}</td></tr>
  <tr><td><b>退出状态</b></td><td>{exit_code}</td></tr>
</table>
<p><i>此邮件由 monitor_process.py 自动发送</i></p>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.attach(MIMEText(body, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, AUTH_CODE)
            server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg.as_string())
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 邮件已发送至 {RECEIVER_EMAIL}")
        return True
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 邮件发送失败: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/monitor_process.py <PID>")
        sys.exit(1)

    pid = int(sys.argv[1])

    # 获取初始进程信息
    proc_info = get_process_info(pid)
    if proc_info is None or "error" in proc_info:
        print(f"❌ PID {pid} 不存在或无法访问")
        sys.exit(1)

    start_time = datetime.now()
    print("=" * 60)
    print(f"🔍 开始监控进程 PID={pid}")
    print(f"   进程名: {proc_info.get('comm')}")
    print(f"   命令行: {proc_info.get('cmdline')}")
    print(f"   开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   检查间隔: {CHECK_INTERVAL} 秒")
    print(f"   邮件通知: {RECEIVER_EMAIL}")
    print("=" * 60)

    last_good_info = proc_info  # 保存最后一次成功获取的进程信息
    last_alive_msg = start_time
    alive_msg_interval = 3600  # 每小时打印一次存活确认

    while True:
        proc_info = get_process_info(pid)

        if proc_info is None or "error" in proc_info:
            # 进程已退出
            exit_time = datetime.now()
            exit_status = f"进程已退出（状态: {proc_info.get('error') if proc_info and 'error' in proc_info else 'PID不存在'}）"
            print(f"\n[{exit_time.strftime('%H:%M:%S')}] ⏹ 进程 PID={pid} 已退出")
            print(f"   退出状态: {exit_status}")
            print(f"   运行时长: {str(exit_time - start_time).split('.')[0]}")
            print(f"   正在发送邮件通知...")

            send_email(last_good_info, start_time, exit_time, exit_status)
            print("👋 监控结束")
            break

        # 更新最后一次成功获取的信息
        last_good_info = proc_info

        # 进程仍在运行，定期打印存活确认
        now = datetime.now()
        if now - last_alive_msg >= timedelta(seconds=alive_msg_interval):
            elapsed = now - start_time
            print(f"[{now.strftime('%H:%M:%S')}] ✅ PID={pid} 仍在运行中... "
                  f"(已监控 {elapsed.seconds // 3600} 小时 "
                  f"{elapsed.seconds % 3600 // 60} 分钟)")
            last_alive_msg = now

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
