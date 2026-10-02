import httpx
import asyncio
import smtplib
from email.mime.text import MIMEText
from fastapi import FastAPI, Request
from typing import Optional, Dict
from loguru import logger
from pathlib import Path
import json
import os
import uvicorn
from datetime import datetime

class OpenClawBridge:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.webhook_port = 5055
        self.admin_email = os.environ.get("ADMIN_EMAIL", "")
        self.openclaw_url = "http://localhost:3000"
        self.whatsapp_number = ""
        self.smtp_host = "localhost"
        self.smtp_port = 25
        
        self.approval_event = asyncio.Event()
        self.approved = False

        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    self.webhook_port = config.get("openclaw_webhook_port", self.webhook_port)
                    self.admin_email = self.admin_email or config.get("alerts", {}).get("admin_email", "")
                    self.whatsapp_enabled = config.get("alerts", {}).get("whatsapp_enabled", True)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

    def _format_trade_message(self, proposal: Dict) -> str:
        action = proposal.get("action", "UNKNOWN")
        symbol = proposal.get("symbol", "UNKNOWN")
        lot = proposal.get("lot", 0.0)
        confidence = proposal.get("confidence", 0)
        reasoning = proposal.get("reasoning", "No reason provided")
        entry = proposal.get("entry", "N/A")
        sl = proposal.get("sl", "N/A")
        tp = proposal.get("tp", "N/A")
        atr = proposal.get("atr", "N/A")
        risk_amount = proposal.get("risk_amount", "N/A")
        risk_reward = proposal.get("risk_reward", "N/A")

        return (
            f"PROPOSAL: {action} {symbol}\n"
            f"Lot: {lot} | Entry: {entry}\n"
            f"SL: {sl} | TP: {tp}\n"
            f"ATR: {atr} | R:R 1:{risk_reward}\n"
            f"Risk: ${risk_amount} | Confidence: {confidence}%\n"
            f"Reason: {reasoning}\n"
            f"Reply YES to execute or NO to abort."
        )

    async def send_trade_proposal(self, proposal: Dict) -> bool:
        message = self._format_trade_message(proposal)
        return await self._send_whatsapp_message(message)

    async def _send_whatsapp_message(self, message: str) -> bool:
        if not self.whatsapp_number:
            logger.debug("[OpenClawBridge] WhatsApp number not configured. Message skipped.")
            return False

        payload = {
            "number": self.whatsapp_number,
            "message": message
        }
        
        for attempt in range(3):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(f"{self.openclaw_url}/api/messages/send", json=payload, timeout=10.0)
                    if response.status_code == 200:
                        logger.info("WhatsApp message sent successfully.")
                        return True
                    else:
                        logger.error(f"Failed to send WhatsApp message: {response.status_code} {response.text}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Network error sending WhatsApp message: {e}")
                await asyncio.sleep(2)
        return False

    async def wait_for_approval(self, timeout_seconds: int = 300) -> bool:
        self.approval_event.clear()
        self.approved = False
        
        logger.info(f"Waiting up to {timeout_seconds} seconds for approval...")
        try:
            await asyncio.wait_for(self.approval_event.wait(), timeout=timeout_seconds)
            if self.approved:
                logger.info("Trade APPROVED by human.")
                return True
            else:
                logger.info("Trade REJECTED by human.")
                return False
        except asyncio.TimeoutError:
            logger.warning("Trade approval TIMEOUT.")
            return False

    async def send_alert(self, message: str, level: str = "INFO"):
        logger.info(f"Sending alert ({level}): {message}")
        await self._send_whatsapp_message(f"[{level}] {message}")
        
        if level.upper() == "CRITICAL":
            await self.send_emergency_email(f"CRITICAL Alert: {datetime.now().isoformat()}", message)

    async def send_emergency_email(self, subject: str, body: str):
        email_enabled = self.config.get("alerts", {}).get("email_enabled", False)
        if not email_enabled or not self.admin_email or self.smtp_host in ("localhost", "127.0.0.1"):
            logger.debug(f"[OpenClawBridge] SMTP not configured. Emergency email skipped: {subject}")
            return

        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = "system@localhost"
            msg['To'] = self.admin_email

            def send_email():
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                    server.send_message(msg)
            
            await asyncio.to_thread(send_email)
            logger.info("Emergency email sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send emergency email: {e}")

    def create_webhook_app(self) -> FastAPI:
        app = FastAPI(title="OpenClaw Bridge Webhook")

        @app.get("/health")
        async def health_check():
            return {"status": "ok", "timestamp": datetime.now().isoformat()}

        @app.post("/webhook/approval")
        async def receive_approval(request: Request):
            try:
                data = await request.json()
                message = data.get("message", "").strip().upper()
                
                if message == "YES":
                    self.approved = True
                    self.approval_event.set()
                    logger.info("Received YES approval from webhook.")
                    return {"status": "received", "result": "approved"}
                elif message == "NO":
                    self.approved = False
                    self.approval_event.set()
                    logger.info("Received NO rejection from webhook.")
                    return {"status": "received", "result": "rejected"}
                else:
                    logger.info(f"Received unknown message from webhook: {message}")
                    return {"status": "ignored", "reason": "Not a YES/NO response"}
            except Exception as e:
                logger.error(f"Error processing webhook: {e}")
                return {"status": "error", "message": str(e)}

        return app

def start_webhook_server(config_path: str):
    logger.warning("Starting webhook server on 0.0.0.0. Ensure network is secured!")
    bridge = OpenClawBridge(config_path)
    app = bridge.create_webhook_app()
    uvicorn.run(app, host="0.0.0.0", port=bridge.webhook_port)
