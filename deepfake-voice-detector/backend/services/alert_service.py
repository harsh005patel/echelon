from loguru import logger
from datetime import datetime

class AlertService:
    def __init__(self):
        # In a real app, this would connect to a DB or notification service (Email, SMS)
        pass

    async def send_alert(self, data):
        """
        data: dict containing detection results and metadata
        """
        timestamp = datetime.now().isoformat()
        confidence = data.get('confidence', 0.0)
        
        if confidence > 0.8:
            level = "CRITICAL"
        elif confidence > 0.6:
            level = "WARNING"
        else:
            level = "INFO"
            
        alert_msg = {
            "timestamp": timestamp,
            "level": level,
            "message": f"Deepfake detected with {confidence:.2%} confidence",
            "details": data
        }
        
        # Log to file/console
        logger.bind(payload=alert_msg).warning(f"ALERT: {alert_msg['message']}")
        
        # Here we could push to a websocket specific for alerts if needed
        return alert_msg
