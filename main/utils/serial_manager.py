import datetime
import uuid

class SerialNumberManager:
    @staticmethod
    def generate_serial_number():
        timestamp_part = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        uuid_part = str(uuid.uuid4())
        return f"REC-{timestamp_part}-{uuid_part}"

serial_code = SerialNumberManager.generate_serial_number()
print(serial_code)
# Example output: REC-20251013103045123456-a1b2c3d4-e5f6-7890-abcd-ef1234567890
