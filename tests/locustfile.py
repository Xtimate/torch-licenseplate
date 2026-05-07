import io
import random

from locust import HttpUser, between, task
from PIL import Image


def make_image_bytes(width=200, height=100):
    color = (random.randint(200, 255), random.randint(200, 255), random.randint(0, 50))
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


class SpotterUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        self.client.get("/health")

    @task(2)
    def get_history(self):
        self.client.get("/history?limit=50")

    @task(2)
    def get_stats(self):
        self.client.get("/stats")

    @task(1)
    def get_analytics(self):
        self.client.get("/analytics")

    @task(2)
    def recognize(self):
        img_bytes = make_image_bytes(width=188, height=48)
        self.client.post(
            "/recognize",
            files={"file": ("plate.jpg", img_bytes, "image/jpeg")},
            name="/recognize",
        )

    @task(1)
    def pipeline(self):
        img_bytes = make_image_bytes(width=640, height=480)
        self.client.post(
            "/pipeline",
            files={"file": ("scene.jpg", img_bytes, "image/jpeg")},
            name="/pipeline",
        )
