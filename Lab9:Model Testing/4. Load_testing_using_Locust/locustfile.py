from locust import HttpUser, task
from random import choice
from string import ascii_uppercase

class User(HttpUser):
    @task
    def predict(self):
        payload = {
            "question": ''.join(choice(ascii_uppercase) for i in range(20)),
            "context": ''.join(choice(ascii_uppercase) for i in range(20))
        }
        self.client.post("/question_answering", json=payload)
