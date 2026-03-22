import os

NOTIFY_TARGETS = {
    "EPITAPH_NODE": {
        "url": os.getenv("NODE_TRAINING_CALLBACK_URL"),
        "headers": {
            "X-API-Key": os.getenv("X_API_KEY"),
            "Content-Type": "application/json",
        },
    }
}
