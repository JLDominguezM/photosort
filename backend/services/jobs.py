import threading
from uuid import uuid4


class JobTracker:
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, name: str) -> str:
        job_id = str(uuid4())[:8]
        with self._lock:
            self.jobs[job_id] = {
                "name": name,
                "status": "running",
                "progress": 0,
                "total": 0,
                "result": None,
            }
        return job_id

    def update(self, job_id: str, progress: int, total: int):
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id]["progress"] = progress
                self.jobs[job_id]["total"] = total

    def complete(self, job_id: str, result: dict | None = None):
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["result"] = result

    def fail(self, job_id: str, error: str):
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["result"] = {"error": error}

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def list_all(self) -> dict:
        return dict(self.jobs)


tracker = JobTracker()
